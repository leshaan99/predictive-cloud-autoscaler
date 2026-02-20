"""
main.py - FastAPI ML Autoscaling Controller
============================================
Research: ML-Driven Predictive Autoscaling with Waterfall Dependency Logic

Waterfall Map:
  Frontend spike -> automatically triggers Backend scale-up
  BEFORE Backend even registers the load (proactive, not reactive)
"""

from contextlib import asynccontextmanager
from collections import deque
from datetime import datetime

from fastapi import FastAPI
import asyncio
import boto3

from metrics import get_all_metrics
from logic import (
    adaptive_predict,
    calculate_target_capacity,
    train_random_forest
)

# ── Configuration ─────────────────────────────────────────
REGION         = "me-central-1"
LOOP_INTERVAL  = 15          # seconds
HISTORY_LENGTH = 50
BOOTSTRAP_PTS  = 20          # collect before first RF training

# ── Waterfall Dependency Map (Core Research Contribution) ─
# Key   = service that receives the load directly
# Value = list of downstream services that MUST scale proactively
DEPENDENCY_MAP = {
    "frontend": ["backend"],   # Frontend spike -> also scale Backend
}

# ASG names must match exactly what CloudFormation created
ASG_NAMES = {
    "frontend": "ML-Frontend-ASG",
    "backend":  "ML-Backend-ASG",
}

# ── Per-service time-series buffers ───────────────────────
histories = {
    svc: {
        "cpu":     deque(maxlen=HISTORY_LENGTH),
        "network": deque(maxlen=HISTORY_LENGTH),
        "memory":  deque(maxlen=HISTORY_LENGTH),
    }
    for svc in ASG_NAMES
}

# Audit log — exported via /log endpoint for research data
scaling_log = deque(maxlen=500)

# ── AWS Client ────────────────────────────────────────────
asg_client = boto3.client("autoscaling", region_name=REGION)


def get_current_capacity(asg_name: str):
    """Fetches current desired capacity for a given ASG."""
    try:
        resp   = asg_client.describe_auto_scaling_groups(AutoScalingGroupNames=[asg_name])
        groups = resp["AutoScalingGroups"]
        if not groups:
            print(f"[AWS] ASG '{asg_name}' not found!")
            return None
        return groups[0]["DesiredCapacity"]
    except Exception as e:
        print(f"[AWS] Error fetching capacity for {asg_name}: {e}")
        return None


def execute_scaling(asg_name: str, new_capacity: int) -> bool:
    """Sends scaling command to AWS."""
    try:
        asg_client.set_desired_capacity(
            AutoScalingGroupName=asg_name,
            DesiredCapacity=new_capacity,
            HonorCooldown=True
        )
        print(f"[AWS] '{asg_name}' scaled to {new_capacity} instances.")
        return True
    except Exception as e:
        print(f"[AWS] Scaling error for {asg_name}: {e}")
        return False


def scale_service(service: str, predicted_cpu: float, reason: str = "direct"):
    """
    Evaluates and executes a scaling decision for one service.
    reason = 'direct'    -> triggered by that service's own metrics
    reason = 'waterfall' -> triggered because an upstream service scaled
    """
    asg_name         = ASG_NAMES[service]
    current_capacity = get_current_capacity(asg_name)

    if current_capacity is None:
        return None, "SKIPPED"

    new_capacity = calculate_target_capacity(service, predicted_cpu, current_capacity)
    action       = "NONE"

    if new_capacity != current_capacity:
        success = execute_scaling(asg_name, new_capacity)
        if success:
            action = f"SCALE_OUT [{reason}]" if new_capacity > current_capacity else f"SCALE_IN [{reason}]"

    return new_capacity, action


# ── Main Autoscaling Loop ─────────────────────────────────
async def autoscaling_loop():
    print("=" * 60)
    print("  ML Waterfall Autoscaling Controller")
    print(f"  Services : {list(ASG_NAMES.keys())}")
    print(f"  Region   : {REGION}")
    print(f"  Interval : {LOOP_INTERVAL}s")
    print(f"  Waterfall: {DEPENDENCY_MAP}")
    print("=" * 60)

    iteration = 0

    while True:
        iteration += 1
        print(f"\n── Iteration {iteration} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ──")

        try:
            # ── Step 1: Fetch all service metrics ────────
            all_metrics = get_all_metrics()

            for svc, m in all_metrics.items():
                histories[svc]["cpu"].append(m["cpu"])
                histories[svc]["network"].append(m["network"])
                histories[svc]["memory"].append(m["memory"])
                print(f"  [{svc}] CPU: {m['cpu']}% | Net: {m['network']:.0f} B/s | Mem: {m['memory']}%")

            # ── Step 2: Bootstrap RF on each service ─────
            for svc in ASG_NAMES:
                if len(histories[svc]["cpu"]) == BOOTSTRAP_PTS:
                    print(f"[Bootstrap:{svc}] Running first RF training...")
                    train_random_forest(
                        svc,
                        list(histories[svc]["cpu"]),
                        list(histories[svc]["network"])
                    )

            # ── Step 3: Predict + Scale each service ─────
            scaled_out_services = []   # Track which services scaled OUT this cycle

            for svc in ASG_NAMES:
                cpu_hist = list(histories[svc]["cpu"])
                net_hist = list(histories[svc]["network"])

                predicted_cpu = adaptive_predict(svc, cpu_hist, net_hist)
                print(f"  [{svc}] Predicted CPU: {predicted_cpu}%")

                new_cap, action = scale_service(svc, predicted_cpu, reason="direct")

                # Track if this service scaled out
                if action.startswith("SCALE_OUT"):
                    scaled_out_services.append(svc)

                scaling_log.append({
                    "timestamp":    all_metrics[svc]["timestamp"],
                    "service":      svc,
                    "cpu":          all_metrics[svc]["cpu"],
                    "network":      all_metrics[svc]["network"],
                    "predicted_cpu": predicted_cpu,
                    "capacity":     new_cap,
                    "action":       action,
                    "trigger":      "direct",
                })

            # ── Step 4: WATERFALL CASCADE ─────────────────
            # This is the core research contribution:
            # If Frontend scaled out, proactively scale Backend
            # BEFORE Backend even shows high CPU load.
            for scaled_svc in scaled_out_services:
                downstream_services = DEPENDENCY_MAP.get(scaled_svc, [])

                for downstream in downstream_services:
                    if downstream not in ASG_NAMES:
                        continue

                    print(f"\n  [WATERFALL] '{scaled_svc}' scaled out "
                          f"-> Proactively scaling '{downstream}'")

                    # Use downstream's own current predicted CPU,
                    # but force a scale-out regardless via a synthetic spike signal
                    downstream_cpu_hist = list(histories[downstream]["cpu"])
                    downstream_net_hist = list(histories[downstream]["network"])

                    # Predict downstream load (it hasn't spiked YET — that's the point)
                    downstream_pred = adaptive_predict(
                        downstream, downstream_cpu_hist, downstream_net_hist
                    )

                    # Waterfall override: if upstream scaled, treat downstream
                    # as if it will reach SCALE_OUT_THRESHOLD (proactive logic)
                    waterfall_cpu = max(downstream_pred, 71.0)

                    new_cap, action = scale_service(
                        downstream, waterfall_cpu, reason="waterfall"
                    )

                    scaling_log.append({
                        "timestamp":     datetime.now().strftime("%H:%M:%S"),
                        "service":       downstream,
                        "cpu":           all_metrics[downstream]["cpu"],
                        "network":       all_metrics[downstream]["network"],
                        "predicted_cpu": waterfall_cpu,
                        "capacity":      new_cap,
                        "action":        action,
                        "trigger":       f"waterfall from {scaled_svc}",
                    })

        except Exception as e:
            print(f"[ERROR] {e}")

        await asyncio.sleep(LOOP_INTERVAL)


# ── FastAPI Lifespan ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(autoscaling_loop())
    yield
    task.cancel()


app = FastAPI(
    title="ML Waterfall Autoscaling Controller",
    description="Hybrid ARIMA/Random Forest Predictive Autoscaler with Waterfall Dependency Logic",
    version="2.0.0",
    lifespan=lifespan
)


# ── API Endpoints ─────────────────────────────────────────
@app.get("/")
def status():
    """Live system status for all services."""
    return {
        "status": "running",
        "services": {
            svc: {
                "data_points": len(histories[svc]["cpu"]),
                "last_cpu":    histories[svc]["cpu"][-1] if histories[svc]["cpu"] else None,
            }
            for svc in ASG_NAMES
        },
        "total_scaling_events": sum(
            1 for e in scaling_log if e["action"] != "NONE"
        ),
        "waterfall_events": sum(
            1 for e in scaling_log if "waterfall" in e.get("trigger", "")
        ),
    }


@app.get("/log")
def get_log():
    """Full scaling audit log — use this to export research data."""
    return {
        "total_iterations": len(scaling_log),
        "log": list(scaling_log)
    }


@app.get("/log/waterfall")
def get_waterfall_events():
    """Returns only waterfall-triggered scaling events for research analysis."""
    events = [e for e in scaling_log if "waterfall" in e.get("trigger", "")]
    return {
        "waterfall_event_count": len(events),
        "events": events
    }


@app.get("/metrics/history")
def get_history():
    """Raw time-series buffers for all services."""
    return {
        svc: {
            "cpu":     list(histories[svc]["cpu"]),
            "network": list(histories[svc]["network"]),
            "memory":  list(histories[svc]["memory"]),
        }
        for svc in ASG_NAMES
    }