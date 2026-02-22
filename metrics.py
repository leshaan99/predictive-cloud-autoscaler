"""
metrics.py - Multi-Metric Fusion Data Fetcher
==============================================
Fetches per-service metrics from Prometheus every 15 seconds.
Supports Frontend and Backend ASGs independently.
"""

import requests
from datetime import datetime

PROMETHEUS_URL = "http://localhost:9090/api/v1/query"


def _query(promql):
    """Sends a PromQL query and returns the float result."""
    try:
        response = requests.get(
            PROMETHEUS_URL,
            params={"query": promql},
            timeout=5
        )
        response.raise_for_status()
        results = response.json()["data"]["result"]
        if results:
            return round(float(results[0]["value"][1]), 4)
        return 0.0
    except Exception as e:
        print(f"[Metrics] Query error: {e}")
        return 0.0


def get_service_metrics(service: str) -> dict:
    """
    Fetches CPU, network, and memory for a specific service label.
    Uses correct PromQL: 100 - idle = actual CPU usage
    """
    # FIXED: correct CPU query using idle mode subtraction
    cpu = _query(
        f'100 - (avg by (instance) (rate(node_cpu_seconds_total{{service="{service}",mode="idle"}}[1m])) * 100)'
    )

    network = _query(
        f'sum(rate(node_network_receive_bytes_total{{service="{service}"}}[1m]))'
    )

    memory = _query(
        f'(1 - (avg(node_memory_MemAvailable_bytes{{service="{service}"}}) '
        f'/ avg(node_memory_MemTotal_bytes{{service="{service}"}}))) * 100'
    )

    return {
        "service":   service,
        "cpu":       cpu,
        "network":   network,
        "memory":    memory,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }


def get_all_metrics() -> dict:
    """Fetches metrics for all services in one call."""
    return {
        "frontend": get_service_metrics("frontend"),
        "backend":  get_service_metrics("backend"),
    }


if __name__ == "__main__":
    print("Testing Prometheus connection...")
    data = get_all_metrics()
    for svc, m in data.items():
        print(f"[{svc}] CPU: {m['cpu']}% | Network: {m['network']} B/s | Memory: {m['memory']}%")