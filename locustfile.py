"""
locustfile.py - Flash Crowd Traffic Simulator
===============================================
Run this on your LOCAL machine (not on AWS).
Install: pip install locust
Run:     locust -f locustfile.py --host=http://<FRONTEND-ALB-DNS>

Test Phases:
  0-2 min  : Stable baseline (ARIMA handles this)
  2-5 min  : Flash Crowd spike (triggers RF + Waterfall cascade)
  5-7 min  : Sudden drop (tests Anti-Flapping / Smart Scale-In)
"""

from locust import HttpUser, task, between, LoadTestShape


class WebUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def hit_frontend(self):
        self.client.get("/", catch_response=True)


class FlashCrowdShape(LoadTestShape):
    """
    Simulates a non-cyclical Flash Crowd — exactly the scenario
    AWS Predictive Scaling CANNOT handle because it hasn't seen it before.
    Your ARIMA/RF hybrid controller handles it in real time.
    """
    def tick(self):
        run_time = self.get_run_time()

        # Phase 1: Stable baseline — ARIMA should predict correctly
        if run_time < 120:
            return (50, 5)           # 50 users, 5 spawn/sec

        # Phase 2: Flash Crowd — triggers RF model + Waterfall to Backend
        elif run_time < 300:
            return (500, 100)        # Instant 10x spike

        # Phase 3: Sudden drop — tests Anti-Flapping scale-in logic
        elif run_time < 420:
            return (10, 5)

        return None                  # End test
