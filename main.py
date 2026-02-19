from fastapi import FastAPI
import asyncio
import boto3
from metrics import get_fused_metrics
from logic import adaptive_predict, calculate_target_capacity
from datetime import datetime

app = FastAPI()

ASG_NAME = "ML-ASG"
REGION = "us-east-1" # Ensure this matches your deployment region

# In-memory storage for our time-series data
cpu_data = []
network_data = []

async def autoscaling_loop():
    print("Starting ML Autoscaling Controller Loop...")
    asg_client = boto3.client('autoscaling', region_name=REGION)
    
    while True:
        try:
            # 1. Fetch live metrics
            metrics = get_fused_metrics()
            cpu_data.append(metrics['cpu'])
            network_data.append(metrics['network'])
            
            # Keep only the last 50 data points to manage memory
            if len(cpu_data) > 50:
                cpu_data.pop(0)
                network_data.pop(0)
                
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Live CPU: {metrics['cpu']} | Network: {metrics['network']}")
            
            # 2. Predict Load
            predicted_cpu = adaptive_predict(cpu_data, network_data)
            
            # 3. Get Current Capacity & Calculate New Target
            response = asg_client.describe_auto_scaling_groups(AutoScalingGroupNames=[ASG_NAME])
            if not response['AutoScalingGroups']:
                print(f"ASG {ASG_NAME} not found!")
                await asyncio.sleep(15)
                continue
                
            current_capacity = response['AutoScalingGroups'][0]['DesiredCapacity']
            new_target = calculate_target_capacity(predicted_cpu, current_capacity)
            
            # 4. Actuate Scaling
            if new_target != current_capacity:
                print(f"Executing Scaling Action: Changing capacity from {current_capacity} to {new_target}")
                asg_client.set_desired_capacity(
                    AutoScalingGroupName=ASG_NAME,
                    DesiredCapacity=new_target,
                    HonorCooldown=True
                )
            
        except Exception as e:
            print(f"Error in operational loop: {e}")
            
        # Run the loop every 15 seconds (High-Frequency Research Standard)
        await asyncio.sleep(15)

@app.on_event("startup")
async def startup_event():
    # Start the background loop when the API starts
    asyncio.create_task(autoscaling_loop())

@app.get("/")
def read_root():
    return {"Status": "ML Autoscaler is running", "Current_Data_Points": len(cpu_data)}