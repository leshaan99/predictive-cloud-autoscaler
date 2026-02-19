from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings
warnings.filterwarnings("ignore") # Ignore ARIMA convergence warnings

def adaptive_predict(cpu_history, network_history):
    """Dynamically switches between ARIMA and a heavier model based on volatility."""
    if len(cpu_history) < 10:
        return cpu_history[-1] # Need at least 10 data points
        
    # Calculate recent volatility
    recent_volatility = np.std(network_history[-5:])
    
    if recent_volatility < 10000: # Adjust this threshold based on actual network bytes
        print("Traffic is stable. Routing to ARIMA model...")
        model = ARIMA(cpu_history, order=(2,1,0)).fit()
        return max(0, model.forecast(steps=1).iloc[0])
    else:
        print("Flash Crowd detected! Routing to Heavy Model (LSTM logic)...")
        # Simulating LSTM output for now: predicts a 50% jump in CPU during a spike
        return min(100, cpu_history[-1] * 1.5)

def calculate_target_capacity(predicted_cpu, current_capacity):
    """Smart Scale-In (Anti-Flapping) and Scale-Out Logic."""
    scale_out_threshold = 70
    scale_in_threshold = 35
    
    if predicted_cpu > scale_out_threshold:
        print(f"Scale-Out triggered. Predicted CPU: {predicted_cpu}%")
        return current_capacity + 1
        
    elif predicted_cpu < scale_in_threshold and current_capacity > 1:
        # Anti-Flapping Check: What happens if we scale down?
        proposed_capacity = current_capacity - 1
        redistributed_cpu = (predicted_cpu * current_capacity) / proposed_capacity
        
        if redistributed_cpu >= scale_out_threshold:
            print(f"Anti-Flapping: Scaling down would spike CPU to {redistributed_cpu}%. Aborting.")
            return current_capacity 
            
        print(f"Scale-In approved. Predicted redistributed CPU: {redistributed_cpu}%")
        return proposed_capacity
        
    return current_capacity