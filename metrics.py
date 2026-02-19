import requests

PROMETHEUS_URL = 'http://localhost:9090/api/v1/query'

def get_fused_metrics():
    """Fetches high-frequency metrics for Multi-Metric Fusion."""
    # Fetch CPU Usage (Average across ASG)
    cpu_query = 'avg(rate(node_cpu_seconds_total{mode!="idle"}[1m])) * 100'
    
    # Fetch Network Traffic as a proxy for Request Rate (since we don't have a web app deployed yet)
    net_query = 'sum(rate(node_network_receive_bytes_total[1m]))' 
    
    try:
        cpu_res = requests.get(PROMETHEUS_URL, params={'query': cpu_query}, timeout=5).json()
        net_res = requests.get(PROMETHEUS_URL, params={'query': net_query}, timeout=5).json()
        
        cpu_val = float(cpu_res['data']['result'][0]['value'][1]) if cpu_res['data']['result'] else 0.0
        net_val = float(net_res['data']['result'][0]['value'][1]) if net_res['data']['result'] else 0.0
        
        return {"cpu": round(cpu_val, 2), "network": round(net_val, 2)}
    except Exception as e:
        print(f"Metrics fetch error: {e}")
        return {"cpu": 0.0, "network": 0.0}