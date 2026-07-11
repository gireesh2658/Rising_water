import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_graphs():
    print("Generating Performance Graphs...")
    
    screenshots_dir = os.path.join(BASE_DIR, 'screenshots')
    reports_dir = os.path.join(BASE_DIR, 'reports')
    os.makedirs(screenshots_dir, exist_ok=True)
    
    # 1. System Metrics
    cpu_metrics_path = os.path.join(reports_dir, 'cpu_metrics.csv')
    mem_metrics_path = os.path.join(reports_dir, 'memory_metrics.csv')
    
    if os.path.exists(cpu_metrics_path):
        try:
            df_cpu = pd.read_csv(cpu_metrics_path)
            df_cpu['TimeIndex'] = range(len(df_cpu))
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=df_cpu, x='TimeIndex', y='CPU_Usage(%)', label='CPU Usage (%)', color='red')
            plt.title('CPU Utilization During Load Test')
            plt.xlabel('Time (Seconds)')
            plt.ylabel('Percentage (%)')
            plt.ylim(0, 100)
            plt.savefig(os.path.join(screenshots_dir, 'cpu_utilization.png'))
            plt.close()
        except Exception as e:
            print(f"Error generating CPU graph: {e}")

    if os.path.exists(mem_metrics_path):
        try:
            df_mem = pd.read_csv(mem_metrics_path)
            df_mem['TimeIndex'] = range(len(df_mem))
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=df_mem, x='TimeIndex', y='Memory_Usage(%)', label='Memory Usage (%)', color='blue')
            plt.title('Memory Utilization During Load Test')
            plt.xlabel('Time (Seconds)')
            plt.ylabel('Percentage (%)')
            plt.ylim(0, 100)
            plt.savefig(os.path.join(screenshots_dir, 'memory_utilization.png'))
            plt.close()
        except Exception as e:
            print(f"Error generating Memory graph: {e}")
            
    # 2. Locust Metrics
    locust_stats_path = os.path.join(reports_dir, 'concurrent_50_users_stats.csv')
    locust_history_path = os.path.join(reports_dir, 'concurrent_50_users_stats_history.csv')
    
    if os.path.exists(locust_history_path):
        try:
            df_hist = pd.read_csv(locust_history_path)
            
            # Throughput
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=df_hist, x='Timestamp', y='Requests/s', label='Throughput', color='green')
            plt.title('Requests Per Second (Throughput)')
            plt.xlabel('Timestamp')
            plt.ylabel('Req/s')
            plt.xticks([])
            plt.savefig(os.path.join(screenshots_dir, 'throughput.png'))
            plt.close()
            
            # Latency (Response Time)
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=df_hist, x='Timestamp', y='Total Average Response Time', label='Average', color='blue')
            sns.lineplot(data=df_hist, x='Timestamp', y='Total Median Response Time', label='Median', color='orange')
            plt.title('Response Time / Latency Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Milliseconds (ms)')
            plt.xticks([])
            plt.savefig(os.path.join(screenshots_dir, 'latency.png'))
            plt.close()
        except Exception as e:
            print(f"Error generating Locust graphs: {e}")
            
    print(f"Graphs generated successfully in {screenshots_dir}")

if __name__ == "__main__":
    generate_graphs()
