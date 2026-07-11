import psutil
import time
import csv
import os

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'reports', 'system_metrics.csv')

def monitor_system(duration_sec=60, interval_sec=1):
    print(f"Starting hardware monitor for {duration_sec} seconds...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'CPU_Usage(%)', 'Memory_Usage(%)', 'Memory_MB'])
        
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            mem_mb = mem.used / (1024 * 1024)
            writer.writerow([time.strftime('%H:%M:%S'), cpu, mem.percent, round(mem_mb, 2)])
            time.sleep(interval_sec)
            
    print(f"Monitoring complete. Results saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=60, help='Duration in seconds')
    args = parser.parse_args()
    monitor_system(duration_sec=args.duration)
