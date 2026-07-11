import psutil
import time
import csv
import os
import argparse

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports', 'cpu_metrics.csv')

def monitor_cpu(duration_sec=60, interval_sec=1):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'CPU_Usage(%)'])
        
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            cpu = psutil.cpu_percent(interval=None)
            writer.writerow([time.strftime('%H:%M:%S'), cpu])
            time.sleep(interval_sec)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=60)
    args = parser.parse_args()
    monitor_cpu(duration_sec=args.duration)
