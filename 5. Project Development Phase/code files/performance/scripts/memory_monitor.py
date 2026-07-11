import psutil
import time
import csv
import os
import argparse

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports', 'memory_metrics.csv')

def monitor_memory(duration_sec=60, interval_sec=1):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Memory_Usage(%)', 'Memory_MB', 'Peak_Memory_MB'])
        
        peak_memory = 0
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            mem = psutil.virtual_memory()
            mem_mb = mem.used / (1024 * 1024)
            if mem_mb > peak_memory:
                peak_memory = mem_mb
            writer.writerow([time.strftime('%H:%M:%S'), mem.percent, round(mem_mb, 2), round(peak_memory, 2)])
            time.sleep(interval_sec)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=60)
    args = parser.parse_args()
    monitor_memory(duration_sec=args.duration)
