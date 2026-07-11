import os
import subprocess
import time
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)

def run_flask():
    print("Starting Flask App...")
    return subprocess.Popen(
        [sys.executable, 'app.py'],
        cwd=PROJECT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def run_locust(users, spawn_rate, duration, report_name):
    print(f"\n--- Running Test: {report_name} ({users} users, {duration}) ---")
    
    # Start monitor
    monitor = subprocess.Popen([sys.executable, os.path.join(BASE_DIR, 'scripts', 'cpu_monitor.py'), '--duration', str(int(duration.replace('s','')))])
    
    cmd = [
        sys.executable, '-m', 'locust', '-f', 'locust/locustfile.py', '--headless',
        '-u', str(users), '-r', str(spawn_rate), '--run-time', duration,
        '--host', 'http://127.0.0.1:5000',
        '--csv', f'reports/{report_name}'
    ]
    subprocess.run(cmd, cwd=BASE_DIR)
    monitor.wait()

def main():
    flask_proc = run_flask()
    time.sleep(5)  # Wait for startup
    
    try:
        # Phase 5 Executions
        run_locust(10, 2, '30s', 'load_10_users')
        run_locust(25, 5, '60s', 'load_25_users')
        run_locust(50, 10, '60s', 'concurrent_50_users')
        run_locust(100, 100, '30s', 'spike_100_users')
        
    finally:
        print("Stopping Flask App...")
        flask_proc.terminate()
        flask_proc.wait()
        print("All tests completed successfully!")

if __name__ == "__main__":
    main()
