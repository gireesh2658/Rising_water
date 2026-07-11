import os
import subprocess
import time
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)

def main():
    print("Starting Flask App (Before Optimization)...")
    flask_proc = subprocess.Popen([sys.executable, 'app.py'], cwd=PROJECT_DIR)
    time.sleep(5)
    
    print("\n--- Running Before Optimization Test (50 users, 60s) ---")
    cmd = [
        sys.executable, '-m', 'locust', '-f', 'locust/locustfile.py', '--headless',
        '-u', '50', '-r', '10', '--run-time', '60s',
        '--host', 'http://127.0.0.1:5000',
        '--csv', 'reports/before_optimization'
    ]
    subprocess.run(cmd, cwd=BASE_DIR)
    
    print("Stopping Flask App...")
    flask_proc.terminate()
    flask_proc.wait()

if __name__ == "__main__":
    main()
