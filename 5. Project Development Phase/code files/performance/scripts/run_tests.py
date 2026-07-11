import os
import subprocess
import time
import signal
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)

def main():
    print("Starting Flask app in background...")
    flask_process = subprocess.Popen(
        [sys.executable, 'app.py'],
        cwd=PROJECT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait for flask to start
    time.sleep(5)
    
    print("Starting hardware monitor in background...")
    monitor_process = subprocess.Popen(
        [sys.executable, os.path.join(BASE_DIR, 'scripts', 'monitor.py'), '--duration', '60'],
        cwd=BASE_DIR
    )
    
    print("Running Locust Load Test (50 users, 60 seconds)...")
    locust_process = subprocess.Popen(
        [sys.executable, '-m', 'locust', '-f', 'locust/locustfile.py', '--headless', '-u', '50', '-r', '5', '--run-time', '60s', '--host', 'http://127.0.0.1:5000', '--csv', 'reports/locust'],
        cwd=BASE_DIR
    )
    
    locust_process.wait()
    monitor_process.wait()
    
    print("Stopping Flask app...")
    flask_process.terminate()
    flask_process.wait()
    print("Tests completed successfully!")

if __name__ == "__main__":
    main()
