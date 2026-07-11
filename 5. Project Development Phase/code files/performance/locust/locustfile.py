from locust import HttpUser, task, between

class FloodPredictionUser(HttpUser):
    wait_time = between(1, 5)

    @task(3)
    def index_page(self):
        self.client.get("/")

    @task(2)
    def predict_page(self):
        self.client.get("/Predict")
        
    @task(2)
    def history_page(self):
        self.client.get("/history")

    @task(10)
    def valid_prediction(self):
        payload = {
            "temp": 30.5,
            "humidity": 78.0,
            "cloud_cover": 42.0,
            "annual": 3200.5,
            "jan_feb": 15.2,
            "mar_may": 210.4,
            "jun_sep": 2500.5,
            "oct_dec": 450.3
        }
        self.client.post("/predict", json=payload)
        
    @task(1)
    def missing_inputs_prediction(self):
        # Missing 'annual'
        payload = {
            "temp": 30.5, 
            "humidity": 78.0,
            "cloud_cover": 42.0,
            "jan_feb": 15.2,
            "mar_may": 210.4,
            "jun_sep": 2500.5,
            "oct_dec": 450.3
        }
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code == 400:
                response.success()

    @task(1)
    def large_inputs_prediction(self):
        # Out of bounds 'temp'
        payload = {
            "temp": 150.0, 
            "humidity": 150.0,
            "cloud_cover": 150.0,
            "annual": 999999.0,
            "jan_feb": 999999.0,
            "mar_may": 999999.0,
            "jun_sep": 999999.0,
            "oct_dec": 999999.0
        }
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code == 400:
                response.success()
                
    @task(1)
    def repeated_predictions(self):
        # Fire 5 predictions in rapid succession
        payload = {
            "temp": 25.0,
            "humidity": 80.0,
            "cloud_cover": 50.0,
            "annual": 2500.0,
            "jan_feb": 10.0,
            "mar_may": 200.0,
            "jun_sep": 2000.0,
            "oct_dec": 290.0
        }
        for _ in range(5):
            self.client.post("/predict", json=payload)
