from locust import HttpUser, task, between

class APIPerformanceTest(HttpUser):
    wait_time = between(1, 2)

    @task(3)
    def test_health_endpoint(self):
        """Test the public health endpoint which verifies API routing speed."""
        self.client.get("/health")

    @task(1)
    def test_home_page(self):
        """Test the rendering of the public home page."""
        self.client.get("/")

    @task(2)
    def test_history_page(self):
        """Test the history endpoint (requires auth, will test redirect behavior)."""
        self.client.get("/history")
