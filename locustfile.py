from locust import HttpUser, task, between
import os

class ClickbaitImageClassifierUser(HttpUser):
    wait_time = between(1, 3)  # seconds between tasks

    @task
    def predict_single_image(self):
        # Path to a sample image for testing
        image_path = "sample.jpg"
        if not os.path.exists(image_path):
            # Create a dummy image if not present
            from PIL import Image
            img = Image.new("RGB", (128, 128), color="white")
            img.save(image_path)

        with open(image_path, "rb") as img_file:
            files = {"file": ("sample.jpg", img_file, "image/jpeg")}
            response = self.client.post("/predict", files=files)
            print(response.text)

    # @task
    # def retrain_model(self):
    #     response = self.client.post("/retrain")
    #     print(response.text)

    # # You can add more tasks for batch prediction, etc.
