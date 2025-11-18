import numpy as np
from locust import HttpUser, between, task
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)
x_0 = X_test[0:1]

class MNISTUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict(self):
        inference_request = {
            "inputs": [
                {
                    "name": "predict",
                    "shape": x_0.shape,
                    "datatype": "FP32",
                    "data": x_0.tolist(),
                }
            ]
        }
        self.client.post(
            "/v2/models/mnist-svm/versions/v0.1.0/infer", json=inference_request
        )
