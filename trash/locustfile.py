import os
import requests
from locust import HttpUser, between, task

from urllib.parse import urlparse
from sklearn.datasets import load_svmlight_file


TRAIN_DATASET_URL = 'https://raw.githubusercontent.com/dmlc/xgboost/master/demo/data/agaricus.txt.train'
TEST_DATASET_URL = 'https://raw.githubusercontent.com/dmlc/xgboost/master/demo/data/agaricus.txt.test'


def _download_file(url: str) -> str:
    parsed = urlparse(url)
    file_name = os.path.basename(parsed.path)
    file_path = os.path.join(os.getcwd(), file_name)
    
    res = requests.get(url)
    
    with open(file_path, 'wb') as file:
        file.write(res.content)
    
    return file_path

train_dataset_path = _download_file(TRAIN_DATASET_URL)
test_dataset_path = _download_file(TEST_DATASET_URL)
X_train, y_train = load_svmlight_file(train_dataset_path)
X_test, y_test = load_svmlight_file(test_dataset_path)
X_train = X_train.toarray()
X_test = X_test.toarray()

x_0 = X_test[0:3]

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
                "data": x_0.tolist()
                }
            ]
        }
        self.client.post(
            "/v2/models/mushroom-xgboost/versions/v0.1.0/infer", json=inference_request
        )
