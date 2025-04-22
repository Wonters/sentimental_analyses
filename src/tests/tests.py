import lightgbm
import pytest
import time
from fastapi.testclient import TestClient
from ..ml import (
    LogisticRegressionModel,
    load_data,
    BertModel,
    RobertaModel,
    LSTMModel,
    RandomForestModel,
    LightGBMModel,
)
from transformers import PreTrainedModel
from ..server import app


class BaseTest:
    file = "data/tweets_test_train.csv"
    class_model = None

    @classmethod
    def setup_class(cls):
        df = load_data(cls.file)
        cls.model = cls.class_model(dataset=df)

    def test_train(self):
        self.model.train()

    def test_tokenizer(self):
        self.model.tokenizer.transform(self.model.x_train)

    def test_preprocessing(self):
        self.model.preprocessing(self.model.x_train)


class TestLogisticRegressionModel(BaseTest):
    class_model = LogisticRegressionModel

    def test_predict(self):
        result = self.model.predict(list(self.x_test))
        print(result, self.y_test.values)
        assert result.tolist() == [0, 1, 0, 0, 0, 0]


class TestLightGBMModel(BaseTest):
    class_model = LightGBMModel

    def test_train(self):
        self.model.train()


class TestBertModel(BaseTest):
    class_model = BertModel

    def test_predict(self):
        result = self.model.predict(list(self.x_test))
        assert result == [1, 1, 0, 0, 0, 0]

    def test_confusion_matrix(self):
        self.model.confusion_matrix()

    def test_optuna_train(self):
        self.model.optuna_train(n_trials=5)

class TestRobertaModel(BaseTest):
    class_model = RobertaModel

    def test_optuna_train(self):
        self.model.optuna_train(n_trials=5)

    def test_predict(self):
        result = self.model.predict(list(self.x_test))
        assert result == [1, 1, 0, 0, 0, 0]


class TestLSTMModel(BaseTest):
    class_model = LSTMModel

    def test_size_vocab(self):
        print(self.model.tokenizer.vocab_size)


    def test_predict(self):
        result = self.model.predict(list(self.x_test))
        assert result.tolist() == [1, 0, 0, 0, 0, 0]


class TestRandomForestModel(BaseTest):
    class_model = RandomForestModel


class TestServer:

    @classmethod
    def setup_class(cls):
        cls.client = TestClient(app)

    def test_main(self):
        rep = self.client.get("/")
        assert rep.status_code == 200

    def test_predict(self):
        response = self.client.post("/predict", json=[{"text": "hello world"}])
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "processing"
        task_id = payload["task_id"]
        response = self.client.get(f"/get_result/{task_id}")
        payload = response.json()
        while payload["status"] == "processing":
            response = self.client.get(f"/get_result/{task_id}")
            time.sleep(1)
            payload = response.json()
            print(payload)
        # assert response.json() == {}
