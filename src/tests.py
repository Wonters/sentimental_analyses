import pytest
from fastapi.testclient import TestClient
from .ml import LogisticRegressionModel, load_data, BertModel, LSTMModel
from .server import app

class TestLogisticRegressionModel:
    file = "data/tweets_test_train.csv"

    @classmethod
    def setup_class(cls):
        x_train, cls.x_test, cls.x_val, y_train, cls.y_test, cls.y_val = load_data(cls.file)
        cls.model = LogisticRegressionModel(x_train, y_train)

    def test_train_logistic_regression(self):
        self.model.train()

    def test_tokenizer(self):
        self.model.tokenizer.transform(self.model.x_train)

    def test_preprocessing(self):
        self.model.preprocessing(self.model.x_train)

    def test_mlflow(self):
        self.model.mlflow_record()

    def test_predict_logistic_regression(self):
        result = self.model.predict(self.x_test)


class TestBertModel:
    file = "data/tweets_test_train.csv"

    @classmethod
    def setup_class(cls):
        x_train, cls.x_test,cls.x_val, y_train, cls.y_test, cls.y_val = load_data(cls.file)
        cls.model = BertModel(x_train, y_train)

    def test_train(self):
        self.model.train()

    def test_tokenizer(self):
        self.model.tokenizer.transform(self.model.x_train)

    def test_preprocessing(self):
        self.model.preprocessing(self.model.x_train)

    def test_mlflow(self):
        self.model.mlflow_record()

    def test_predict(self):
        result = self.model.predict(list(self.x_test))

class TestLSTMModel:
    file = "data/tweets_test_train.csv"

    @classmethod
    def setup_class(cls):
        x_train, cls.x_test,cls.x_val, y_train, cls.y_test, cls.y_val = load_data(cls.file)
        cls.model = LSTMModel(x_train, y_train)

    def test_train(self):
        self.model.train()

    def test_tokenizer(self):
        self.model.tokenizer.transform(self.model.x_train)

    def test_preprocessing(self):
        self.model.preprocessing(self.model.x_train)

    def test_mlflow(self):
        self.model.mlflow_record()

    def test_predict(self):
        result = self.model.predict(self.x_test)

class TestServer:

    @classmethod
    def setup_class(cls):
        cls.client = TestClient(app)

    def test_main(self):
        rep = self.client.get("/")
        assert rep.status_code == 200

    def test_predict(self):
        response = self.client.post('/predict', json=[{"text": "hello world"}])
        assert response.status_code == 200
        assert response.json() == {}



