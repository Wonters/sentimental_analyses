import lightgbm
import fastapi
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
from ..server import PredictApp

class BaseTest:
    file = "data/tweets_test_train.csv"
    class_model = None

    @classmethod
    def setup_class(cls):
        df = load_data(cls.file)
        cls.model = cls.class_model(dataset=df, tracking=False)

    def test_train(self):
        self.model.train()

    def test_tokenizer(self):
        self.model.tokenizer.transform(self.model.x_train)

    def test_preprocessing(self):
        self.model.preprocessing(self.model.x_train)


class TestLogisticRegressionModel(BaseTest):
    class_model = LogisticRegressionModel

    def test_predict(self):
        result = self.model.predict(list(self.model.x_test))
        assert len(result.tolist()) == 6


class TestLightGBMModel(BaseTest):
    class_model = LightGBMModel

    def test_train(self):
        self.model.train()


class TestBertModel(BaseTest):
    class_model = BertModel

    def test_predict(self):
        result = self.model.predict(list(self.model.x_test))
        assert len([r['prediction'] for r in result]) == 6

    def test_tokenizer(self):
        """"""

    def test_confusion_matrix(self):
        self.model.confusion_matrix()

    def test_optuna_train(self):
        self.model.optuna_train(n_trials=1)

class TestRobertaModel(BaseTest):
    class_model = RobertaModel

    def test_optuna_train(self):
        """"""

    def test_tokenizer(self):
        """"""

    def test_predict(self):
        result = self.model.predict(list(self.model.x_test))
        assert len([r['prediction'] for r in result]) == 6


class TestLSTMModel(BaseTest):
    class_model = LSTMModel

    def test_size_vocab(self):
        print(self.model.tokenizer.vocab_size)

    def test_tokenizer(self):
        """"""

    def test_predict(self):
        result = self.model.predict(list(self.model.x_test))
        assert type(result) == list
        assert len(result) == 6


class TestRandomForestModel(BaseTest):
    class_model = RandomForestModel


class TestServer:

    @classmethod
    def setup_class(cls):
        app = fastapi.FastAPI()
        test_app = PredictApp(save_db=False)
        app.include_router(test_app.router)
        cls.client = TestClient(app)

    def test_main(self):
        rep = self.client.get("/")
        assert rep.status_code == 200

    def test_predict(self):
        response = self.client.post("/predict", json=[{"text": "hello world"}])
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "processing"
