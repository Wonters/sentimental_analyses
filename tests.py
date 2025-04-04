import pytest
from tornado.testing import AsyncHTTPTestCase
from ml import *
from server import TornadoApplication
class TestLogisticRegressionModel:
    file = "../data/training.1600000.processed.noemoticon.csv"

    @classmethod
    def setup_class(cls):
        cls.model = LogisticRegressionModel()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = load_data(cls.file)

    def test_train_logistic_regression(self):
        self.model.train(self.X_train, self.y_train)

    def test_predict_logistic_regression(self):
        result = self.model.predict(self.X_test)

class TestServer(AsyncHTTPTestCase):

    def get_app(self):
        return TornadoApplication()

    def _test_homepage(self):
        response = self.fetch('/api/predict')
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body, 'Hello, world')



