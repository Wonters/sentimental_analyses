import numpy as np
import joblib
from pathlib import Path
from abc import ABC
from typing import Union
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import logging
import torch.nn.functional as F
import mlflow
from .mixins import TorchModelTrainMixin
from .torch_models import LSTMTorchNN
from .dataset import TweetDataset

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    logger.info("Using CUDA")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    logger.info("Using MPS")
else:
    DEVICE = torch.device("cpu")
    logger.info("Using CPU")

SENTIMENT_LABELS = {
    0: "😡 unsatisfy",
    4: "😊 satisfy",
}


class BaseModelABC(ABC):
    """
    Base class to train and predict on a dataset and register data on MLFLow
    """

    checkpoint: str = ""
    tokenizer = None
    epoch: int = 100
    dataset_class = None
    tokenizer_class = None
    batch_size = 32

    def __init__(self, x_train=None, y_train=None, x_val=None, y_val=None):
        self.load_checkpoint()
        self.dataset = None
        self.name = self.__class__.__name__
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.dataset = self.dataset_class(self.tokenizer, x_train, y_train)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

    def log_metrics(self):
        """
        Compute metrics for MLFlow here
        """
        y_pred = self.model.predict(self.preprocessing(self.x_val))
        report = classification_report(self.y_val, y_pred, output_dict=True)
        data = {}
        for label, scores in report.items():
            if isinstance(scores, dict):
                for metric, value in scores.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

    def save(self):
        """
        Save the model on disk
        """
        Path(self.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.checkpoint)
        joblib.dump(self.tokenizer, self.checkpoint_tokenizer)

    def init_items(self):
        """
        Initialize the items for the model
        """
        self.model = None
        self.tokenizer = None

    def init_mlflow(self):
        self.run = mlflow.start_run(run_name=self.name)
        self.run_id = self.run.info.run_id

    def load_checkpoint(self) -> object:
        """
        Logic to load the model from a checkpoint or create a new one
        """
        if Path(self.checkpoint).parent.exists():
            self.model = joblib.load(self.checkpoint)
            self.tokenizer = joblib.load(self.checkpoint_tokenizer)
        else:
            self.init_items()

    def preprocessing(self, data):
        """
        Preprocess the input data here
        """
        return self.tokenizer.transform(data)

    def confusion_matrix(self):
        with NamedTemporaryFile(suffix=".png") as f:
            conf_mat = confusion_matrix(self.y_train, self.predict(self.x_train))
            sns.heatmap(
                conf_mat,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.x_train.keys(),
                yticklabels=self.y_train.keys(),
            )
            plt.xlabel("Cluster prédits")
            plt.ylabel("Cluster réels")
            plt.savefig("test.png")
            # mlflow.log_artifact(f.name)

    def train(self):
        """
        Train the model here
        """
        mlflow.set_tag("model_type", self.name)
        mlflow.log_param("max_iter", self.epoch)
        mlflow.log_params(self.model.get_params())
        mlflow.sklearn.log_model(self.model, self.name)
        self.log_metrics()
        self.confusion_matrix()
        mlflow.log_artifact(self.checkpoint)
        mlflow.end_run()

    def predict(self, x: Union[pd.Series, np.ndarray]):
        """
        Predict the sentiment of the input data
        """
        predicted_class = self.model.predict(self.preprocessing(x))
        # logger.info(f"predicted {x.shape}")
        return predicted_class


class RandomForestModel(BaseModelABC):
    """
    Using a Random Forest model to predict sentiment on tweets
    """

    checkpoint = "checkpoints/randomforest/model.pkl"
    checkpoint_tokenizer = "checkpoints/randomforest/tokenizer.pkl"
    dataset_class = TweetDataset
    tokenizer_class = TfidfVectorizer

    def log_metrics(self):
        """Callback pour logger les métriques dans MLflow"""
        super().log_metrics()
        mlflow.log_metric("oob_score", self.model.oob_score_)

    def init_items(self):
        self.model = RandomForestClassifier(
            n_estimators=1000,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            oob_score=True,
            warm_start=True,
        )
        self.tokenizer = self.tokenizer_class(
            max_features=1000, ngram_range=(1, 2), binary=True
        )

    def train(self):
        """
        Train the Random Forest model with progress tracking
        """
        self.init_mlflow()
        try:
            # Vectorisation du texte
            X_train = self.tokenizer.fit_transform(self.x_train)
            self.model.fit(X_train, self.y_train)
            logger.info(f"Score OOB: {self.model.oob_score_:.4f}")
            self.save()
            super().train()
            # self.model.n_estimators += 10
        except Exception as e:
            logger.error(f"Erreur pendant l'entraînement: {str(e)}")
            raise


class LightGBMModel(BaseModelABC):
    """
    Using a LightGBM model to predict sentiment on tweets
    """

    checkpoint = "checkpoints/lightgbm/model.pkl"
    checkpoint_tokenizer = "checkpoints/lightgbm/tokenizer.pkl"
    dataset_class = TweetDataset
    tokenizer_class = TfidfVectorizer

    def log_metrics(self):
        """Callback pour logger les métriques dans MLflow"""
        super().log_metrics()
        mlflow.log_metric("oob_score", self.model.oob_score_)

    def init_items(self):
        self.model = LGBMClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        )
        self.tokenizer = self.tokenizer_class(
            max_features=1000, ngram_range=(1, 2), binary=True
        )

    def train(self):
        """
        Train the Random Forest model with progress tracking
        """
        self.init_mlflow()
        try:
            # Vectorisation du texte
            X_train = self.tokenizer.fit_transform(self.x_train)
            self.model.fit(X_train, self.y_train)
            self.save()
            super().train()
            # self.model.n_estimators += 10
        except Exception as e:
            logger.error(f"Erreur pendant l'entraînement: {str(e)}")
            raise


class LogisticRegressionModel(BaseModelABC):
    """
    A logistic regression model for sentiment analysis on tweets.

    This class implements a binary sentiment classifier using scikit-learn's LogisticRegression.
    It handles the full ML pipeline including:
    - Text preprocessing using TF-IDF vectorization
    - Model training with cross-validation
    - Prediction of sentiment polarity (positive/negative)
    - Model persistence via checkpoints
    - Performance metrics tracking with MLflow
    """

    checkpoint = "checkpoints/lr/logistic_regression.pkl"
    checkpoint_tokenizer = "checkpoints/lr/logistic_regression_tokenizer.pkl"
    tokenizer_class = TfidfVectorizer
    dataset_class = TweetDataset

    def init_items(self):
        """
        Initialize the items for the model
        """
        self.model = LogisticRegression(max_iter=1000, n_jobs=4, verbose=True)
        self.tokenizer = self.tokenizer_class()

    def train(self):
        self.init_mlflow()
        tokens = self.tokenizer.fit_transform(self.x_train)
        self.model.fit(tokens, self.y_train)
        self.save()
        super().train()


class BertModel(TorchModelTrainMixin, BaseModelABC):
    """
    Using a bert base mutilingual uncased sentiment to predict tweet sentiments
    """

    # Directory to save the model
    checkpoint = "checkpoints/bert"
    tokenizer_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    dataset_class = TweetDataset
    epoch = 1
    batch_size = 200
    out_features = 2
    lr = 2e-5
    device = DEVICE

    def load_checkpoint(self):
        if Path(self.checkpoint).exists():
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.checkpoint
            )
        else:

            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                ignore_mismatched_sizes=True,
                num_labels=self.out_features,
            )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.1
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def save(self):
        self.model.save_pretrained(self.checkpoint)
        self.tokenizer.save_pretrained(self.checkpoint)

    def predict(self, x: list):
        inputs = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).tolist()
        return predicted_class


class LSTMModel(TorchModelTrainMixin, BaseModelABC):
    checkpoint = "checkpoints/lstm"
    tokenizer_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    name = "LSTM"
    dataset_class = TweetDataset
    epoch = 1
    batch_size = 120
    # test with BCEWithLogitLoss -> 1 logit -> post traitment sigmoïd
    out_features = 1
    lr = 1e-4
    device = DEVICE
    # torch.nn.CrossEntropyLoss()

    @property
    def get_metrics(self) -> dict:
        for k, v in self.model.state_dict().items():
            mlflow.log_metric(k, v)

    def load_checkpoint(self):
        if Path(self.checkpoint).exists():
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = LSTMTorchNN(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=768,
            hidden_dim=256,
            output_dim=self.out_features,
            num_layers=1,
            bidirectional=True,
        )
        if Path(self.checkpoint).exists():
            if torch.backends.mps.is_available():
                logger.info("Load checkpoint on MPS")
                checkpoint = torch.load(
                    self.checkpoint + "/model.pth",
                    map_location={"cuda:0": "mps", "cuda": "mps"},
                )
            else:
                checkpoint = torch.load(self.checkpoint + "/model.pth")
            embedding_weights = {
                k: v
                for k, v in checkpoint.items()
                if k.startswith("embeddings.") or k.startswith("lstm.")
            }
            self.model.load_state_dict(embedding_weights, strict=False)
            self.model.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def save(self):
        # Create the parent directory saving tokenizer
        self.tokenizer.save_pretrained(self.checkpoint)
        torch.save(self.model.state_dict(), self.checkpoint + "/model.pth")

    def predict(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Appliquer sigmoïde sur les 4 prédictions
            probs = torch.sigmoid(outputs)
            # Convertir en classes (0 ou 1) en utilisant un seuil de 0.5
            predicted_classes = (probs > 0.5).int()
        return predicted_classes.tolist()


def load_data(path, shuffle=True):
    headers = ["target", "ids", "date", "flag", "user", "text"]
    df_tweets = pd.read_csv(path, names=headers, encoding="latin-1")
    # On prend target 0 negatif 1 positif
    df_tweets.loc[:, "target"] = df_tweets.target.map({0: int(0), 4: int(1)})
    train, test, y_train, y_test = train_test_split(
        df_tweets["text"],
        df_tweets["target"],
        test_size=0.2,
        random_state=42,
        shuffle=shuffle,
    )
    train, val, y_train, y_val = train_test_split(
        train, y_train, test_size=0.25, random_state=42
    )
    return train, test, val, y_train, y_test, y_val
