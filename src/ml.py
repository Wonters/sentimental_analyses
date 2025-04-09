import numpy as np
import joblib
import time
from pathlib import Path
from abc import ABC
from typing import Union
from tempfile import NamedTemporaryFile
import matplotlib as plt
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import logging
import torch.nn.functional as F
from tqdm import tqdm
import mlflow
import gc

from .torch_models import LSTMTorchNN
from .dataset import TweetDataset

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    logger.info("Using CUDA")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    logger.info("Using MPS")
else:
    DEVICE = torch.device('cpu')
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

    def __init__(self, x_train=None, y_train=None):
        self.load_checkpoint()
        self.dataset = None
        self.name = self.__class__.__name__
        self.x_train = x_train
        self.y_train = y_train
        self.dataset = self.dataset_class(self.tokenizer, x_train, y_train)

    @property
    def metrics(self) -> dict:
        """
        Compute metrics for MLFlow here
        """

    def save(self):
        """
        Save the model on disk
        """

    def load_checkpoint(self) -> object:
        """
        Logic to load the model from a checkpoint or create a new one
        """

    def mlflow_record(self, **kwargs):
        """
        MLFlow base implementation to register the model and a confusion matrix
        """
        with mlflow.start_run():
            mlflow.set_tag("model_type", self.name)
            mlflow.log_param("max_iter", self.epoch)
            mlflow.log_params(self.model.get_params())
            for k, v in self.metrics.items():
                mlflow.log_metric(k, v)
            mlflow.sklearn.log_model(self.model, self.name)
            with NamedTemporaryFile(suffix=".png") as f:
                conf_mat = confusion_matrix(self.y_train, self.predict(self.x_train))
                sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                            xticklabels=self.x_train.keys(),
                            yticklabels=self.y_train.keys())
                plt.xlabel('Cluster prédits')
                plt.ylabel('Cluster réels')
                plt.savefig(f.name)
                mlflow.log_artifact(f.name)
            mlflow.log_artifact(self.checkpoint)
        return 0

    def preprocessing(self, data):
        """
        Preprocess the input data here
        """

    def train(self):
        """
        Train the model here
        """

    def predict(self, x: Union[pd.Series, np.ndarray]):
        """
        Method to predict on new data
        """


class TorchModelTrainMixin:
    """
    Mixin to use with BaseModelABC
    """
    checkpoint: str = ""
    lr: float = 2e-5

    def _train_batch(self, x, y):
        inputs = self.tokenizer(x,
                                return_tensors="pt",
                                truncation=True,
                                padding=True)
        inputs.to(DEVICE)
        labels = y.to(DEVICE)
        self.optimizer.zero_grad()
        outputs = self.model(**inputs)
        try:
            loss = self.criterion(outputs.logits, labels)
        except AttributeError:
            loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        del inputs, labels, outputs, loss
        gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
        time.sleep(0.2)

    def train(self):
        self.model.train()
        self.model.to(DEVICE)
        for epoch in tqdm(range(self.epoch)):
            for tweets, labels in self.dataloader:
                try:
                    self._train_batch(tweets, labels)
                except RuntimeError as e:
                    logger.error(e)
                    del tweets, labels, self.optimizer
                    gc.collect()
                    if torch.backends.mps.is_available(): torch.mps.empty_cache()
                    time.sleep(0.2)
                    self.save()
                    self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint)
                    self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
                    self.model.to(DEVICE)
                    self.model.train()
                    continue
                if torch.backends.mps.is_available():
                    logger.info(f"MPS allocated memory: {torch.mps.driver_allocated_memory()}")
                if torch.cuda.is_available():
                    logger.info(f"CUDA allocated memory: {torch.cuda.memory_allocated()}")
        self.save()


class LogisticRegressionModel(BaseModelABC):
    """
    This class is used to train and predict on a dataset
    It uses a LogisticRegression model
    Usually used to predict sentiment on tweets
    """
    checkpoint = "checkpoints/logistic_regression.pkl"
    checkpoint_tokenizer = "checkpoints/Logistic_regression_tokenizer.pkl"
    tokenizer_class = TfidfVectorizer
    name = "LogisticRegression"
    dataset_class = TweetDataset

    def __init__(self, x_train=None, y_train=None):
        super().__init__(x_train, y_train)
        self.tokenizer = self.tokenizer_class()

    def load_checkpoint(self):
        if Path(self.checkpoint).exists():
            self.model = joblib.load(self.checkpoint)
        else:
            self.model = LogisticRegression()
        if Path(self.checkpoint_tokenizer).exists():
            self.tokenizer = joblib.load(self.checkpoint_tokenizer)

    def save(self):
        joblib.dump(self.model, self.checkpoint)
        joblib.dump(self.tokenizer, self.checkpoint_tokenizer)

    # @lru_cache(maxsize=10)
    def preprocessing(self, data):
        """"""
        return self.tokenizer.transform(data)

    @property
    def metrics(self) -> dict:
        y_pred = self.model.predict(self.preprocessing(self.x_train))
        report = classification_report(self.y_train, y_pred, output_dict=True)
        data = {}
        for label, scores in report.items():
            if isinstance(scores, dict):
                for metric, value in scores.items():
                    data[f"{label}_{metric}"] = value
        return data

    def predict(self, x: Union[pd.Series, np.ndarray]):
        predicted_class = self.model.predict(self.preprocessing(x))
        logger.info(f"predicted {x.shape}")
        return predicted_class  # [SENTIMENT_LABELS[p] for p in predicted_class]

    def train(self):
        tokens = self.tokenizer.fit_transform(self.x_train)
        self.model.fit(tokens, self.y_train)
        self.save()
        self.mlflow_record()


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

    def __init__(self, x_train=None, y_train=None):
        super().__init__(x_train, y_train)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True)

    def load_checkpoint(self):
        if Path(self.checkpoint).exists():
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                                            ignore_mismatched_sizes=True,
                                                                            num_labels=self.out_features)
    def save(self):
        self.model.save_pretrained(self.checkpoint)
        self.tokenizer.save_pretrained(self.checkpoint)

    def predict(self, x: list):
        inputs = self.tokenizer(x,
                                return_tensors='pt',
                                truncation=True,
                                padding=True)
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
    batch_size = 200
    out_features = 2
    lr = 2e-5

    def __init__(self, x_train=None, y_train=None):
        super().__init__(x_train, y_train)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True)

    def load_checkpoint(self):
        if Path(self.checkpoint).exists():
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            self.model = torch.load(self.checkpoint+"/model.pt")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.model = LSTMTorchNN(vocab_size=self.tokenizer.vocab_size,
                                   embedding_dim=100,
                                   hidden_dim=128,
                                   output_dim=self.out_features,
                                   num_layers=1)

    def save(self):
        # Create the parent directory saving tokenizer
        self.tokenizer.save_pretrained(self.checkpoint)
        torch.save(self.model, self.checkpoint+"/model.pt")


    def predict(self, x):
        inputs = self.tokenizer(x,
                                return_tensors='pt',
                                truncation=True,
                                padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
        return SENTIMENT_LABELS[predicted_class]


def load_data(path):
    headers = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df_tweets = pd.read_csv(path, names=headers, encoding="latin-1")
    # On prend target 0 negatif 1 positif
    df_tweets.loc[:, 'target'] = df_tweets.target.map({0: 0, 4: 1})
    train, test, y_train, y_test = train_test_split(df_tweets['text'],
                                                    df_tweets['target'],
                                                    test_size=0.2,
                                                    random_state=42)
    train, val, y_train, y_val = train_test_split(
        train, y_train, test_size=0.25, random_state=42)
    return train, test, val, y_train, y_test, y_val
