import numpy as np
import joblib
import os 
import re
import string
from functools import partial
from pathlib import Path
from abc import ABC
from typing import Union
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import lightgbm as lgm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup 
import pandas as pd
from skopt import BayesSearchCV, gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
import logging
from transformers import PreTrainedModel
import mlflow
from mlflow.data.pandas_dataset import from_pandas
from mlflow.models import infer_signature
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

#DEVICE = torch.device("cpu")

SENTIMENT_LABELS = {
    0: "😡 unsatisfy",
    4: "😊 satisfy",
}

# todo: set the tracking uri to the mlflow ui in the container 
# todo: problem with paths container vs system when run with pytest
#os.environ['MLFLOW_TRACKING_URI']='http://localhost:5001'


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
    #artifact_uri = "file:///app/mlruns"

    def __init__(self, dataset: pd.DataFrame):
        self.original_dataset = dataset
        self.x_train, self.x_test, self.x_val, self.y_train, self.y_test, self.y_val = split_data(dataset)
        self.load_checkpoint()
        self.dataset = None
        self.name = self.__class__.__name__
        self.dataset = self.dataset_class(self.tokenizer, self.x_train, self.y_train)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

    def log_metrics(self):
        """
        Compute metrics for MLFlow here
        """
        y_pred = self.predict(list(self.x_val))
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

    def init_mlflow(self, name:str = ""):
        self.run = mlflow.start_run(run_name=name if name else self.name)
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
            conf_mat = confusion_matrix(self.y_train, self.predict(list(self.x_train)))
            group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
            group_counts = [f"{value: 0.0f}" for value in conf_mat.flatten()]
            group_percentages = [f"{value:.2%}" for value in conf_mat.flatten() / np.sum(conf_mat)]
            labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
            labels = np.asarray(labels).reshape(2, 2)
            sns.heatmap(
                conf_mat,
                annot=labels,
                fmt="",
                cmap="Blues"
            )
            plt.xlabel("Cluster prédits")
            plt.ylabel("Cluster réels")
            plt.savefig(f.name)
            plt.close()
            mlflow.log_artifact(f.name, "confusion_matrix.png")

    def train(self):
        """
        Train the model here
        """
        self.save()
        mlflow.set_tag("model_type", self.name)
        self.log_metrics()
        self.confusion_matrix()
        mlflow.log_artifact(self.checkpoint)
        signature = infer_signature(self.x_train, self.predict(self.x_train))
        dataset = from_pandas(self.original_dataset.loc[self.x_train.index], source="local")
        mlflow.log_input(dataset, context="tweet-dataset")
        if isinstance(self.model, PreTrainedModel):
            mlflow.transformers.log_model(
                transformers_model=self.checkpoint,
                artifact_path=self.name,
                task="text-classification",  # important !
                tokenizer=self.tokenizer,
                signature=signature,
                registered_model_name=f"{self.name}-quickstart"
            )
        elif isinstance(self.model, lgm.LGBMClassifier):
            mlflow.lightgbm.log_model(
                lgb_model=self.model,
                artifact_path=self.name,
                signature=signature,
                registered_model_name=f"{self.name}-quickstart",
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path=self.name,
                signature=signature,
                registered_model_name=f"{self.name}-quickstart",
            )
        mlflow.end_run()

    def predict(self, x: Union[pd.Series, np.ndarray]):
        """
        Predict the sentiment of the input data
        """
        predicted_class = self.model.predict(self.preprocessing(x))
        return predicted_class
    
class SklearnBaseModel(BaseModelABC):
    def log_metrics(self):
        super().log_metrics()
        #mlflow.sklearn.log_model(self.model, self.name)
        mlflow.log_params(self.model.get_params())

class TorchBaseModel(TorchModelTrainMixin, BaseModelABC):
    """
    Base class to train and predict on a dataset and register data on MLFLow
    """

    def __init__(self, dataset: pd.DataFrame):
        if dist.is_available():
            dist.init_process_group("nccl")
            if dist.is_initialized():
                self.local_rank = dist.get_rank()
                torch.cuda.set_device(self.local_rank)
            super().__init__(dataset)
            if dist.is_initialized():
                self.dataloader, self.sampler = self.get_ddp_dataloader()
                logger.info(f"Rank {dist.get_rank()} using DDP")

    def parralle_model(self):
        self.model = self.model.cuda(f"cuda:{self.local_rank}")
        self.model = nn.parallel.DistributedDataParallel(self.model, 
                                                            device_ids=[self.local_rank], 
                                                            output_device=self.local_rank,
                                                            find_unused_parameters=True)

    def preprocessing(self, data):
        return self.tokenizer(list(data), return_tensors="pt", truncation=True, padding=True)

class RandomForestModel(SklearnBaseModel):
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
            super().train()
            # self.model.n_estimators += 10
        except Exception as e:
            logger.error(f"Erreur pendant l'entraînement: {str(e)}")
            raise


class LightGBMModel(SklearnBaseModel):
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


    def init_items(self):
        self.model = lgm.LGBMClassifier(
            #n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        )
        self.tokenizer = self.tokenizer_class(
            max_features=1000, min_df=2, max_df=0.95
        )

    def clean(self, tweet):
        translator = str.maketrans('','', string.punctuation)
        tweet = tweet.translate(translator)
        tweet = re.sub("^[a-z][A-Z]", " ",tweet)
        tweet = tweet.lower()
        tweet = ' '.join(tweet.split())
        return tweet

    def train(self):
        """
        Train the Random Forest model with progress tracking
        """
        self.init_mlflow()
        try:
            # Vectorisation du texte
            X_train = self.tokenizer.fit_transform(self.x_train.apply(self.clean))
            self.model.fit(X_train, self.y_train)
            super().train()
        except Exception as e:
            logger.error(f"Erreur pendant l'entraînement: {str(e)}")
            raise


class LogisticRegressionModel(SklearnBaseModel):
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
    space = [
                Real(1e-6, 100.0, prior='log-uniform', name='C'),
                Categorical(['l1', 'l2'], name='penalty')
                ]
    def init_items(self):
        """
        Initialize the items for the model
        """
        self.model = LogisticRegression(max_iter=1000,
                                        C=1.7279373898388395,
                                        penalty='l1',
                                        n_jobs=4, 
                                        verbose=True)
        self.tokenizer = self.tokenizer_class()

    def objective(self, tokens, params):
        with mlflow.start_run(nested=True):
            # Enregistrement des paramètres
            param_dict = {dim.name: val for dim, val in zip(self.space, params)}
            mlflow.log_params(param_dict)
            # Modèle
            model = LogisticRegression(
                max_iter=1000,
                solver='liblinear',
                **param_dict
            )
            score = np.mean(cross_val_score(model, tokens, self.y_train, cv=5, scoring='accuracy'))
            mlflow.log_metric("accuracy", score)
            # Return negative score to minimize
            return -score
        
    def train(self, run_name: str = "", optim:str = ""):
        self.init_mlflow(run_name)
        tokens = self.tokenizer.fit_transform(self.x_train)
        if optim == "bayezian":
            gp_minimize(partial(self.objective, tokens), self.space, n_calls=30, random_state=42)
        else:
            self.model.fit(tokens, self.y_train)
            super().train()


class BertModel(TorchBaseModel):
    """
    Using a bert base mutilingual uncased sentiment to predict tweet sentiments
    """

    # Directory to save the model
    checkpoint = "checkpoints/bert"
    tokenizer_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    dataset_class = TweetDataset
    epoch = 1
    batch_size = 100
    out_features = 2
    lr = 2.561e-4
    device = DEVICE

    def params_optim(self, trial):
        lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
        gamma = trial.suggest_float('gamma', 0.1, 0.9)
        step_size = trial.suggest_int('step_size', 2, 10)
        return {'lr': lr, 'gamma': gamma, 'step_size': step_size} 

    def reinit_scheduler_optimizer(self, lr, gamma, step_size):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

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
        if dist.is_available() and dist.is_initialized():
            self.parralle_model()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        total_steps = len(self.dataloader) * self.epoch
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
            num_training_steps=total_steps
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def save(self):
        if dist.is_available() and dist.is_initialized():
            self.model.module.save_pretrained(self.checkpoint)
        else:
            self.model.save_pretrained(self.checkpoint)
        self.tokenizer.save_pretrained(self.checkpoint)

    def predict(self, x: list):
        predicted_class = []
        for i in range(0, len(x), self.batch_size):
            inputs = self.preprocessing(x[i:i+self.batch_size])
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                predicted_class.extend(torch.argmax(probs, dim=1).tolist())
        return predicted_class

class RobertaModel(BertModel):
    """
    Using a roberta base sentiment to predict tweet sentiments
    """
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer_name = "cardiffnlp/twitter-roberta-base-sentiment"
    checkpoint = "checkpoints/roberta"


class LSTMModel(TorchBaseModel):
    checkpoint = "checkpoints/lstm"
    tokenizer_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    name = "LSTM"
    dataset_class = TweetDataset
    epoch = 1
    batch_size = 32
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
                checkpoint = torch.load(self.checkpoint + "/model.pth",map_location=f"cuda:{self.local_rank}")
            embedding_weights = {
                k: v
                for k, v in checkpoint.items()
                if k.startswith("embeddings.") or k.startswith("lstm.")
            }
            self.model.load_state_dict(embedding_weights, strict=False)
            self.model.eval()

        if dist.is_available() and dist.is_initialized():
            self.parralle_model()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def params_optim(self, trial):
        lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
        factor = trial.suggest_float('factor', 0.1, 0.9)
        patience = trial.suggest_int('patience', 2, 10)
        return {'lr': lr, 'factor': factor, 'patience': patience} 

    def reinit_scheduler_optimizer(self, lr, factor, patience):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=factor, patience=patience
        )

    def save(self):
        # Create the parent directory saving tokenizer
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() != 0:
                return  # ne rien faire sur les autres GPU
        self.tokenizer.save_pretrained(self.checkpoint)
        torch.save(self.model.state_dict(), self.checkpoint + "/model.pth")

    def predict(self, x):
        predicted_class = []
        for i in range(0, len(x), self.batch_size):
            inputs = self.preprocessing(x[i:i+self.batch_size])
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Appliquer sigmoïde sur les 4 prédictions
                probs = torch.sigmoid(outputs)
                # Convertir en classes (0 ou 1) en utilisant un seuil de 0.5
                predicted_class.extend((probs > 0.5).int().tolist())
        return predicted_class


def split_data(df: pd.DataFrame, shuffle: bool = True):
    train, test, y_train, y_test = train_test_split(
        df["text"],
        df["target"],
        test_size=0.2,
        random_state=42,
        shuffle=shuffle,
    )
    train, val, y_train, y_val = train_test_split(
        train, y_train, test_size=0.25, random_state=42
    )
    return train, test, val, y_train, y_test, y_val

def load_data(path):
    headers = ["target", "ids", "date", "flag", "user", "text"]
    df_tweets = pd.read_csv(path, names=headers, encoding="latin-1")
    # On prend target 0 negatif 1 positif
    df_tweets.loc[:, "target"] = df_tweets.target.map({0: int(0), 4: int(1)})

    return df_tweets
