import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
from abc import ABC
from typing import Union
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import logging
import torch.nn.functional as F
from tqdm import tqdm
import mlflow

logger = logging.getLogger(__name__)



SENTIMENT_LABELS = {
    0: "😡 unsatisfy",
    4: "😊 satisfy",
}

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, tweets, labels):
        self.tokenizer = tokenizer
        self.tweets = tweets
        self.labels = labels

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(tweet, return_tensors="pt", truncation=True, padding=True)
        return tweet, inputs, label


class BaseModel(ABC):
    checkpoint: str = ""
    tokenizer = None
    def __init__(self):
        self.model = None
        self.dataset = None

    def mlflow_record(self, params: dict, metrics: dict, model, model_name: str, **kwargs):
        with mlflow.start_run():
            mlflow.log_params(params)
            for k,v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.sklearn.log_model(model, model_name)
            mlflow.log_artifact(self.checkpoint)

    def train(self, x_train, y_train):
        """"""

    def predict(self, x:Union[pd.Series, numpy.ndarray]):
        """"""


class LogisticRegressionModel(BaseModel):
    checkpoint = "checkpoints/logistic_regression.pkl"
    checkpoint_tokenizer = "checkpoints/Logistic_regression_tokenizer.pkl"
    tokenizer = TfidfVectorizer()
    def __init__(self):
        super().__init__()
        if Path(self.checkpoint).exists():
            self.model = joblib.load(self.checkpoint)
        else:
            self.model = LogisticRegression()
        if Path(self.checkpoint_tokenizer).exists():
            self.tokenizer = joblib.load(self.checkpoint_tokenizer)


    def predict(self, x: Union[pd.Series, numpy.ndarray]):
        x = self.tokenizer.transform(x)
        predicted_class = self.model.predict(x)
        logger.info(f"predicted {x.shape}")
        return [SENTIMENT_LABELS[p] for p in predicted_class]

    def train(self, x_train, y_train):
        x_train = self.tokenizer.fit_transform(x_train)
        self.model.fit(x_train, y_train)
        params = self.model.get_params()
        metrics = {"score": self.model.score(x_train, y_train)}
        self.mlflow_record(params,
                           metrics,
                           self.model,
                           "logistic_regression")
        joblib.dump(self.model, self.checkpoint)
        joblib.dump(self.tokenizer, self.checkpoint_tokenizer)


class BertModel(BaseModel):
    checkpoint = "checkpoints/bert.pkl"
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment")
        self.model.classifier = torch.nn.Linear(768, 2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, x_train, y_train):
        dataset = TweetDataset(self.tokenizer, x_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        self.model.train()
        self.model.to("mps")
        for epoch in range(3):
            for tweet, inputs, label in tqdm(dataloader):
                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = self.criterion(outputs.logits, label)
                loss.backward()
                self.optimizer.step()
        self.model.save("sentiment_model_checkpoint.pkl")

    def predict(self, x):
        tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        inputs = tokenizer(x, return_tensor='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
        return SENTIMENT_LABELS[predicted_class]


def load_data(path):
    headers = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df_tweets = pd.read_csv(path, names=headers, encoding="latin-1")
    train, test, y_train, y_test = train_test_split(df_tweets['text'], df_tweets['target'], test_size=0.2,
                                                    random_state=42)
    return train, test, y_train, y_test

