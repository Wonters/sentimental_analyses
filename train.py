from src.ml import load_data, RandomForestModel, LogisticRegressionModel, BertModel
import logging
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
file = "../data/training.1600000.processed.noemoticon.csv"
original_df = load_data(file)
model = BertModel(original_df)
model.train()
