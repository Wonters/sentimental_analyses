from ml import load_data, BertModel
import logging
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
file = "../data/training.1600000.processed.noemoticon.csv"
x_train, x_test, x_val, y_train, y_test, y_val = load_data(file)
model = BertModel(x_train, y_train)
model.train()