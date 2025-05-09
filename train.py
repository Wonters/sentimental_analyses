## Import lightgbm avoiding segfault error, protection against segfault
#import lightgbm as lgb
import logging
from src.ml import LightGBMModel, load_data, LSTMModel
import logging

logging.basicConfig(level=logging.INFO)

df  = load_data('../data/training.1600000.processed.noemoticon.csv')
df = df.sample(frac=0.1, random_state=42)
model = LSTMModel(df)
model.optuna_train(run_name="lstm", n_trials=10)
