## Import lightgbm avoiding segfault error, protection against segfault
#import lightgbm as lgb
from src.ml import LightGBMModel, load_data, LSTMModel
import logging

df  = load_data('../data/training.1600000.processed.noemoticon.csv')
#df = df.sample(frac=1, random_state=42)
model = LSTMModel(df)
model.train()
