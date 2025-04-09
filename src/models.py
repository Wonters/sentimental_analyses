from pydantic import BaseModel
from typing import List

class Tweet(BaseModel):
    text: str

class Sentiment(BaseModel):
    sentiment: int
