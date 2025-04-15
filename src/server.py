import fastapi
from fastapi.routing import APIRouter
from fastapi.requests import Request
from fastapi import Form
from fastapi.templating import Jinja2Templates
from typing import List
from rich.logging import RichHandler
import logging
from .ml import BertModel
from .models import Tweet, Sentiment

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger(__name__)
app = fastapi.FastAPI()
templates = Jinja2Templates(directory="templates")


class PredictApp:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/", self.get, methods=["GET"])
        self.router.add_api_route("/predict", self.predict, methods=["POST"])

    async def get(self, request: Request):
        """"""
        return templates.TemplateResponse(
            "index.html", {"request": request, "prediction": None}
        )

    async def predict(self, request: Request, text: List[Tweet]):
        """
        Predict
        """
        result = BertModel().predict(text)
        return {"prediction": result}


predict_app = PredictApp()
app.include_router(predict_app.router)
