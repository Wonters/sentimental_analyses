import fastapi
import httpx
from fastapi.routing import APIRouter
from fastapi.requests import Request
from fastapi.responses import Response
from fastapi import Form
from fastapi.templating import Jinja2Templates
from multiprocessing import Process, Pipe
from multiprocessing.pool import Pool
from typing import List
from rich.logging import RichHandler
import prometheus_client as prom
import logging
import pymongo
import uuid
import time
import random
from .ml import BertModel
from .models import Tweet, Sentiment

# Configuration des métriques Prometheus
PREDICTION_COUNT = prom.Counter(
    "prediction_count_total", "Nombre total de prédictions effectuées", ["model"]
)

PREDICTION_TIME = prom.Gauge(
    "prediction_time_seconds", "Temps pris pour effectuer une prédiction", ["model"]
)

PREDICTION_STATUS = prom.Gauge(
    "prediction_status", "Statut de la prédiction (1=en cours, 0=terminé)", ["model"]
)

LOKI_URL = "http://loki:3100/loki/api/v1/push"
MONGO_URI = "mongodb://db:27017/"
FLAG_START = "started"
FLAG_DONE = "done"

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger(__name__)
app = fastapi.FastAPI()
templates = Jinja2Templates(directory="templates")


pool = None
tasks = {}
pipes = {}


def get_pool():
    global pool
    if pool is None:
        pool = Pool(processes=1)
    return pool


def run_predict(text: List[Tweet], sender):
    logger.info("prediction started")
    sender.send(FLAG_START)
    start_time = time.time()
    time.sleep(3)
    # result = BertModel().predict(text)
    sender.send(FLAG_DONE)
    result = [("tweet",random.randint(0, 1)) for _ in range(len(text))]
    return result, time.time() - start_time


class PredictApp:
    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/", self.get, methods=["GET"])
        self.router.add_api_route("/predict", self.predict, methods=["POST"])
        self.router.add_api_route("/get_result/{id_task}", self.get_result, methods=["GET"])
        self.router.add_api_route("/metrics", self.metrics, methods=["GET"])
        self.router.add_api_route("/ws/{task_id}", self.websocket_endpoint, methods=["GET"])
        self.active_connections = {}

    async def get(self, request: Request):
        """"""
        return templates.TemplateResponse(
            "index.html", {"request": request, "prediction": None}
        )

    def metrics(self):
        return Response(prom.generate_latest(), media_type=prom.CONTENT_TYPE_LATEST)

    async def get_result(self, request: Request, id_task: str):
        """
        Get the result of the prediction
        """
        try:
            process = tasks[id_task]
            pipe_receiver, pipe_sender = pipes[id_task]
            if process.ready():
                PREDICTION_STATUS.labels("bert").set(0)
                result, duration= process.get()
                PREDICTION_TIME.labels("bert").set(duration)
                PREDICTION_COUNT.labels("bert").inc()
                del tasks[id_task]
                del pipes[id_task]
                await self.loki_push(result)
                return {"status": "done", "result": result}
            else:
                status = "pending"
                if pipe_receiver.poll():
                    status = pipe_receiver.recv()
                return {"status": status, "message": ""}
        except KeyError:
            logger.warning(f"task {id_task} not found")
            return {"status": "error", "message": "Task not found"}

    async def loki_push(self, results: List[tuple]):
        """
        Push the text to loki
        """
        log_payload = {
            "streams": [
                {
                    "stream": {
                        "app": "tweet-analyzer",
                        "prediction": str(sentiment),
                    },
                    "values": [
                        [
                            str(int(time.time() * 1e9)),
                            f"Tweet: {tweet} | Prediction: {sentiment} | Correct: {True if sentiment == 1 else False}",
                        ]
                    ],
                }
            for tweet, sentiment in results
            ]
        }
        print(log_payload)

        async with httpx.AsyncClient() as client:
            await client.post(LOKI_URL, json=log_payload)

    async def predict(self, request: Request, text: List[Tweet]):
        """
        Predict
        """
        with pymongo.MongoClient(MONGO_URI) as client:
            db = client["sentiment_analyses"]
            collection = db["tweets"]
            # Convertir la liste de tweets en liste de documents
            tweets = [{"text": str(tweet)} for tweet in text]
            collection.insert_many(tweets)
            logger.info(f"tweets added to db: {len(tweets)} tweets")

        logger.info(f"predicting {text}")
        p = get_pool()
        pipe = Pipe()
        result = p.apply_async(run_predict, (text, pipe[0]))
        PREDICTION_STATUS.labels("bert").set(1)
        task_id = str(uuid.uuid4())
        tasks[task_id] = result
        pipes[task_id] = pipe
        logger.info(f"task {task_id} started")
        return {"task_id": task_id, "status": "processing"}


predict_app = PredictApp()
app.include_router(predict_app.router)

