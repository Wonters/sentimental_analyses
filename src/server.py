import fastapi
import httpx
import asyncio
from fastapi.routing import APIRouter
from fastapi.requests import Request
from fastapi.responses import Response
from fastapi import Form, WebSocket
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

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger(__name__)
app = fastapi.FastAPI()
templates = Jinja2Templates(directory="templates")


pool = None


def get_pool():
    global pool
    if pool is None:
        pool = Pool(processes=1)
    return pool


def run_predict(text: List[Tweet], sender):
    FLAG_START = "started"
    FLAG_DONE = "done"
    logger.info(f"prediction started {text}")
    sender.send(FLAG_START)
    start_time = time.time()
    time.sleep(3)
    # result = BertModel().predict(text)
    logger.info(f"prediction done {text}")
    sender.send(FLAG_DONE)
    result = [("tweet",random.randint(0, 1)) for _ in range(len(text))]
    return result, time.time() - start_time


class PredictApp:
    ACK_TIMEOUT = 1.0
    

    def __init__(self):
        self.router = APIRouter()
        self.router.add_api_route("/", self.get, methods=["GET"])
        self.router.add_api_route("/predict", self.predict, methods=["POST"])
        self.router.add_api_websocket_route("/ws/{id_task}", self.get_result)
        self.router.add_api_route("/metrics", self.metrics, methods=["GET"])
        self.active_connections = {}
        self.tasks = {}
        self.pipes = {}

    async def get(self, request: Request):
        """"""
        return templates.TemplateResponse(
            "index.html", {"request": request, "prediction": None}
        )

    def metrics(self):
        return Response(prom.generate_latest(), media_type=prom.CONTENT_TYPE_LATEST)
    
    async def wait_for_ack(self, websocket: WebSocket, id_task: str):
        try:
            ack = await asyncio.wait_for(websocket.receive_text(), timeout=self.ACK_TIMEOUT)
            if ack == "ACK":
                return 1
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for ACK on task {id_task}")
        return 0
    
    async def get_result(self, websocket: WebSocket, id_task: str):
        """
        Get the result of the prediction
        """
        await websocket.accept()
        self.active_connections[id_task] = websocket
        logger.info(f"websockets opened {len(self.active_connections)}")
        messages_sent = 0
        messages_acknowledged = 0
        try:
            process = self.tasks[id_task]
            pipe_receiver, pipe_sender = self.pipes[id_task]
            while True:
                if process.ready():
                    logger.info(f"task {id_task} done")
                    PREDICTION_STATUS.labels("bert").dec()
                    result, duration= process.get()
                    PREDICTION_TIME.labels("bert").set(duration)
                    PREDICTION_COUNT.labels("bert").inc()
                    del self.tasks[id_task]
                    del self.pipes[id_task]
                    await self.loki_push(result)
                    await websocket.send_json({"status": "done", "result": result, "duration": duration})
                    messages_sent += 1
                    #messages_acknowledged += await self.wait_for_ack(websocket, id_task)
                    break
                else:
                    status = "pending"
                    if pipe_receiver.poll():
                        status = pipe_receiver.recv()
                        await websocket.send_json({"status": status, "message": ""})
                        messages_sent += 1
                        #messages_acknowledged += await self.wait_for_ack(websocket, id_task)
                await asyncio.sleep(0.1)
        except KeyError:
            logger.warning(f"task {id_task} not found")
            await websocket.json({"status": "error", "message": "Task not found"})
        finally:
            # Vérifier que tous les messages ont été reçus
            while True:
                messages_acknowledged += await self.wait_for_ack(websocket, id_task)
                print(f"messages_sent: {messages_sent}, messages_acknowledged: {messages_acknowledged}")
                if messages_sent == messages_acknowledged:
                    break
                await asyncio.sleep(0.1)
            if messages_sent != messages_acknowledged:
                logger.warning(f"Not all messages were acknowledged for task {id_task}. Sent: {messages_sent}, Acknowledged: {messages_acknowledged}")
            logger.info(f"task {id_task} closed")
            del self.active_connections[id_task]
            await websocket.close()

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
        p = get_pool()
        pipe = Pipe()
        result = p.apply_async(run_predict, (text, pipe[1]))
        PREDICTION_STATUS.labels("bert").inc()
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = result
        self.pipes[task_id] = pipe
        logger.info(f"task {task_id} started")
        return {"task_id": task_id, "status": "processing"}


predict_app = PredictApp()
app.include_router(predict_app.router)

