import pytest
import httpx
import asyncio
import json
from typing import List
import websockets
from ..models import Tweet


async def launch_prediction():
    async with httpx.AsyncClient(base_url="http://127.0.0.1:5000") as client:
        # Test de la prédiction
        response = await client.post(
            "/predict",
            json=[{"text": "hello world, this is realy a good day"}]
        )
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        task_id = data["task_id"]
    async with websockets.connect(f"ws://127.0.0.1:5000/ws/{task_id}") as ws:
        payload = json.loads(await ws.recv())
        # Ackknowledge system in case of latency
        # Avoid packet loss 
        await ws.send("ACK")
        while payload["status"] != "done":
            payload = json.loads(await ws.recv())
            await ws.send("ACK")
            print(f"wait process {task_id} finish ...")
            await asyncio.sleep(2)
    return payload

@pytest.mark.asyncio
async def test_predict():

    rep  = await asyncio.gather(*[launch_prediction() for _ in range(10)], return_exceptions=True)
    print(rep)

    
    