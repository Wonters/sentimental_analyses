import pytest
import httpx
import asyncio
from typing import List
from ..models import Tweet

@pytest.mark.asyncio
async def test_predict():
    async with httpx.AsyncClient(base_url="http://127.0.0.1:5000") as client:
        # Test de la prédiction
        response = await client.post(
            "/predict",
            json=[{"text": "hello world"}]
        )
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        task_id = data["task_id"]
        response = await client.get(f"/get_result/{task_id}")
        payload = response.json()
        print(payload)
        while payload["status"] != "done":
            await asyncio.sleep(1)
            response = await client.get(f"/get_result/{task_id}")
            payload = response.json()
            print(payload)
    