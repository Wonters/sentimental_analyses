import pytest
import asyncio
from typing import List
from ..client import launch_prediction





@pytest.mark.asyncio
async def test_predict():

    rep  = await asyncio.gather(*[launch_prediction() for _ in range(10)], return_exceptions=True)
    print(rep)

    
    