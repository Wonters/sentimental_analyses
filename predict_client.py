"""
Use this script as exemple to connect to the server and launch prediction
"""


import asyncio
from src.client import launch_prediction
import os

os.environ["IP"] = "tweetsentiment.shift.python.software.fr:5000"
async def main():

    rep  = await asyncio.gather(*[launch_prediction() for _ in range(10)], return_exceptions=True)
    print(rep)

if __name__ == "__main__":
    asyncio.run(main())