import json
import logging
import os
import sys

from loguru import logger
import pandas as pd
from .libs.logging import LogHandler
from .program import run
from .pipeline import run_pipeline

# Set logging level (default to INFO to reduce debug logs)
logging_level = os.environ.get("LOG_LEVEL", "INFO").upper()

# Configure standard Python logging
logging.basicConfig(level=logging_level, handlers=[LogHandler()])

# Configure Loguru to match the same level
logger.configure(handlers=[{"sink": sys.stdout, "level": logging_level}])

with open(f"./input/data.json") as file:
    data = json.load(file)

with open(f"./input/params.json") as file:
    params = json.load(file)



print('*'*50)   
print('Running main function')

response = run(data, params)

print('*'*50)   
print("END OF MAIN")