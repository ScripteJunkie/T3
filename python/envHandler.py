import os
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path('../external/.env')
load_dotenv(dotenv_path=dotenv_path)

# BALL_HSV_MIN = list(map(int, os.getenv('BALL_HSV_MIN').split(", ")))
# BALL_HSV_MAX = list(map(int, os.getenv('BALL_HSV_MAX').split(", ")))

def get(target):
    return os.getenv(target)

def toList(target):
    return list(map(int, os.getenv(target).split(", ")))