import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger

from src.model import CrossEncoder
from src.config import DEFAULT_MODEL_NAME, MAX_LENGTH

MODEL_NAME = os.getenv("MODEL_NAME") or DEFAULT_MODEL_NAME
MAX_LENGTH = os.getenv("MAX_LENGTH") or MAX_LENGTH

logger.info(f'Using rerank model: {MODEL_NAME}')

class RerankInput(BaseModel):
    sentences: List[List[str]]


rerank_model = CrossEncoder(model_name=MODEL_NAME, max_length=MAX_LENGTH)

app = FastAPI()


@app.post("/rerank")
def rerank(data: RerankInput):
    scores = rerank_model.predict(data.sentences)
    return {"scores": scores}
