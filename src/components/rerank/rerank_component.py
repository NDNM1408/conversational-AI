import logging
from collections.abc import Callable
from typing import Any

from injector import inject, singleton
from llama_index.core.settings import Settings as LlamaIndexSettings
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.utils import set_global_tokenizer
from transformers import AutoTokenizer  # type: ignore

from src.components.llm.prompt_helper import get_prompt_style
from src.paths import models_cache_path, models_path
from src.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class RerankComponent:
    rerank: BaseNodePostprocessor

    @inject
    def __init__(self, settings: Settings) -> None:
        rerank_mode = settings.rag.rerank.mode
        top_n = settings.rag.rerank.top_n
        logger.info("Initializing the Rerank in mode=%s", rerank_mode)
        match rerank_mode:
            case "model_api":
                from src.components.rerank.custom.model_api.base import (
                    ModelAPIRerank,
                )

                model_api_endpoint = settings.model_api_rerank.model_api_endpoint

                self.rerank = ModelAPIRerank(api_url=model_api_endpoint, top_n=top_n)

            case "text_embeddings_inference":
                from src.components.rerank.custom.text_embeddings_inference.base import (
                    TextEmbeddingInferenceRerank,
                )

                text_embeddings_inference_endpoint = (
                    settings.text_embeddings_inference_rerank.text_embeddings_inference_endpoint
                )
                self.rerank = TextEmbeddingInferenceRerank(
                    api_url=text_embeddings_inference_endpoint, top_n=top_n
                )

            case "local":
                from llama_index.core.postprocessor import SentenceTransformerRerank

                model = settings.local_rerank.model
                self.rerank = SentenceTransformerRerank(model=model, top_n=top_n)
