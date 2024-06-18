from typing import List, Optional

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle


from src.components.rerank.custom.utils import post_http_request, get_response


class TextEmbeddingInferenceRerank(BaseNodePostprocessor):
    api_url: str = Field(description="Embedding inference endpoint")
    top_n: int = Field(description="Number of nodes to return sorted by score.")
    keep_retrieval_score: bool = Field(
        default=False,
        description="Whether to keep the retrieval score in metadata.",
    )

    def __init__(
        self,
        api_url: str,
        top_n: int = 2,
        keep_retrieval_score: Optional[bool] = False,
    ):
        super().__init__(
            api_url=api_url,
            top_n=top_n,
            keep_retrieval_score=keep_retrieval_score,
        )

    @classmethod
    def class_name(cls) -> str:
        return "TextEmbeddingInferenceRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        query_and_nodes = {}
        query_and_nodes["query"] = query_bundle.query_str
        query_and_nodes["texts"] = [
            node.node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes
        ]

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:

            response = post_http_request(self.api_url, query_and_nodes)
            # Get the results from rerank model's backend of Text Inference Embedding
            # Return a list sorted scores
            results = get_response(response)
            assert len(results) == len(nodes)

            sorted_index = [result["index"] for result in results][self.top_n]
            sorted_score = [result["score"] for result in results][self.top_n]

            new_nodes = []
            for index, score in zip(sorted_index, sorted_score):
                if self.keep_retrieval_score:
                    # keep the retrieval score in metadata
                    nodes[index].node.metadata["retrieval_score"] = score
                nodes[index].score = score
                new_nodes.append(nodes[index])

            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes
