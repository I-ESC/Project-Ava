from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar

import numpy as np

from embeddings.BaseEmbeddingModel import BaseEmbeddingModel

T = TypeVar("T")


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_model: BaseEmbeddingModel
    embedding_dim: int
    meta_fields: set = field(default_factory=set)

    def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError


from dataclasses import dataclass
from typing import Union
import numpy as np

@dataclass
class BaseGraphStorage(StorageNameSpace):
    def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    def get_node_edges(self, source_node_id: str) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    def delete_node(self, node_id: str):
        raise NotImplementedError

    def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in lightrag.")