import os
import json
import base64
import hashlib
import logging
import sqlite3
import html
from uuid import uuid4
from logging import getLogger
from dataclasses import dataclass, asdict, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Tuple,
    TypedDict,
    Union,
    cast,
)

import numpy as np
import networkx as nx
from PIL import Image
from .utils import logger
from .base import BaseGraphStorage, BaseVectorStorage
from embeddings.BaseEmbeddingModel import BaseEmbeddingModel

logging.basicConfig(level=logging.INFO)

f_ID = "__id__"
f_VECTOR = "__vector__"
f_METRICS = "__metrics__"
Data = TypedDict("Data", {"__id__": str, "__vector__": np.ndarray})
DataBase = TypedDict(
    "DataBase", {"embedding_dim": int, "data": list[Data], "matrix": np.ndarray}
)

Float = np.float32
ConditionLambda = Callable[[Data], bool]
logger = getLogger("nano-vectordb")


def array_to_buffer_string(array: np.ndarray) -> str:
    return base64.b64encode(array.tobytes()).decode()


def buffer_string_to_array(base64_str: str, dtype=Float) -> np.ndarray:
    return np.frombuffer(base64.b64decode(base64_str), dtype=dtype)


def load_storage(file_name) -> Union[DataBase, None]:
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        data = json.load(f)
    data["matrix"] = buffer_string_to_array(data["matrix"]).reshape(
        -1, data["embedding_dim"]
    )
    logger.info(f"Load {data['matrix'].shape} data")
    return data


def hash_ndarray(a: np.ndarray) -> str:
    return hashlib.md5(a.tobytes()).hexdigest()


def normalize(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


@dataclass
class NanoVectorDB:
    embedding_dim: int
    metric: Literal["cosine"] = "cosine"
    storage_file: str = "nano-vectordb.json"

    def pre_process(self):
        if self.metric == "cosine":
            self.__storage["matrix"] = normalize(self.__storage["matrix"])

    def __post_init__(self):
        default_storage = {
            "embedding_dim": self.embedding_dim,
            "data": [],
            "matrix": np.array([], dtype=Float).reshape(0, self.embedding_dim),
        }
        storage: DataBase = load_storage(self.storage_file) or default_storage
        assert (
            storage["embedding_dim"] == self.embedding_dim
        ), f"Embedding dim mismatch, expected: {self.embedding_dim}, but loaded: {storage['embedding_dim']}"
        self.__storage = storage
        self.usable_metrics = {
            "cosine": self._cosine_query,
        }
        assert self.metric in self.usable_metrics, f"Metric {self.metric} not supported"
        self.pre_process()
        logger.info(f"Init {asdict(self)} {len(self.__storage['data'])} data")

    def get_additional_data(self):
        return self.__storage.get("additional_data", {})

    def store_additional_data(self, **kwargs):
        self.__storage["additional_data"] = kwargs

    def upsert(self, datas: list[Data]):
        _index_datas = {
            data.get(f_ID, hash_ndarray(data[f_VECTOR])): data for data in datas
        }
        if self.metric == "cosine":
            for v in _index_datas.values():
                v[f_VECTOR] = normalize(v[f_VECTOR])
        report_return = {"update": [], "insert": []}
        for i, already_data in enumerate(self.__storage["data"]):
            if already_data[f_ID] in _index_datas:
                update_d = _index_datas.pop(already_data[f_ID])
                self.__storage["matrix"][i] = update_d[f_VECTOR].astype(Float)
                del update_d[f_VECTOR]
                self.__storage["data"][i] = update_d
                report_return["update"].append(already_data[f_ID])
        if len(_index_datas) == 0:
            return report_return
        report_return["insert"].extend(list(_index_datas.keys()))
        new_matrix = np.array(
            [data[f_VECTOR] for data in _index_datas.values()], dtype=Float
        )
        new_datas = []
        for new_k, new_d in _index_datas.items():
            del new_d[f_VECTOR]
            new_d[f_ID] = new_k
            new_datas.append(new_d)
        self.__storage["data"].extend(new_datas)
        self.__storage["matrix"] = np.vstack([self.__storage["matrix"], new_matrix])
        return report_return

    def get(self, ids: list[str]):
        return [data for data in self.__storage["data"] if data[f_ID] in ids]

    def delete(self, ids: list[str]):
        ids = set(ids)
        left_data = []
        delete_index = []
        for i, data in enumerate(self.__storage["data"]):
            if data[f_ID] in ids:
                delete_index.append(i)
                ids.remove(data[f_ID])
            else:
                left_data.append(data)
        self.__storage["data"] = left_data
        self.__storage["matrix"] = np.delete(
            self.__storage["matrix"], delete_index, axis=0
        )

    def save(self):
        storage = {
            **self.__storage,
            "matrix": array_to_buffer_string(self.__storage["matrix"]),
        }
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(storage, f, ensure_ascii=False)

    def __len__(self):
        return len(self.__storage["data"])

    def query(
        self,
        query: np.ndarray,
        top_k: int = 10,
        better_than_threshold: float = None,
        filter_lambda: ConditionLambda = None,
    ) -> list[dict]:
        return self.usable_metrics[self.metric](
            query, top_k, better_than_threshold, filter_lambda=filter_lambda
        )

    def _cosine_query(
        self,
        query: np.ndarray,
        top_k: int,
        better_than_threshold: float,
        filter_lambda: ConditionLambda = None,
    ):
        query = normalize(query)
        if filter_lambda is None:
            use_matrix = self.__storage["matrix"]
            filter_index = np.arange(len(self.__storage["data"]))
        else:
            filter_index = np.array(
                [
                    i
                    for i, data in enumerate(self.__storage["data"])
                    if filter_lambda(data)
                ]
            )
            use_matrix = self.__storage["matrix"][filter_index]
        scores = np.dot(use_matrix, query)
        sort_index = np.argsort(scores)[-top_k:]
        sort_index = sort_index[::-1]
        sort_abs_index = filter_index[sort_index]
        results = []
        for abs_i, rel_i in zip(sort_abs_index, sort_index):
            if (
                better_than_threshold is not None
                and scores[rel_i] < better_than_threshold
            ):
                break
            results.append({**self.__storage["data"][abs_i], f_METRICS: scores[rel_i]})
        return results

@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.1

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    def _upsert(self, datas):
        results = self._client.upsert(datas=datas)
        return results

    def _query(self, embedding, top_k=5):
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results
    
    def _batch_query(self, embeddings, top_k=5):
        results = []
        for embedding in embeddings:
            results.append(self._client.query(
                query=embedding,
                top_k=top_k,
                better_than_threshold=self.cosine_better_than_threshold,
            ))
        results = [
            [
                {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in res
            ] for res in results
        ]
        return results

    def _delete(self, ids:list):
        try:
            if self._client.get(ids):
                self._client.delete(ids)
                logger.info(f"Delete {ids} from {self.namespace}")
            else:
                logger.warning(f"{ids} not found in {self.namespace}")
        except Exception as e:
            logger.error(f"Delete {ids} from {self.namespace} failed: {e}")    

    def is_empty(self):
        return len(self._client) == 0
    
    def get_datas(self):
        res = []
        datas = self.client_storage["data"]
        for data in datas:
            cur_data = {}
            cur_data["id"] = data["__id__"]
            for key, value in data.items():
                if key not in ["__id__", "content"]:
                    cur_data[key] = value
            res.append(cur_data)
        
        return res
            

    @property
    def client_storage(self):
        return getattr(self._client, "_NanoVectorDB__storage")

    def index_done_callback(self):
        self._client.save()

@dataclass
class TextNanoVectorDBStorage(NanoVectorDBStorage):
    def upsert(self, datas: dict[str, dict]):
        logger.info(f"Inserting {len(datas)} vectors to {self.namespace}")
        if not len(datas):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in datas.items()
        ]
        
        # text embedding, entities need to compute 
        text_contents = [v["content"] for v in datas.values()]
        num_text_list = [len(v["content"]) for v in datas.values()]
        texts = [text for content in text_contents for text in content]
        text_embeddings = self.embedding_model.get_text_features(texts)
        start_idx = 0
        for i, (d, num) in enumerate(zip(list_data, num_text_list)):
            end_idx = start_idx + num
            d["__vector__"] = text_embeddings[start_idx:end_idx].mean(axis=0)
            start_idx = end_idx
        
        results = self._upsert(list_data)
    
    def query(self, query: str, top_k=5):
        embedding = self.embedding_model.get_text_features([query])[0]
        return self._query(embedding, top_k=top_k)
    
    def batch_query(self, queries, top_k=1):
        embeddings = self.embedding_model.get_text_features(queries)
        embeddings = embeddings.tolist()
        return self._batch_query(embeddings, top_k=top_k)
    
    def delete(self, ids:Union[list[str], str]):
        if isinstance(ids, str):
            ids = [ids]
        
        self._delete(ids)
        
    def is_empty(self):
        return len(self._client) == 0
    
    def get_data(self, id):
        return self._client.get([id])[0]
    
    def get_datas(self):
        res = []
        datas = self.client_storage["data"]
        for data in datas:
            cur_data = {}
            cur_data["id"] = data["__id__"]
            for key, value in data.items():
                if key not in ["__id__", "content"]:
                    cur_data[key] = value
            res.append(cur_data)
        
        return res

    def get_previous_data(self, event_id: str) -> Union[dict, None]:
        datas = self.client_storage["data"]
        for i, data in enumerate(datas):
            if data["__id__"] == event_id:
                return datas[i - 1] if i > 0 else None
        return None

    def get_next_data(self, event_id: str) -> Union[dict, None]:
        datas = self.client_storage["data"]
        for i, data in enumerate(datas):
            if data["__id__"] == event_id:
                return datas[i + 1] if i < len(datas) - 1 else None
        return None

@dataclass
class ImageNanoVectorDBStorage(NanoVectorDBStorage):
    cosine_better_than_threshold: float = 0.1
    def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []

        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        
        contents = [v["content"] for v in data.values()]
        num_list = [len(v["content"]) for v in data.values()]

        images = [Image.open(image_path).convert("RGB") for content in contents for image_path in content]

        embeddings = self.embedding_model.get_image_features(images)

        start_idx = 0
        for i, (d, num) in enumerate(zip(list_data, num_list)):
            end_idx = start_idx + num
            d["__vector__"] = embeddings[start_idx:end_idx].mean(axis=0)
            # d["related_count"] = num
            start_idx = end_idx

        results = self._upsert(list_data)
        return results

    def query(self, query: str, top_k=5):
        embedding = self.embedding_model.get_text_features([query])[0]
        return self._query(embedding, top_k=top_k)

    def batch_query(self, queries, top_k=1, mode="text"):
        if mode == "text":
            embeddings = self.embedding_model.get_text_features(queries)
        elif mode == "image":
            embeddings = self.embedding_model.get_image_features(queries)
        else:
            raise ValueError("Mode should be either 'text' or 'image'")
        embeddings = embeddings.tolist()
        return self._batch_query(embeddings, top_k=top_k)

    def delete(self, ids:Union[list[str], str]):
        if isinstance(ids, str):
            ids = [ids]
        
        self._delete(ids)

    def is_empty(self):
        return len(self._client) == 0
    
    def get_datas(self):
        res = []
        datas = self.client_storage["data"]
        for data in datas:
            cur_data = {}
            cur_data["id"] = data["__id__"]
            for key, value in data.items():
                if key not in ["__id__", "content"]:
                    cur_data[key] = value
            res.append(cur_data)
        
        return res
    

@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {
            node: html.unescape(node.upper().strip()) for node in graph.nodes()
        }  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()

    def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)

    def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)
    
    def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    def delete_node(self, node_id: str):
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.info(f"Node {node_id} deleted from the graph.")
        else:
            logger.warning(f"Node {node_id} not found in the graph for deletion.")

    def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return self._node_embed_algorithms[algorithm]()