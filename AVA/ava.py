from .operate import (
    extract_knowledge_graph,
    tree_search,
    generate_sa_self_consistency_result,
    calculate_sa_score,
    choose_best_sa_answer,
    generate_ca_self_consistency_result,
    calculate_ca_score,
    choose_best_ca_answer,
)

from .storage import (
    TextNanoVectorDBStorage,
    ImageNanoVectorDBStorage,
    NetworkXStorage,
)

from .utils import (
    logger,
    set_logger,
)

from .base import (
    BaseGraphStorage,
    BaseVectorStorage,
)

from embeddings.BaseEmbeddingModel import BaseEmbeddingModel
from embeddings.JinaCLIP import JinaCLIP

import os
import json
import copy
from typing import Union
from dataclasses import dataclass, field
from typing import Type
from llms.BaseModel import BaseVideoModel, BaseLanguageModel
from video_utils import VideoRepresentation

@dataclass
class AVA:
    working_dir: str = field(default=None)
    video: VideoRepresentation = field(default=None)
    llm_model: Union[BaseVideoModel, BaseLanguageModel] = field(default=None)

    kv_storage: str = field(default="JsonKVStorage")
    text_vector_storage: str = field(default="TextNanoVectorDBStorage")
    image_vector_storage: str = field(default="ImageNanoVectorDBStorage")
    dualdim_vector_storage: str = field(default="DualDimNanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # video chunking
    video_chunk_duration: int = field(default=3) # seconds
    video_chunk_overlap: int = field(default=0) # seconds
    video_chunk_num_frames: int = field(default=6) # seconds
    entity_extraction_num_frames: int = field(default=8)
    
    event_merge_algorithm: str = field(default="partition")

    embedding_batch_num: int = 64

    # extension
    addon_params: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.video is not None, "Please provide video!"
        self.working_dir = os.path.join(self.video.work_dir, "kg")
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        
        # log
        log_file = os.path.join(self.working_dir, "kg.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)
        
        # check video file directory
        assert self.working_dir is not None and os.path.exists(self.working_dir), "Please provide video directory!"
        
        self.global_config = {
                "video": self.video,
                "working_dir": self.working_dir,
                "video_chunk_duration": self.video_chunk_duration,
                "video_chunk_overlap": self.video_chunk_overlap,
                "video_chunk_num_frames": self.video_chunk_num_frames,
                "entity_extraction_num_frames": self.entity_extraction_num_frames,
                "embedding_batch_num": self.embedding_batch_num,
                # "event_merge_algorithm": self.event_merge_algorithm,
            }
        
        # models
        self.text_embedding_model: BaseEmbeddingModel = JinaCLIP("jinaai/jina-clip-v1")
        self.image_embedding_model: BaseEmbeddingModel = self.text_embedding_model
        self.text_embedding_dim = self.text_embedding_model.embedding_dim
        self.image_embedding_dim = self.image_embedding_model.embedding_dim
        
        # check storage class
        self.text_vector_db_storage_cls: Type[BaseVectorStorage] = TextNanoVectorDBStorage
        self.image_vector_db_storage_cls: Type[BaseVectorStorage] = ImageNanoVectorDBStorage
        self.graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage

        # storage
        self.video_knowledge_graph = self.graph_storage_cls(
            namespace="event_knowledge_graph",
            global_config=self.global_config,
        )
        self.events_vdb = self.text_vector_db_storage_cls(
            namespace="events",
            global_config=self.global_config,
            embedding_model=self.text_embedding_model,
            embedding_dim=self.text_embedding_dim,
            meta_fields={"id", "name", "description", "duration"},
        )
        self.entities_vdb = self.text_vector_db_storage_cls(
            namespace="entities",
            global_config=self.global_config,
            embedding_model=self.text_embedding_model,
            embedding_dim=self.text_embedding_dim,
            meta_fields={"id", "descriptions", "timestamps", "frame_indices", "durations", "events"},
        )
        self.relations_vdb = self.text_vector_db_storage_cls(
            namespace="relations",
            global_config=self.global_config,
            embedding_model=self.text_embedding_model,
            embedding_dim=self.text_embedding_dim,
            meta_fields={"id", "entity1", "entity2", "description"},
        )
        self.features_vdb = self.image_vector_db_storage_cls(
            namespace="features",
            global_config=self.global_config,
            embedding_model=self.image_embedding_model,
            embedding_dim=self.image_embedding_dim,
            meta_fields={"id", "frame_dir", "event"},
        )

    def construct(self):
        """
        construct from the video directory
        """
        try:
            logger.info(f"Constructing Graph with working directory: {self.working_dir}")
            
            self.kg = extract_knowledge_graph(
                llm=self.llm_model,
                embedding_model=self.text_embedding_model,
                knowledge_graph_inst=self.video_knowledge_graph,
                events_vdb=self.events_vdb,
                entities_vdb=self.entities_vdb,
                relations_vdb=self.relations_vdb,
                features_vdb=self.features_vdb,
                global_config=self.global_config,
            )
        finally:
            self._insert_done()
            # pass

    def _insert_done(self):
        for storage_inst in [
            self.events_vdb,
            self.entities_vdb,
            self.relations_vdb,
            self.features_vdb,
            self.video_knowledge_graph
        ]:
            if storage_inst is None:
                continue
            storage_inst.index_done_callback()
    
    def query_tree_search(self, query: str, question_id: int, re_process: bool = False):
        questions_folder = os.path.join(self.video.work_dir, "questions")
        if not os.path.exists(questions_folder):
            os.makedirs(questions_folder)
        question_folder = os.path.join(questions_folder, f"{question_id}")
        if not os.path.exists(question_folder):
            os.makedirs(question_folder)
            
        if not re_process and os.path.exists(os.path.join(question_folder, "tree_information.json")):
            logger.info(f"Tree information already exists for question {question_id}, skipping...")
            return
        try:
            logger.info(f"Querying AVA with query: {query}")
            tree_information = tree_search(query, self.llm_model, self.video, self.events_vdb, self.entities_vdb, self.features_vdb)
            
            with open(os.path.join(question_folder, "tree_information.json"), "w") as f:
                json.dump(tree_information, f)
        finally:
            pass
    
    def generate_SA_answer(self, query: str, question_id: int):
        question_folder = os.path.join(self.video.work_dir, "questions")
        tree_information_file = os.path.join(question_folder, f"{question_id}", "tree_information.json")
        with open(tree_information_file, "r") as f:
            tree_information = json.load(f)
            
        # generate self-consistency result
        if os.path.exists(os.path.join(question_folder, f"{question_id}", "SA_self_consistency_result.json")):
            with open(os.path.join(question_folder, f"{question_id}", "SA_self_consistency_result.json"), "r") as f:
                cleaned_self_consistency_results = json.load(f)
        else:
            self_consistency_result = generate_sa_self_consistency_result(tree_information, self.llm_model)
            cleaned_self_consistency_results = [
                {k: v for k, v in item.items() if k not in {"structed_information", "input_prompt"}}
                for item in self_consistency_result
            ]
        
            # save self-consistency result
            with open(os.path.join(question_folder, f"{question_id}", "SA_self_consistency_result.json"), "w") as f:
                json.dump(cleaned_self_consistency_results, f, indent=4)
        
        # calculate sa nodes scores
        score_results = calculate_sa_score(copy.deepcopy(cleaned_self_consistency_results))
        
        cleaned_score_results = [
            {k: v for k, v in item.items() if k not in {"responses"}}
            for item in score_results
        ]
        
        # save score results
        with open(os.path.join(question_folder, f"{question_id}", "SA_score_result.json"), "w") as f:
            json.dump(cleaned_score_results, f, indent=4)
        
        sorted_score_results = choose_best_sa_answer(cleaned_score_results)
        
        with open(os.path.join(question_folder, f"{question_id}", "sorted_SA_score_result.json"), "w") as f:
            json.dump(sorted_score_results, f, indent=4)
            
        final_sa_answer = list(sorted_score_results[0]["final_score"].keys())[0]

        return final_sa_answer
    
    def generate_CA_answer(self, query: str, question_id: int):
        question_folder = os.path.join(self.video.work_dir, "questions")
        SA_score_result_file = os.path.join(question_folder, f"{question_id}", "sorted_SA_score_result.json")
        assert os.path.exists(SA_score_result_file), "SA score result file does not exist, please generate SA answer first!"
        
        with open(SA_score_result_file, "r") as f:
            SA_score_result = json.load(f)
        
        # generate self-consistency result
        if os.path.exists(os.path.join(question_folder, f"{question_id}", "CA_self_consistency_result.json")):
            with open(os.path.join(question_folder, f"{question_id}", "CA_self_consistency_result.json"), "r") as f:
                ca_self_consistency_result = json.load(f)
        else:
            ca_self_consistency_result = generate_ca_self_consistency_result(
                query=query,
                sorted_sa_nodes=SA_score_result,
                llm=self.llm_model,
                video=self.video,
                self_consistency_num=4,
                max_frames=256,
                max_retries=3,
            )
            
            # save ca self-consistency result
            with open(os.path.join(question_folder, f"{question_id}", "CA_self_consistency_result.json"), "w") as f:
                json.dump(ca_self_consistency_result, f, indent=4)
            
        # calculate ca nodes scores
        score_results = calculate_ca_score(ca_self_consistency_result)
        
        cleaned_score_results = [
            {k: v for k, v in item.items() if k not in {"responses"}}
            for item in score_results
        ]
        
        # save score results
        with open(os.path.join(question_folder, f"{question_id}", "CA_score_result.json"), "w") as f:
            json.dump(cleaned_score_results, f, indent=4)
            
        sorted_score_results = choose_best_ca_answer(cleaned_score_results)
        
        with open(os.path.join(question_folder, f"{question_id}", "sorted_CA_score_result.json"), "w") as f:
            json.dump(sorted_score_results, f, indent=4)
            
        final_ca_answer = list(sorted_score_results[0]["final_score"].keys())[0]
        
        return final_ca_answer
        