import time
import os
import json
import re
from json_repair import repair_json
from typing import Union
from embeddings.BaseEmbeddingModel import BaseEmbeddingModel
from llms.BaseModel import BaseVideoModel, BaseLanguageModel
from .base import (
    BaseGraphStorage,
    BaseVectorStorage,
)
from .events import extract_events
from .entities import extract_entities_and_relations
from .tree_search import TreeSearch
from video_utils import VideoRepresentation
from bert_score import BERTScorer


def extract_knowledge_graph(
        llm: BaseVideoModel,
        embedding_model: BaseEmbeddingModel,
        knowledge_graph_inst: BaseGraphStorage,
        events_vdb: BaseVectorStorage,
        entities_vdb: BaseVectorStorage,
        relations_vdb: BaseVectorStorage,
        features_vdb: BaseVectorStorage,
        global_config: dict,
        if_check: bool = True
)->Union[BaseGraphStorage]:
    
    if events_vdb.is_empty() or if_check:
        events = extract_events(
            llm=llm,
            video=global_config["video"],
            global_config=global_config,
        )
    else:
        events = events_vdb.get_datas()
    
    if entities_vdb.is_empty() or relations_vdb.is_empty() or if_check:
        entities, relations = extract_entities_and_relations(
            llm=llm,
            embedding_model=embedding_model,
            events=events,
            video=global_config["video"],
            global_config=global_config,
        )
    else:
        entities = entities_vdb.get_datas()
        relations = relations_vdb.get_datas()
    
    # add to event_vdb
    datas_for_vdb = {
        event["id"]:{
            "id": event["id"],
            "description": event["description"],
            "duration": event["duration"],
            "content": event["description"]
        } for event in events
    }
    
    events_vdb.upsert(datas_for_vdb)
    
    # add to entities_vdb
    datas_for_vdb = {
        entity["id"]:{
            "id": entity["id"],
            "descriptions": entity["descriptions"],
            "timestamps": entity["timestamps"],
            "frame_indices": entity["frame_indices"],
            "durations": entity["durations"],
            "events": entity["events"],
            "content": entity["descriptions"]
        } for entity in entities
    }
    
    entities_vdb.upsert(datas_for_vdb)
    
    # add to relations_vdb
    datas_for_vdb = {
        relation["id"]:{
            "id": relation["id"],
            "entity1": relation["entity1"],
            "entity2": relation["entity2"],
            "description": relation["description"],
            "content": relation["description"]
        } for relation in relations
    }
    
    relations_vdb.upsert(datas_for_vdb)
    
    # add to features vdb
    if features_vdb.is_empty():
        video = global_config["video"]
        datas_for_vdb = {}
        frame_path = video.frames_dir
        for event in events:
            duration = event["duration"]
            for i in range(int(duration[0]), int(duration[1])):
                _, _, index = video.get_frames_by_fps(fps=1, duration=(i,i))
                if not index:
                    continue
                frame_dir = os.path.join(frame_path, f"{index[0]}.jpg")
                datas_for_vdb[f"{i}"] = {
                    "content": [frame_dir],
                    "event": event["id"],
                    "frame_dir": frame_dir
                }
        
        features_vdb.upsert(datas_for_vdb)
    
    for event in events:
        if not knowledge_graph_inst.has_node(event["id"]):
            knowledge_graph_inst.upsert_node(
                node_id=event["id"],
                node_data={
                    "type": "event",
                    "id": event["id"],
                    "description": event["description"],
                }
            )
    
    for entity in entities:
        if not knowledge_graph_inst.has_node(entity["id"]):
            knowledge_graph_inst.upsert_node(
                node_id=entity["id"],
                node_data={
                    "type": "entity",
                    "id": entity["id"],
                    "description": f" ".join(entity["descriptions"]),
                }
            )
        for event_id in entity["events"]:
            if not knowledge_graph_inst.has_edge(entity["id"], event_id):
                knowledge_graph_inst.upsert_edge(
                    entity["id"],
                    event_id,
                    edge_data={
                        "type": "belong_to",
                        "id1": entity["id"],
                        "id2": event_id,
                    }
                )
    
    for relation in relations:
        if not knowledge_graph_inst.has_edge(relation["entity1"], relation["entity2"]):
            knowledge_graph_inst.upsert_edge(
                relation["entity1"],
                relation["entity2"],
                edge_data={
                    "type": "relation",
                    "id1": relation["entity1"],
                    "id2": relation["entity2"],
                    "description": relation["description"]
                }
            )
    
    # add event time relation
    total_events = len(events)
    for i in range(total_events-1):
        if not knowledge_graph_inst.has_edge(events[i]["id"], events[i+1]["id"]):
            knowledge_graph_inst.upsert_edge(
                events[i]["id"],
                events[i+1]["id"],
                edge_data={
                    "type": "event_time_relation",
                    "id1": events[i]["id"],
                    "id2": events[i+1]["id"],
                }
            )
    
    return knowledge_graph_inst

def tree_search(
    query: str,
    llm: BaseLanguageModel,
    video: VideoRepresentation,
    events_vdb: BaseVectorStorage,
    entities_vdb: BaseVectorStorage,    
    features_vdb: BaseVectorStorage,
):
    tree_search = TreeSearch(query, llm, video, events_vdb, entities_vdb, features_vdb)
    tree_search.search()
    
    tree_information = tree_search.collect_tree_information()
    return tree_information

def generate_self_consistency_result(
    tree_information: dict,
    llm: BaseLanguageModel,
    self_consistency_num: int = 8,
):
    all_input_prompts = []
    
    for sa_node in tree_information:
        input_prompt = sa_node["input_prompt"]
        for _ in range(self_consistency_num):
            all_input_prompts.append(
                {
                    "text": input_prompt,
                }
            )
    
    # batched generate
    all_responses = llm.batch_generate_response(all_input_prompts)
    
    for i in range(0, len(all_responses), self_consistency_num):
        node_responses = all_responses[i:i+self_consistency_num]
        tree_information[int(i/self_consistency_num)]["responses"] = node_responses
    
    return tree_information

def calculate_sa_score(
    node_results: list,
):
    scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en")
    
    # calculate the score of each node
    for i, node_result in enumerate(node_results):
        responses = node_result["responses"]
        answer_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        thoughts_counts = {"A": [], "B": [], "C": [], "D": []}
        node_score = {"A": [0.0, 0.0], "B": [0.0, 0.0], "C": [0.0, 0.0], "D": [0.0, 0.0]}
        
        def extract_json(llm_output):
            json_pattern = r'{.*}'
            match = re.search(json_pattern, llm_output, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    return parsed_json
                except json.JSONDecodeError as e:
                    return f"JSON parse error: {e}"
            else:
                return None
        
        for response in responses:
            json_result = extract_json(response)
            if isinstance(json_result, dict) and "Answer" in json_result.keys() and "Analysis" in json_result.keys():
                answer = json_result["Answer"]
                thoughts = json_result["Analysis"]
                if answer in answer_counts.keys():
                    answer_counts[answer] += 1
                    thoughts_counts[answer].append(thoughts)
        
        self_consistency_num = len(responses)
        
        # cal answer score
        for choice in answer_counts.keys():
            node_score[choice][0] = 1.0 * answer_counts[choice] / self_consistency_num
            
        # cal thoughts score
        for choice in answer_counts.keys():
            if len(thoughts_counts[choice]) > 1:
                text1 = []
                text2 = []
                f1_scores = []
                for i in range(len(thoughts_counts[choice])):
                    for j in range(len(thoughts_counts[choice])):
                        if i != j:
                            text1.append(thoughts_counts[choice][i])
                            text2.append(thoughts_counts[choice][j])
                    f1,_, _ = scorer.score(text1, text2)
                    f1 = f1.mean().item()
                    f1_scores.append(f1)
                node_score[choice][1] = 1.0 * sum(f1_scores) / len(f1_scores)
        
        node_result["scores"] = node_score
    
    return node_results

def choose_best_sa_answer(
    score_results: list,
    alpha: float = 0.3,
):
    node_best_choices = []
    for score_result in score_results:
        scores = {
            choice: score_result["scores"][choice][0] * alpha + score_result["scores"][choice][1] * (1 - alpha)
            for choice in score_result["scores"].keys()
        }
        best_choice = max(scores, key=scores.get)
        best_score = scores[best_choice]
        node_best_choices.append((best_choice, best_score))
    
    tree_best_choice = max(node_best_choices, key=lambda x: x[1])
    return tree_best_choice[0]
    
    
            
            
            
                    
            