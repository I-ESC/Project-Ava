import json
import random
import copy
import time

import numpy as np
from llms.BaseModel import BaseLanguageModel, BaseVideoModel
from typing import Union, List, Dict
from .storage import BaseVectorStorage
from .prompt import PROMPTS
from json_repair import repair_json
from video_utils import VideoRepresentation

def tri_view_retrieval(
    query: str,
    llm: Union[BaseLanguageModel, BaseVideoModel],
    events_vdb: BaseVectorStorage,
    entities_vdb: BaseVectorStorage,
    features_vdb: BaseVectorStorage,
    generation: int = 0
):
    top_k_for_events = 5
    top_k_for_entities = 5
    top_k_for_features = 32
    S = 1
    
    start_time = time.time()
    
    keywords_prompt = PROMPTS["keyword_extraction"].format(
        input_text=query
    )
    
    rewrite_entity_prompt = PROMPTS["query_rewrite_for_entity_retrieval"].format(
        input_text=query
    )
    
    rewrite_feature_prompt = PROMPTS["query_rewrite_for_visual_retrieval"].format(
        input_text=query
    )
    
    batch_inputs = [
        {
            "text": keywords_prompt
        },
        {
            "text": rewrite_entity_prompt
        },
        {
            "text": rewrite_feature_prompt
        }
    ]
    
    batch_outputs = llm.batch_generate_response(batch_inputs)
    
    keywords_response = batch_outputs[0]
    rewrite_entity_response = batch_outputs[1]
    rewrite_feature_response = batch_outputs[2]
    
    events_result = events_vdb.query(keywords_response, top_k=top_k_for_events)
    entities_result = entities_vdb.query(rewrite_entity_response, top_k=top_k_for_entities)
    features_result = features_vdb.query(rewrite_feature_response, top_k=top_k_for_features)

    # event simalarity scores
    events_from_events = {event["id"]: event["__metrics__"] for event in events_result}
    events_from_entities = {}
    for entity in entities_result:
        for event_id in entity["events"]:
            if event_id not in events_from_entities:
                events_from_entities[event_id] = entity["__metrics__"]
            else:
                events_from_entities[event_id] += entity["__metrics__"]
    events_from_features = {}
    for feature in features_result:
        if feature["event"] not in events_from_features:
            events_from_features[feature["event"]] = feature["__metrics__"]
        else:
            events_from_features[feature["event"]] += feature["__metrics__"]
    
    events_from_events = sorted(events_from_events.items(), key=lambda x: x[1], reverse=True)
    events_from_entities = sorted(events_from_entities.items(), key=lambda x: x[1], reverse=True)
    events_from_features = sorted(events_from_features.items(), key=lambda x: x[1], reverse=True)
    
    # Normalized Borda Count
    event_scores = {}
    events_from_events_scores_sum = sum([score for _, score in events_from_events])
    events_from_entities_scores_sum = sum([score for _, score in events_from_entities])
    events_from_features_scores_sum = sum([score for _, score in events_from_features])
    
    for event_id, score in events_from_events:
        event_scores[event_id] = score / events_from_events_scores_sum * S
    for event_id, score in events_from_entities:
        if event_id not in event_scores:
            event_scores[event_id] = score / events_from_entities_scores_sum * S
        else:
            event_scores[event_id] += score / events_from_entities_scores_sum * S
    for event_id, score in events_from_features:
        if event_id not in event_scores:
            event_scores[event_id] = score / events_from_features_scores_sum * S
        else:
            event_scores[event_id] += score / events_from_features_scores_sum * S
            
    results = [{"event_id": [event_id], "query": [query], "score": event_scores[event_id], "generation": generation+1} for event_id in event_scores]
    
    for result in results:
        event_id = result["event_id"][0]
        event_data = events_vdb.get_data(event_id)
        result["event_data"] = [event_data]

    results = sorted(results, key=lambda x: x["event_data"][0]["duration"][0])
    
    end_time = time.time()
    
    return results

class EventList:
    def __init__(self, datas:List[Dict], Limited_length:int=16):
        self.datas = []
        self.Limited_length = Limited_length
        self.max_generation = max([event_chunk["generation"] for event_chunk in self.datas]) if self.datas else 0
        self.event_positions = self.get_event_positions()
        self.insert(datas)
        
    def insert(self, new_datas: List[Dict]):
        for new_data in new_datas:
            event_id = new_data["event_id"][0]
            if event_id not in self.event_positions:
                self.datas.append(new_data)
        
        # self.drop()
        self.merge_adjacent_events()
        self.drop()
        self.event_positions = self.get_event_positions()
    
    def get_event_positions(self):
        positions = {}
        for i, event_chunk in enumerate(self.datas):
            for j, event_id in enumerate(event_chunk["event_id"]):
                positions[event_id] = [i, j]
        
        return positions
        
    def merge_adjacent_events(self):
        self.datas.sort(key=lambda x: x["event_data"][0]["duration"][0])
        merged_datas = []
        
        for event_chunk in self.datas:
            if not merged_datas:
                merged_datas.append(event_chunk)
                continue
            
            last_chunk = merged_datas[-1]
            last_end = last_chunk["event_data"][-1]["duration"][1]
            current_start = event_chunk["event_data"][0]["duration"][0]
            current_end = event_chunk["event_data"][-1]["duration"][1]
            
            if last_end >= current_start:
                if last_end >= current_end:
                    continue
                else:
                    if last_end > current_start:
                        overlap_start_index = next(i for i, event in enumerate(event_chunk["event_data"]) if event["duration"][0] >= last_end)
                        # overlap_start_index = next((i for i, event in enumerate(event_chunk["event_data"]) if event["duration"][1] > last_end))
                        event_chunk["event_data"] = event_chunk["event_data"][overlap_start_index:]

                    last_chunk["event_data"].extend(event_chunk["event_data"])
                    # last_chunk["score"] = max(last_chunk["score"], event_chunk["score"])
                    last_chunk["score"] = last_chunk["score"] + event_chunk["score"]
                    last_chunk["generation"] = max(last_chunk["generation"], event_chunk["generation"])
                    last_chunk["event_id"].extend(event_chunk["event_id"])
                    last_chunk["query"].extend(event_chunk["query"])
            else:
                merged_datas.append(event_chunk)
        
        self.datas = merged_datas
        self.event_positions = self.get_event_positions()
    
    def format_information(self, limited_ratio=None):
        if limited_ratio is not None:
            candidate_datas = sorted(self.datas, key=lambda x: x["score"], reverse=True)
            candidate_datas = candidate_datas[:int(len(candidate_datas) * limited_ratio)]
        else:
            candidate_datas = self.datas
        
        formatted_information = ""
        sorted_datas = sorted(candidate_datas, key=lambda x: x["event_data"][0]["duration"][0])
        
        for event_chunk in sorted_datas:
            description = ""
            if len(event_chunk["event_data"]) == 1:
                description = event_chunk["event_data"][0]["description"]
            else:
                description_list = [event["description"] for event in event_chunk["event_data"]]
                description = "After, ".join(description_list)
                
            formatted_information += "{}s - {}s, {}\n".format(
                event_chunk["event_data"][0]["duration"][0],
                event_chunk["event_data"][-1]["duration"][1],
                description
            )
        return formatted_information

    def structed_information(self):
        return self.datas
    
    def format_durations(self, limited_frames:int=128, limited_ratio=None):
        if limited_ratio is not None:
            candidate_datas = sorted(self.datas, key=lambda x: x["score"], reverse=True)
            candidate_datas = candidate_datas[:int(len(candidate_datas) * limited_ratio)]
        else:
            candidate_datas = self.datas
        
        sorted_datas = sorted(candidate_datas, key=lambda x: x["event_data"][0]["duration"][0])
        durations = []
        for event_chunk in sorted_datas:
            durations.append([event_chunk["event_data"][0]["duration"][0], event_chunk["event_data"][-1]["duration"][1]])
        return durations
    
    def drop(self, drop_ratio=None):
        if drop_ratio is not None:
            cur_length = len(self.datas)
            drop_length = int(cur_length * drop_ratio)
            self.datas.sort(key=lambda x: x["score"], reverse=True)
            self.datas = self.datas[:cur_length - drop_length]
            self.datas = sorted(self.datas, key=lambda x: x["event_data"][0]["duration"][0])
        else:
            cur_length = len(self.datas)
            if cur_length > self.Limited_length:
                self.datas.sort(key=lambda x: x["score"], reverse=True)
                self.datas = self.datas[:self.Limited_length]
                
                self.datas = sorted(self.datas, key=lambda x: x["event_data"][0]["duration"][0])
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        return self.datas[index]
    
    def __iter__(self):
        return iter(self.datas)
    
    def __contains__(self, item):
        return item in self.datas


class Node:
    def __init__(self, state, query, action=None, parent=None, initial_event_list=None, events_vdb=None, entities_vdb=None, features_vdb=None, llm=None, video=None):
        self.action_function = {
            "RQ": self._re_query,
            "F": self._forward,
            "B": self._backward,
            "SA": self._prepare_summary_and_answer_input,
            # "CA": self._prepare_checkframe_and_answer_input
        }
        self.queries = [query]
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.event_list = copy.deepcopy(initial_event_list)
        self.events_vdb = events_vdb
        self.entities_vdb = entities_vdb
        self.features_vdb = features_vdb
        self.llm = llm
        self.video = video
        
        self.input_prompt = None
        self.responses = None
        
        self.max_retry_times = 5
        
        self.path = []
        if parent:
            self.path = copy.deepcopy(parent.path)
        if action:
            self.path.append(action)
            
        self.inference_input = None

    def apply_action(self):
        if self.action in self.action_function:
            self.action_function[self.action]()

    def _re_query(self):
        print("Re-query")
        retrieved_information = self.event_list.format_information()
        query_prompt = PROMPTS["re-query"].format(
            video_segments=retrieved_information,
            user_query=self.queries[0]
        )
        sub_query = self.queries[0]
        for i in range(self.max_retry_times):
            try:
                response = self.llm.generate_response({"text": query_prompt})
                response = repair_json(response)
                response = json.loads(response)
                sub_query = response["sub_query"]
                if sub_query:
                    break
            except Exception as e:
                print(f"Retry {i+1} failed, error: {e}")
        print(f"sub_query: {sub_query}")

        events_result = tri_view_retrieval(sub_query, self.llm, self.events_vdb, self.entities_vdb, self.features_vdb, generation=self.event_list.max_generation)
        self.event_list.insert(events_result)
        self.queries.append(sub_query)
        
    def _forward(self):
        print("Forward")
        self.event_list.drop(drop_ratio=0.2)
        for event_chunk in self.event_list:
            event_id = event_chunk["event_id"][0]
            previous_event = self.events_vdb.get_previous_data(event_id)
            if previous_event:
                event_chunk["event_id"].insert(0, previous_event["__id__"])
                event_chunk["event_data"].insert(0, previous_event)
                event_chunk["query"].insert(0, event_chunk["query"][0])
        self.event_list.merge_adjacent_events()
        self.event_list.event_positions = self.event_list.get_event_positions()

    def _backward(self):
        print("Backward")
        self.event_list.drop(drop_ratio=0.2)
        for event_chunk in self.event_list:
            event_id = event_chunk["event_id"][-1]
            next_event = self.events_vdb.get_next_data(event_id)
            if next_event:
                event_chunk["event_id"].append(next_event["__id__"])
                event_chunk["event_data"].append(next_event)
                event_chunk["query"].append(event_chunk["query"][-1])
        self.event_list.merge_adjacent_events()
        self.event_list.event_positions = self.event_list.get_event_positions()

    def _prepare_summary_and_answer_input(self):
        print("Prepare summary and answer input")
        retrieved_information = self.event_list.format_information(limited_ratio=0.75)
        summary_prompt = PROMPTS["summary_and_answer_COT"].format(
            video_segments=retrieved_information,
            user_query=self.queries[0]
        )
        
        self.input_prompt = summary_prompt

        inference_input = {"text": summary_prompt}
        self.inference_input = inference_input
        return inference_input

class TreeSearch:
    ACTIONS = ["SA", "RQ", "F", "B"]
    
    def __init__(self, 
                 query: str, 
                 llm: Union[BaseLanguageModel, BaseVideoModel],
                 video: VideoRepresentation,
                 events_vdb: BaseVectorStorage,
                 entities_vdb: BaseVectorStorage,
                 features_vdb: BaseVectorStorage,
                 max_depth: int = 3):
        self.query = query
        self.llm = llm
        self.video = video
        self.events_vdb = events_vdb
        self.entities_vdb = entities_vdb
        self.features_vdb = features_vdb
        self.max_depth = max_depth
        self.event_list = self.init_event_list(query, llm, events_vdb, entities_vdb, features_vdb)
        self.root = Node(state={}, action="Root", query=query, initial_event_list=self.event_list, events_vdb=events_vdb, entities_vdb=entities_vdb, features_vdb=features_vdb, llm=llm, video=video)
        
        self.actions = self.ACTIONS

    def init_event_list(self, query, llm, events_vdb, entities_vdb, features_vdb):
        events_result = tri_view_retrieval(query, llm, events_vdb, entities_vdb, features_vdb)
        event_list = EventList(events_result)
        return event_list

    def merge_event_lists(self, event_lists, select_ratio=None):
        event_ids = set()
        if select_ratio is not None:
            for event_list in event_lists:
                event_list.drop(drop_ratio=(1-select_ratio))
            
        for event_list in event_lists:
            for event_chunk in event_list:
                for event_id in event_chunk["event_id"]:
                    event_ids.add(event_id)
        
        event_results = [{"event_id": [event_id], "query": [""], "score": 0.0, "generation": 0} for event_id in event_ids]
        for event_result in event_results:
            event_id = event_result["event_id"][0]
            event_data = self.events_vdb.get_data(event_id)
            event_result["event_data"] = [event_data]
            
        return EventList(event_results, Limited_length=1000)
                    
    def search(self):
        self._search(self.root, 0)
        # self._delay_inference()

    def _search(self, node, current_depth):
        if current_depth >= self.max_depth or node.action == "SA" or node.action == "CA":
            return
        
        if current_depth == self.max_depth - 1:
            child_node = Node(state="node", query=node.queries[0], action="SA", parent=node, initial_event_list=node.event_list, events_vdb=self.events_vdb, entities_vdb=self.entities_vdb, features_vdb=self.features_vdb, llm=self.llm, video=self.video)
            child_node.apply_action()
            node.children.append(child_node)
            return

        for action in self.actions:
            child_node = Node(state="node", query=node.queries[0], action=action, parent=node, initial_event_list=node.event_list, events_vdb=self.events_vdb, entities_vdb=self.entities_vdb, features_vdb=self.features_vdb, llm=self.llm, video=self.video)
            child_node.apply_action()
            node.children.append(child_node)
            self._search(child_node, current_depth + 1)
            
    
    def collect_tree_information(self):
        tree_information = []

        def collect_answers(node, depth, path):
            if node.action in ["SA"]:
                tree_information.append({
                    "action": node.action,
                    "depth": depth,
                    "path": path,
                    "input_prompt": node.input_prompt,
                    "structed_information": node.event_list.datas,
                })
            for child in node.children:
                collect_answers(child, depth + 1, path + [child.action])

        collect_answers(self.root, 0, [self.root.action])

        return tree_information
        