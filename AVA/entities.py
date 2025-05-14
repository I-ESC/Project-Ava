import re
import os
import json
import shutil
import time
from typing import Union
from json_repair import repair_json
from embeddings.BaseEmbeddingModel import BaseEmbeddingModel
from llms.BaseModel import BaseLanguageModel, BaseVideoModel
from video_utils import VideoRepresentation
from .prompt import PROMPTS
from .utils import compute_mdhash_id, clean_str, clean_json
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


########## clustering ##########
def get_entity_embeddings(
    entities: list[dict],
    embedding_model: BaseEmbeddingModel,
    mode: str = "text",
):
    if mode not in {"text", "visual", "hybrid"}:
        raise ValueError("Mode must be 'text', 'visual', or 'hybrid'")

    embeddings = []

    if mode == "text":
        descriptions = [entity["description"][0] for entity in entities]
        text_features = embedding_model.get_text_features(descriptions)
        embeddings = [text_feature for text_feature in text_features.tolist()]
    else:
        raise ValueError("Mode only support text")

    return embeddings

    
def clusters_by_kmeans(
    entities: list[dict],
    relations: list[dict],
    embedding_model: BaseEmbeddingModel,
    file_path: str,
    global_config: dict,
    compression_ratio: float = 0.5,
    mode="text",
):
    assert 0 < compression_ratio <= 1, "Compression ratio must be between 0 and 1."
    embeddings = get_entity_embeddings(entities, embedding_model, mode=mode)
    
    n_samples = len(embeddings)
    n_clusters = max(1, int(n_samples * compression_ratio))
    
    keams = KMeans(n_clusters=n_clusters)
    keams.fit(embeddings)
    labels = keams.labels_.tolist()
    
    merged_entities = []
    clusters = [[] for _ in range(len(set(labels)))]
    
    for entity, label in zip(entities, labels):
        clusters[label].append(entity)
    
    for cluster in clusters:
        descriptions = [entity["description"][0] for entity in cluster]
        timestamps = sorted(list(set([timestamp for entity in cluster for timestamp in entity["timestamps"]])))
        frame_indices = sorted(list(set([frame_index for entity in cluster for frame_index in entity["frame_indices"]])))
        duration = sorted([entity["duration"] for entity in cluster], key=lambda x: x[0])
        events = [entity["event"] for entity in cluster]

        new_id = compute_mdhash_id("".join(descriptions), prefix="Entity-")
        merged_entity = {
            "id": new_id,
            "descriptions": descriptions,
            "timestamps": timestamps,
            "frame_indices": frame_indices,
            "durations": duration,
            "events": events,
        }
        
        merged_entities.append(merged_entity)
        
        # rewrite relations
        cluster_entity_ids = [entity["id"] for entity in cluster]
        for relation in relations:
            if relation["entity1"] in cluster_entity_ids:
                relation["entity1"] = new_id
            if relation["entity2"] in cluster_entity_ids:
                relation["entity2"] = new_id
    
    return merged_entities, relations

#################################################################################################

def update_response(response: Union[str, list], response_file: str):
    if not os.path.exists(response_file):
        with open(response_file, "w") as f:
            json.dump([], f)

    with open(response_file, "r") as f:
        data = json.load(f)
        
    if isinstance(response, str):
        data.append(response)
    elif isinstance(response, list):
        data.extend(response)
    
    with open(response_file, "w") as f:
        json.dump(data, f, indent=4)

def clear_response(response_file: str):
    if os.path.exists(response_file):
        os.remove(response_file)

def validate_entities_relations(
    entities: list[dict], 
    relations: list[dict],
    event_retries: list[int],
    event_idx: int):
    if not entities:
        event_retries[event_idx] += 1
        raise ValueError("No entities detected.")
    
    for entity in entities:
        if len(entity.keys()) < 3:
            event_retries[event_idx] += 1
            raise ValueError("Invalid entity format.")

        if not isinstance(entity["Entity_name"], str):
            event_retries[event_idx] += 1
            raise ValueError("Entity name must be a string.")

        if not isinstance(entity["Entity_description"], str):
            event_retries[event_idx] += 1
            raise ValueError("Entity description must be a string.")
        
        if not isinstance(entity["Index"], list) or not all(isinstance(i, int) for i in entity["Index"]):
            event_retries[event_idx] += 1
            raise ValueError("Entity index must be a list of integers.")
    
    if relations:
        for relation in relations:
            if len(relation) < 3:
                event_retries[event_idx] += 1
                raise ValueError("Invalid relation format.")

            if not isinstance(relation["Entity1"], str):
                event_retries[event_idx] += 1
                raise ValueError("Relation entity1 must be a string.")
            
            if not isinstance(relation["Entity2"], str):
                event_retries[event_idx] += 1
                raise ValueError("Relation entity2 must be a string.")
            
            if not isinstance(relation["Relation_description"], str):
                event_retries[event_idx] += 1
                raise ValueError("Relation description must be a string.")
        
        
    

def format_entities_and_relations(
    entities: list[dict],
    relations: list[dict],
    timestamps: list[float],
    frame_indices: list[int],
    event: dict,
):
    formatted_entities = []
    entities_set = set()
    for entity in entities:
        if len(entity.keys()) < 3:
            continue
        if entity["Entity_name"] in entities_set:
            continue
        formatted_entity = {
            "id": compute_mdhash_id(entity["Entity_name"]+entity["Entity_description"], prefix="Entity-"),
            "name": clean_str(entity["Entity_name"]),
            "description": [clean_str(entity["Entity_description"])],
            "timestamps": [timestamps[i] for i in entity["Index"] if i>=0 and i<len(timestamps)],
            "frame_indices": [frame_indices[i] for i in entity["Index"] if i>=0 and i<len(frame_indices)],
            "duration": event["duration"],
            "event": event["id"],
        }
        
        if not formatted_entity["timestamps"]:
            continue
        formatted_entities.append(formatted_entity)
        entities_set.add(formatted_entity["name"])
    
    formatted_relations = []
    for relation in relations:  
        if len(relation.keys()) < 3:
            continue
        entity1 = clean_str(relation["Entity1"])
        entity2 = clean_str(relation["Entity2"])
        if entity1 in entities_set and entity2 in entities_set and entity1 != entity2:
            description = clean_str(relation["Relation_description"])
            formatted_relation = {
                "id": compute_mdhash_id(entity1 + entity2 + description, prefix="Relation-"),
                "entity1": next((entity["id"] for entity in formatted_entities if entity["name"] == entity1), None),
                "entity2": next((entity["id"] for entity in formatted_entities if entity["name"] == entity2), None),
                "description": description,
            }
            formatted_relations.append(formatted_relation)
    
    return formatted_entities, formatted_relations
    

def batch_generate_entities_and_relations(
    llm: BaseVideoModel,
    events: list[dict],
    video: VideoRepresentation,
    file_path: str,
    global_config: dict,
    max_retries: int = 5,
    batch_size: int = 32,
):
    done_entities_file = os.path.join(file_path, "entities.json")
    done_relations_file = os.path.join(file_path, "relations.json")
    
    if os.path.exists(done_entities_file) and os.path.exists(done_relations_file):
        with open(done_entities_file, "r") as f:
            entities = json.load(f)
        with open(done_relations_file, "r") as f:
            relations = json.load(f)
        
        if entities and relations:
            return entities, relations
    
    response_file = os.path.join(file_path, "responses.json")
    entities_file = os.path.join(file_path, "entities_cache.json")
    relations_file = os.path.join(file_path, "relations_cache.json")

    if os.path.exists(entities_file) and os.path.exists(relations_file):
        with open(entities_file, "r") as f:
            processed_entities = json.load(f)
        with open(relations_file, "r") as f:
            processed_relations = json.load(f)
    else:
        num_events = len(events)
        processed_entities = [None] * num_events
        processed_relations = [None] * num_events

    event_retries = [0] * len(events)
    processed_timestamps = [None] * len(events)
    processed_frame_indices = [None] * len(events)
    
    for event_idx, event in enumerate(events):
        _, timestamps, frame_indices = video.get_frames_by_num(num_frames=8, duration=event["duration"])
        processed_timestamps[event_idx] = timestamps
        processed_frame_indices[event_idx] = frame_indices

    remaining_indices = [
        i for i, entities in enumerate(processed_entities)
        if entities is None
    ]

    while remaining_indices:
        batch_indices = remaining_indices[:batch_size]
        batch_events = [events[i] for i in batch_indices]
        batch_inputs = []

        for event_idx in batch_indices:
            event = events[event_idx]
            duration = event["duration"]
            frames, timestamps, frame_indices = video.get_frames_by_num(num_frames=8, duration=duration)
            batch_inputs.append({
                "text": PROMPTS["entity_relation_extraction"],
                "video": frames,
            })

        try:
            print(f"Extracting entities from {batch_events[0]['duration'][0]} to {batch_events[-1]['duration'][-1]}")
            batch_responses = llm.batch_generate_response(batch_inputs=batch_inputs, max_new_tokens=1024, temperature=0.5)

            for idx, response in enumerate(batch_responses):
                event_idx = batch_indices[idx]

                try:
                    clean_response = clean_json(response)
                    clean_response = repair_json(clean_response)
                    result_json = json.loads(clean_response)

                    update_response(clean_response, response_file)

                    entities = result_json.get("Entities", [])
                    relations = result_json.get("Relations", [])

                    validate_entities_relations(entities, relations, event_retries, event_idx)
                    
                    processed_entities[event_idx] = entities
                    processed_relations[event_idx] = relations

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error processing response for event {events[event_idx]['id']}: {e}")
                    event_retries[event_idx] += 1

        except Exception as e:
            print(f"Batch processing failed: {e}")
            for idx in batch_indices:
                event_retries[idx] += 1

        with open(entities_file, "w") as f:
            json.dump(processed_entities, f, indent=4)
        with open(relations_file, "w") as f:
            json.dump(processed_relations, f, indent=4)

        remaining_indices = [
            idx for idx in remaining_indices
            if event_retries[idx] < max_retries and processed_entities[idx] is None
        ]

    all_entities = []
    all_relations = []

    for i, event in enumerate(events):
        if processed_entities[i] is not None and processed_relations[i] is not None:
            formatted_entities, formatted_relations = format_entities_and_relations(
                entities=processed_entities[i],
                relations=processed_relations[i],
                timestamps=processed_timestamps[i],
                frame_indices=processed_frame_indices[i],
                event=event
            )
            all_entities.extend(formatted_entities)
            all_relations.extend(formatted_relations)

    return all_entities, all_relations

def extract_entities_and_relations(
    llm: BaseVideoModel,
    embedding_model: BaseEmbeddingModel,
    events: list[dict],
    video: VideoRepresentation,
    global_config: dict,
):
    file_path = os.path.join(global_config["working_dir"], "entities")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    # if there exists entities and relations, means have extracted entities and relations as well as merged them
    if os.path.exists(os.path.join(file_path, "entities.json")) and os.path.exists(os.path.join(file_path, "relations.json")):
        with open(os.path.join(file_path, "entities.json"), "r") as f:
            entities = json.load(f)
        with open(os.path.join(file_path, "relations.json"), "r") as f:
            relations = json.load(f)
            
        return entities, relations
    
    # extract entities and relations
    unmerged_entities_file = os.path.join(file_path, "unmerged_entities.json")
    unmerged_relations_file = os.path.join(file_path, "unmerged_relations.json")
    
    unmerged_entities, unmerged_relations = batch_generate_entities_and_relations(
        llm=llm,
        events=events,
        video=video,
        file_path=file_path,
        global_config=global_config,
    )
    
    with open(unmerged_entities_file, "w") as f:
        json.dump(unmerged_entities, f, indent=4)
    
    with open(unmerged_relations_file, "w") as f:
        json.dump(unmerged_relations, f, indent=4)
    
    merged_entities, merged_relations = clusters_by_kmeans(
        entities=unmerged_entities,
        relations=unmerged_relations,
        embedding_model=embedding_model,
        file_path=file_path,
        global_config=global_config,
        compression_ratio=0.4,
        mode="text",
    )
    
    entities_file = os.path.join(file_path, "entities.json")
    relations_file = os.path.join(file_path, "relations.json")
    
    with open(entities_file, "w") as f:
        json.dump(merged_entities, f, indent=4)
        
    with open(relations_file, "w") as f:
        json.dump(merged_relations, f, indent=4)
        
    return merged_entities, merged_relations