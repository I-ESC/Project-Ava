import re
import os
import json
import time
from typing import Union
from matplotlib import pyplot as plt
from llms.BaseModel import BaseLanguageModel, BaseVideoModel
from video_utils import VideoRepresentation
from .prompt import PROMPTS
from bert_score import score, BERTScorer
from .utils import compute_mdhash_id, clean_str

def get_chunk_timestamp(video, chunk_duration, chunk_overlap):
    video_config = video.config
    video_duration = video_config["duration"]

    if chunk_duration <= 0:
        raise ValueError("Chunk duration must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("Chunk overlap must be 0 or greater.")
    if chunk_overlap >= chunk_duration:
        raise ValueError("Chunk overlap must be less than chunk duration.")

    timestamps = []
    start_time = 0

    while start_time < video_duration:
        end_time = start_time + chunk_duration
        if end_time > video_duration:
            timestamps.append((start_time, video_duration))
            break
            
        timestamps.append((start_time, end_time))
        start_time = start_time + chunk_duration - chunk_overlap
    
    return timestamps

def batch_generate_descriptions(
    llm: BaseVideoModel,
    video: VideoRepresentation,
    chunk_durations: list,
    file_path: str,
    batch_size: int,
    global_config: dict,
    max_retries: int = 5
):
    """
    batch generate dense descriptions for each chunk
    """
    descriptions_file = os.path.join(file_path, f"descriptions.json")
    if os.path.exists(descriptions_file):
        with open(descriptions_file, "r") as f:
            descriptions = json.load(f)
    else:
        descriptions = [None] * len(chunk_durations)

    fail_count = {i: 0 for i in range(len(chunk_durations))}
    

    remaining_indices = [i for i, desc in enumerate(descriptions) if desc is None]

    while remaining_indices:
        batch_indices = remaining_indices[:batch_size]
        batch_timestamps = [chunk_durations[i] for i in batch_indices]
        batch_inputs = []
        
        print(f"Generating descriptions for timestamps {batch_timestamps[0][0]} to {batch_timestamps[-1][1]}")

        for duration in batch_timestamps:
            num_frames = global_config["video_chunk_num_frames"]
            frames, _, _ = video.get_frames_by_num(num_frames=num_frames, duration=duration)
            inputs = {
                "text": PROMPTS["generate_description"],
                "video": frames,
            }
            batch_inputs.append(inputs)

        try:
            batch_descriptions = llm.batch_generate_response(batch_inputs, max_new_tokens=512, temperature=0.5)
        except Exception as e:
            print(f"Error generating descriptions for batch {batch_indices}: {e}")
            batch_descriptions = [None] * len(batch_indices)

        for idx, description in zip(batch_indices, batch_descriptions):
            if description is not None:
                descriptions[idx] = description
            else:
                fail_count[idx] += 1

        with open(descriptions_file, "w") as f:
            json.dump(descriptions, f, indent=4)

        remaining_indices = [
            i for i, desc in enumerate(descriptions)
            if desc is None and fail_count[i] < max_retries
        ]

    return descriptions

def semantic_chunking(
    llm: BaseVideoModel,
    video: VideoRepresentation,
    descriptions: list,
    chunk_durations: list,
    file_path: str,
    global_config: dict,
    threshold: float = 0.65,
    reprocess: bool = False,
    window_size: int = 8,
    max_retries: int = 5,
    batch_size: int = 64,
):
    source_file = os.path.join(file_path, "events.json")
    scores_file = os.path.join(file_path, "scores.json")
    
    if os.path.exists(source_file) and not reprocess:
        with open(source_file, "r") as f:
            events = json.load(f)
            # have finished semantic chunking
            if events[-1]["duration"][-1] == video.config["duration"]:
                return events
    else:
        events = []
    
    # start from first-unmerged description
    merged_timestamp = events[-1]["duration"][-1] if events else 0
    chunk_duration = global_config["video_chunk_duration"]
    num_merged_descriptions = merged_timestamp // chunk_duration + (1 if merged_timestamp % chunk_duration > 0 else 0)
    
    unmerged_descriptions = [descriptions[i] for i in range(num_merged_descriptions, len(descriptions))]
    
    scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en")
    
    # cal bert score within window_size
    description_list1 = []
    description_list2 = []
    for i in range(len(unmerged_descriptions)):
        for j in range(i+1, min(i + 1 + window_size, len(unmerged_descriptions))):
            description_list1.append(unmerged_descriptions[i])
            description_list2.append(unmerged_descriptions[j])
    _, recall, _ = scorer.score(description_list1, description_list2)
    recall = recall.tolist()
    
    scores_metric = [[0.0] * len(unmerged_descriptions) for _ in range(len(unmerged_descriptions))]
    unsed_count = 0
    for i in range(len(unmerged_descriptions)):
        for j in range(i+1, min(i + 1 + window_size, len(unmerged_descriptions))):
            scores_metric[i][j] = recall[unsed_count]
            unsed_count += 1
    
    
    def cal_chunk_score(i, j, scores_metric):
        chunk_score = 0
        chunk_count = 0
        for k in range(i, j+1):
            for l in range(k+1, j+1):
                chunk_score += scores_metric[k][l]
                chunk_count += 1
        return 1.0 * chunk_score / chunk_count
    
    partitions = []
    start_index = 0
    while start_index < len(unmerged_descriptions):
        end_index = start_index + 1
        while end_index < len(unmerged_descriptions) and end_index - start_index < window_size and cal_chunk_score(start_index, end_index, scores_metric) > threshold and scores_metric[end_index-1][end_index] > threshold:
            end_index += 1
        partitions.append((start_index, end_index-1))
        start_index = end_index
    
    
    batch_inputs = []
    summary_indices = []
    for i in range(len(partitions)):
        partition = partitions[i]
        if partition[0] == partition[1]:
            continue
        else:
            summary_indices.append(i)
            prompt = PROMPTS["summarize_descriptions"].format(
                inputs=[descriptions[j] for j in range(partition[0], partition[1])]
            )
            batch_inputs.append({"text": prompt})
    
    for i in range(max_retries):
        batch_summaries = llm.batch_generate_response(batch_inputs, max_new_tokens=1024, temperature=0.5, max_batch_size=batch_size)
        if batch_summaries:
            break
        else:
            print(f"Attempt {i} failed. Retrying...")
    
    for i in range(len(partitions)):
        partition = partitions[i]
        if partition[0] == partition[1]:
            start_timestamp = chunk_durations[partition[0]+num_merged_descriptions][0]
            end_timestamp = chunk_durations[partition[1]+num_merged_descriptions][1]
            events.append({
                "duration": [start_timestamp, end_timestamp],
                "description": unmerged_descriptions[partition[0]],
            })
        else:
            start_timestamp = chunk_durations[partition[0]+num_merged_descriptions][0]
            end_timestamp = chunk_durations[partition[1]+num_merged_descriptions][1]
            
            summary_index = summary_indices.index(i)
            summary = batch_summaries[summary_index]
            
            events.append({
                "duration": [start_timestamp, end_timestamp],
                "description": summary,
            })
        
        with open(source_file, "w") as f:
            json.dump(events, f, indent=4)
    
    with open(scores_file, "w") as f:
        json.dump(scores_metric, f, indent=4)
    
    events = format_events(events)
    return events
    
def format_events(events):
    if events:
        formatted_events = []
        for event in events:
            formatted_event = {
                "id": compute_mdhash_id(event["description"], prefix="Event-"),
                "duration": clean_str(event["duration"]),
                "description": clean_str(event["description"]),
            }
            formatted_events.append(formatted_event)
        return formatted_events
        

def extract_events(
    llm:BaseVideoModel,
    video:VideoRepresentation,
    global_config:dict,
):
    # step 1: get small chunk durations
    chunk_durations = get_chunk_timestamp(
        video=video,
        chunk_duration=global_config["video_chunk_duration"],
        chunk_overlap=global_config["video_chunk_overlap"],
    )
    
    file_path = os.path.join(global_config["working_dir"], "events")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    if os.path.exists(os.path.join(file_path, "events.json")):
        with open(os.path.join(file_path, "events.json"), "r") as f:
            events = json.load(f)
            
            events = format_events(events)
            return events
    
    # step 2: batched generate descriptions
    descriptions = batch_generate_descriptions(
        llm=llm,
        video=video,
        chunk_durations=chunk_durations,
        file_path=file_path,
        batch_size=64,
        global_config=global_config,  
    )
    
    # step 3: merge descriptions to events
    events = semantic_chunking(
        llm=llm,
        video=video,
        descriptions=descriptions,
        chunk_durations=chunk_durations,
        file_path=file_path,
        global_config=global_config,
    )
    
    return events
    
