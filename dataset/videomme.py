import json
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from video_utils import VideoRepresentation
from typing import Union

class VideoMME(Dataset):
    def __init__(self, json_file="datas/VideoMME/VideoMME.json", videos_path="datas/VideoMME/videos", work_path="AVA_cache/VideoMME/"):
        """
        Args:
            json_file (string): Path to the JSON file with video data.
            videos_path (string): Directory with all the videos.
        Self:
            video_info: 
                video_path -> source video path
                others -> other video dataset information
            work_path: directory to save the processed video frames
        """
        with open(json_file, 'r') as f:
            self.video_infos = json.load(f)
        
        self.videos_path = videos_path
        self.work_path = work_path
        
        for video_info in self.video_infos:
            video_info["video_path"] = os.path.join(videos_path, f'{video_info["videoID"]}.mp4')

    def __len__(self):
        return len(self.video_list)

    def get_video_info(self, video_id: Union[int, str]):
        if isinstance(video_id, str):
            video_id = int(video_id)
        
        idx_start = (video_id - 1) * 3
        idx_end = idx_start + 3
            
        qas = self.video_infos[idx_start:idx_end]
        for qa in qas:
            question = qa["question"]
            options = qa["options"]
            qa["question"] = f"{question}\n{options[0]}\n{options[1]}\n{options[2]}\n{options[3]}"

        qas = [{k:v for k,v in qa.items() if k in ["question", "answer", "domain", "task_type", "sub_category", "url", "task_type"]} for qa in qas]
        format_video_info = {"video_id": video_id, "qa": qas}

        return format_video_info
    
    def get_video(self, video_id: Union[int, str]):
        if isinstance(video_id, str):
            video_id = int(video_id)
        
        idx = (video_id - 1) * 3
        source_path = self.video_infos[idx]["video_path"]
        work_path = os.path.join(self.work_path, f"{video_id}")
        if not os.path.exists(work_path):
            os.makedirs(work_path)
        
        return VideoRepresentation(source_path, work_path)