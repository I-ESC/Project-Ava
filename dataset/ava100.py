import json
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from video_utils import VideoRepresentation
from typing import Union

class AVA100(Dataset):
    def __init__(self, json_file="datas/AVA100", videos_path="datas/AVA100/videos", work_path="AVA_cache/AVA100/"):
        """
        Args:
            json_file (string): Path to the JSON file with video data.
            videos_path (string): Directory with all the videos.
        """
        json_file_list = [
            os.path.join(json_file, "ego.json"),
            os.path.join(json_file, "citytour.json"),
            os.path.join(json_file, "wildlife.json"),
            os.path.join(json_file, "traffic.json"),
        ]
        self.video_infos_list = []
        for json_file in json_file_list:
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    self.video_infos_list.append(json.load(f))
            else:
                self.video_infos_list.append([])
        
        self.videos_path = videos_path
        self.work_path = work_path
    
    def __len__(self):
        pass
    
    def get_video_info(self, video_id: Union[int, str]):
        video_id = int(video_id)
        video_list_idx = (video_id-1) // 2
        video_idx = (video_id-1) % 2
        video_info = self.video_infos_list[video_list_idx][video_idx]
        video_info["video_path"] = os.path.join(self.videos_path, f'{video_info["video_key"]}.mp4')
        
        qas = video_info["qa"]
        for qa in qas:
            question = qa["query"]
            options = qa["options"]
            concat_question = f"{question}\n{options[0]}\n{options[1]}\n{options[2]}\n{options[3]}"
            qa["question"] = concat_question
        
        video_info["qa"] = qas
            
        return video_info
    
    def get_video(self, video_id: Union[int, str]):
        video_info = self.get_video_info(video_id)
        video_path = video_info["video_path"]
        work_path = os.path.join(self.work_path, f"{video_id}")
        if not os.path.exists(work_path):
            os.makedirs(work_path)
        
        return VideoRepresentation(video_path, work_path)