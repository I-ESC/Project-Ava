import json
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from video_utils import VideoRepresentation
from typing import Union

class LVBench(Dataset):
    def __init__(self, json_file="datas/LVBench/LVBench.json", videos_path="datas/LVBench/videos", work_path="AVA_cache/LVBench/"):
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
            video_info["video_path"] = os.path.join(videos_path, f'{video_info["key"]}.mp4')

    def __len__(self):
        return len(self.video_list)

    def get_video_info(self, video_id:int):
        video_info = self.video_infos[video_id-1]

        return video_info
    
    def get_video(self, video_id):        
        source_path = self.video_infos[video_id-1]["video_path"]
        work_path = os.path.join(self.work_path, f"{video_id}")
        if not os.path.exists(work_path):
            os.makedirs(work_path)
        
        return VideoRepresentation(source_path, work_path)