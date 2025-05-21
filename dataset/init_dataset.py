from dataset.lvbench import LVBench
from dataset.videomme import VideoMME
from dataset.ava100 import AVA100

dataset_zoo = {
    "lvbench": LVBench,
    "videomme": VideoMME,
    "ava100": AVA100  
}

def init_dataset(dataset_name):
    if dataset_name not in dataset_zoo:
        supported_datasets = ", ".join(dataset_zoo.keys())
        raise ValueError(f"Dataset {dataset_name} not found in dataset_zoo. Supported datasets: {supported_datasets}")
    return dataset_zoo[dataset_name]()