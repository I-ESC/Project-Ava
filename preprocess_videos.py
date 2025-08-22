from dataset.init_dataset import init_dataset, get_video_idx
import os
import argparse
from concurrent.futures import ThreadPoolExecutor

def process_video(dataset, vid):
    dataset.get_video(vid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num_threads", type=int, required=True)
    args = parser.parse_args()
    
    dataset = init_dataset(args.dataset)
    start, end = get_video_idx(args.dataset)
    vids = range(start, end + 1)
    
    num_threads = min(args.num_threads, len(vids))
    
    print(f"Processing videos from {start} to {end} using {num_threads} threads.")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(lambda v: process_video(dataset, v), vids)