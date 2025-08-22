from AVA.ava import AVA
import os
import json
from dataset.init_dataset import init_dataset, get_video_idx
from llms.init_model import init_model
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwenvl", help="Name of the LLM model to use")
    parser.add_argument("--dataset", default="lvbench", help="Name of the dataset")
    parser.add_argument("--video_id", type=int, default=-1, help="ID of the video to process")
    parser.add_argument("--question_id", type=int, help="ID of the question to process")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    if args.video_id != -1:    
        dataset = init_dataset(args.dataset)
        llm = init_model(args.model, args.gpus)
        
        video = dataset.get_video(args.video_id)
        video_info = dataset.get_video_info(video_id = args.video_id)
        
        qas = video_info["qa"]
        
        ava = AVA(
            video=video,
            llm_model=llm,
        )
        
        final_ca_answer = ava.generate_CA_answer(qas[args.question_id]["question"], args.question_id)
        print(final_ca_answer)
    else:
        output_folder = "./outputs"
        json_file = f"{output_folder}/query_CA_{args.dataset}_{args.model}.json"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        if os.path.exists(json_file):
            results = json.load(open(json_file, "r"))
        else:
            results = []
        
        dataset = init_dataset(args.dataset)
        llm = init_model(args.model, args.gpus)
        video_idx = get_video_idx(args.dataset)
        
        last_video_id = results[-1]["video_id"] if results else video_idx[0]
        last_question_id = results[-1]["question_id"] if results else -1
        
        for video_id in range(last_video_id, video_idx[1] + 1):
            video = dataset.get_video(video_id)
            video_info = dataset.get_video_info(video_id=video_id)
            qas = video_info["qa"]
            
            for question_id in range(last_question_id+1, len(qas)):
                print(f"Processing video {video_id}, question {question_id}, \n question: {qas[question_id]['question']}")
                ava = AVA(
                    video=video,
                    llm_model=llm,
                )
                
                final_ca_answer = ava.generate_CA_answer(qas[question_id]["question"], question_id)
                
                results.append({
                    "video_id": video_id,
                    "question_id": question_id,
                    "question": qas[question_id]["question"],
                    "answer": qas[question_id]["answer"],
                    "response": final_ca_answer
                })
                
                with open(json_file, "w") as f:
                    json.dump(results, f, indent=4)
            
            last_question_id = -1