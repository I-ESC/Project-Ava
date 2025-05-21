from AVA.ava import AVA
from dataset.init_dataset import init_dataset
from llms.init_model import init_model
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Name of the LLM model to use")
    parser.add_argument("--dataset", required=True, help="Name of the dataset")
    parser.add_argument("--video_id", type=int, required=True, help="ID of the video to process")
    parser.add_argument("--question_id", type=int, required=True, help="ID of the question to process")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    dataset = init_dataset(args.dataset)
    llm = init_model(args.model, args.gpus)
    
    video = dataset.get_video(args.video_id)
    video_info = dataset.get_video_info(video_id = args.video_id)
    
    qas = video_info["qa"]
    
    ava = AVA(
        video=video,
        llm_model=llm,
    )
    
    ava.query_tree_search(qas[args.question_id]["question"], args.question_id)
    
    final_sa_answer = ava.generate_SA_answer(qas[args.question_id]["question"], args.question_id)
    print(final_sa_answer)