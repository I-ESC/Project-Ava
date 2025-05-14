from AVA.ava import AVA
from dataset.videomme import VideoMME
from llms.QwenLM import QwenLM

if __name__ == "__main__":
    video_id = 601
    question_id = 0
    
    llm = QwenLM()
    dataset = VideoMME()
    video = dataset.get_video(video_id)
    
    ava = AVA(
        video=video,
        llm_model=llm,
    )
    
    video_info = dataset.get_video_info(video_id = video_id)
    qas = video_info["qa"]
    
    ava.query_tree_search(qas[question_id]["question"], question_id)
    
    final_sa_answer = ava.generate_SA_answer(qas[question_id]["question"], question_id)
    print(final_sa_answer)
    
    