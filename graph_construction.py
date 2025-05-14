from AVA.ava import AVA
from dataset.videomme import VideoMME
from llms.QwenVL import QwenVL

if __name__ == "__main__":
    video_id = 601
    
    llm = QwenVL()
    dataset = VideoMME()
    video = dataset.get_video(video_id)
    
    ava = AVA(
        video=video,
        llm_model=llm,
    )
    ava.construct()
    
    