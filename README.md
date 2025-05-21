# Project-Ava
An implementation of Paper "Empowering Agentic Video Analytics Systems with Video Language Models"

> 📰 **Paper:**  [Arxiv](https://arxiv.org/abs/2505.00254)
## 🔧 Key Features
- **Task Definition**: Define five levels of intelligence (L1–L5) for current and future video analysis systems. AVA is the first L4 video analytics system powered by VLMs, enabling open-ended comprehension, reasoning, and analytics—marking a significant advancement.
- **Near-real-time index construction**: AVA employs Event-Knowledge Graphs (EKGs) to construct video index, supporting near real-time indexing even on common edge devices (2 $\times$ RTX 4090).
- **Agentic retrieval and generation**: AVA uses LLM as an agent to proactively explore and retrieve additional event information related to prior results, enabling multi-path reasoning and response generation based on aggregated data.
- **Proposed benchmark**: Proposes AVA-100, an ultra-long video benchmark designed to evaluate video analysis capabilities, comprising 8 videos (each over 10 hours) and 120 manually annotated questions across four scenarios: human daily activities, city walking, wildlife surveillance, and traffic monitoring.
- AVA achieves $62.3\%$ on LVBench, $62.3\%$ on VideoMME-Long, and $75.8\%$ on the proposed AVA-100 benchmark, outperforming mainstream VLMs and Video-RAG methods under the same settings.

---
## 📦 Installation
```bash
git clone https://github.com/I-ESC/Project-AVA.git
cd Project-AVA
conda create -n babel python=3.9
conda activate ava
pip install -r requirements.txt
```

---
## Dataset Preparation
```bash
# supported dataset: LVBench, VideoMME-Long, AVA-100
cd datas/[dataset-name]

# For LVBench and VideoMME-Long
./download.sh

# For AVA-100
mkdir videos
# Then download the video from Google Drive and place it in the videos folder.
```

---
## Graph Construction
```bash
# View llms.init_model.py for supported models
# View dataset.init_dataset.py for supported dataset
# View datas/[dataset_name]/[dataset_name.json] for video_id, LVBench: 1-103, VideoMME: 601-900, AVA-100: 1-8
python graph_construction.py --model [name_of_model] --dataset [name_of_dataset] --video_id [id_of_video] --gpus [num_of_gpus]

# example
python graph_construction.py --model qwenvl --dataset lvbench --video_id 1 --gpus 1
```

## Generate SA Result
```bash
# View datas/[dataset_name]/[dataset_name.json] for question_id
python query_SA.py --model [name_of_model] --dataset [name_of_dataset] --video_id [id_of_video] --question_id [id_of_question]--gpus [num_of_gpus]

# example
python query_SA.py --model qwenlm --dataset lvbench --video_id 1 --question_id 0 --gpus 1
```
## Generate CA Result
```bash
# Before generating the CA Result, the corresponding SA Result must have already been produced.
python query_SA.py --model [name_of_model] --dataset [name_of_dataset] --video_id [id_of_video] --question_id [id_of_question]--gpus [num_of_gpus]

# example
python query_CA.py --model qwenvl --dataset lvbench --video_id 1 --question_id 0 --gpus 1
```

## ❤️ Acknowledgements
AVA is implemented with reference to the following projects：

[LightRAG](https://github.com/HKUDS/LightRAG)

[VideoRAG](https://github.com/HKUDS/VideoRAG)


## 📄 Citation

If you use this repo, please cite our paper:
```bibtex
@article{ava,
  title={Empowering Agentic Video Analytics Systems with Video Language Models},
  author={Yan, Yuxuan and Jiang, Shiqi and Cao, Ting and Yang, Yifan and Yang, Qianqian and Shu, Yuanchao and Yang, Yuqing and Qiu, Lili},
  journal={arXiv preprint arXiv:2505.00254},
  year={2025}
}
```

