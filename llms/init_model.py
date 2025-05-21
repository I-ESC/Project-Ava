from llms.QwenLM import QwenLM
from llms.QwenVL import QwenVL
from llms.Gemini import Gemini

model_zoo = {
    "qwenlm": QwenLM,
    "qwenvl": QwenVL,
    "gemini": Gemini
}

def init_model(model_name, num_gpus=1):
    if model_name not in model_zoo:
        supported_models = ", ".join(model_zoo.keys())
        raise ValueError(f"Model {model_name} not found in model_zoo. Supported models: {supported_models}")
    return model_zoo[model_name](tp=num_gpus)