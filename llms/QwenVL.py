import torch
from llms.BaseModel import BaseVideoModel
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy import pipeline, GenerationConfig
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy.vl.utils import encode_image_base64

class QwenVL(BaseVideoModel):
    def __init__(self, model_type="Qwen/Qwen2.5-VL-7B-Instruct-AWQ", tp=1):
        """
        Initialize the QwenVL model.

        Args:
            model_type (str): The type or name of the model.
            tp (int): The number of GPUs to use.
        """
        self.pipe = pipeline(model_type, 
                backend_config=TurbomindEngineConfig(session_len=8192*4, tp=tp, cache_max_entry_count=0.3))
    
    def generate_response(self, inputs, max_new_tokens=512, temperature=0.5):
        """
        Generate a response based on the inputs.

        Args:
            inputs (dict): Input data containing text
            {
                "text": str,
                "video": list[Image.Image](optional)
            }

        Returns:
            str: Generated response.
        """
        assert "text" in inputs.keys(), "Please provide a text prompt."
        gen_config = GenerationConfig(do_sample=True, max_new_tokens=max_new_tokens, temperature=temperature)
    
        if "video" in inputs.keys():
            question = ''
            imgs = inputs["video"]
            for i in range(len(imgs)):
                question = question + f'{IMAGE_TOKEN}\n'

            question += inputs["text"]

            content = [{'type': 'text', 'text': question}]
            for img in imgs:
                content.append({'type': 'image_url', 'image_url': {'max_dynamic_patch': 1, 'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'}})

            messages = [dict(role='user', content=content)]
            response = self.pipe(messages, gen_config=gen_config)
            text_response = response.text
        else: 
            response = self.pipe(inputs["text"], gen_config=gen_config)
            text_response = response.text
        
        return text_response

    def batch_generate_response(self, batch_inputs, max_batch_size=64, max_new_tokens=512, temperature=0.5):
        prompts = []
        responses = []
        gen_config = GenerationConfig(do_sample=True, max_new_tokens=max_new_tokens, temperature=temperature)
        
        if "video" in batch_inputs[0].keys():
            for inputs in batch_inputs:
                question = ''
                imgs = inputs["video"]

                question += inputs["text"]

                content = []
                for img in imgs:
                    content.append({'type': 'image_url', 'image_url': {'max_dynamic_patch': 1, 'url': f'data:image/jpeg;base64,{encode_image_base64(img)}'}})
                content.append({'type': 'text', 'text': question})
                messages = [dict(role='user', content=content)]
                prompts.append(messages)
            
            for i in range(0, len(prompts), max_batch_size):
                responses.extend(self.pipe(prompts[i:i+max_batch_size], gen_config=gen_config))
        else:
            for inputs in batch_inputs:
                prompts.append(inputs["text"])
            for i in range(0, len(prompts), max_batch_size):
                responses.extend(self.pipe(prompts[i:i+max_batch_size], gen_config=gen_config))
    
        return [response.text for response in responses]
        