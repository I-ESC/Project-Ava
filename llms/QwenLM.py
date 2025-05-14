import torch
from llms.BaseModel import BaseLanguageModel
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

class QwenLM(BaseLanguageModel):
    def __init__(self, model_type="Qwen/Qwen2.5-14B-Instruct-AWQ", tp=4):
        """
        Initialize the QwenLM model.

        Args:
            model_type (str): The type or name of the model.
            tp (int): The number of GPUs to use.
        """
        self.pipe = pipeline(model_type, 
                backend_config=TurbomindEngineConfig(session_len=8192*4, tp=tp, cache_max_entry_count=0.5))
    
    def generate_response(self, inputs, max_new_tokens=512, temperature=0.5):
        """
        Generate a response based on the inputs.

        Args:
            inputs (dict): Input data containing text
            {
                "text": str,
            }

        Returns:
            str: Generated response.
        """
        assert "text" in inputs.keys(), "Please provide a text prompt."
        gen_config = GenerationConfig(do_sample=True, max_new_tokens=max_new_tokens, temperature=temperature)
    
        response = self.pipe(inputs["text"], gen_config=gen_config)
        text_response = response.text
        
        return text_response

    def batch_generate_response(self, batch_inputs, max_batch_size=64, max_new_tokens=512, temperature=0.5):
        prompts = []
        responses = []
        gen_config = GenerationConfig(do_sample=True, max_new_tokens=max_new_tokens, temperature=temperature)
        
        for inputs in batch_inputs:
            prompts.append(inputs["text"])
        
        for i in range(0, len(prompts), max_batch_size):
            responses.extend(self.pipe(prompts[i:i+max_batch_size], gen_config=gen_config))
        
        responses = [response.text for response in responses]
            
        return responses
        