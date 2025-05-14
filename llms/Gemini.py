import base64
import io
import asyncio
import random
import itertools
from PIL import Image
from openai import OpenAI
from llms.BaseModel import BaseVideoModel, BaseLanguageModel

API_KEY = "YOUR_API_KEY"

class Gemini(BaseVideoModel, BaseLanguageModel):
    def __init__(self, model_type="gemini-1.5-pro"):
        self.model_type = model_type
        self.key = API_KEY

    def generate_response(self, inputs):
        assert "text" in inputs.keys(), "Please provide a text prompt."
        
        model = OpenAI(
            api_key=self.key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        messages = [{"role": "system", "content": "You are a helpful assistant."}]

        if "video" in inputs.keys():
            images = [encode_image(image) for image in inputs["video"]]
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": inputs["text"]},
                    *[
                        {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{img}', "detail": "low"}}
                        for img in images
                    ]
                ]
            })
        else:
            messages.append({"role": "user", "content": inputs["text"]})

        response = model.chat.completions.create(
            model=self.model_type,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content

    async def generate_response_async(self, inputs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.generate_response, inputs)

    async def _generate_batch_response(self, batch_inputs):
        tasks = [self.generate_response_async(inputs) for inputs in batch_inputs]
        responses = await asyncio.gather(*tasks)
        return responses

    def batch_generate_response(self, batch_inputs):
        return asyncio.run(self._generate_batch_response(batch_inputs))


def encode_image(image):
    if isinstance(image, str):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image, bytes):
        return base64.b64encode(image).decode('utf-8')
    elif isinstance(image, Image.Image):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        raise ValueError("Unsupported image format.")