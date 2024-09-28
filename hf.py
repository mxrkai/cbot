from pydantic import BaseModel, Field
from typing import Optional, Union, Generator, Iterator
import os
import logging
from huggingface_hub import HfApi
from transformers import pipeline, set_seed

logging.basicConfig(level=logging.INFO)

class Pipe:
    class Valves(BaseModel):
        NAME_PREFIX: str = Field(
            default="HUGGINGFACE/",
            description="Prefix to be added before model names.",
        )
        HUGGINGFACE_API_URL: str = Field(
            default="https://api-inference.huggingface.co/models/",
            description="Base URL for accessing Hugging Face API endpoints.",
        )
        HUGGINGFACE_API_KEY: str = Field(
            default=os.getenv("HUGGINGFACE_API_KEY", ""),
            description="API key for authenticating requests to the Hugging Face API.",
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()
        self.hf_api = HfApi()

    def fetch_models(self):
        if not self.valves.HUGGINGFACE_API_KEY:
            logging.error("API Key not provided.")
            return [
                {
                    "id": "error",
                    "name": "API Key not provided.",
                },
            ]

        try:
            # List models containing 'gpt' in their ID
            models = [
                {"id": model.modelId, "name": model.modelId}
                for model in self.hf_api.list_models()
                if "gpt" in model.modelId
            ]
            return [
                {
                    "id": model["id"],
                    "name": f'{self.valves.NAME_PREFIX}{model["name"]}',
                }
                for model in models
            ]

        except Exception as e:
            logging.error(f"Failed to fetch models: {e}")
            return [
                {
                    "id": "error",
                    "name": "Could not fetch models, please update the API Key in the valves.",
                },
            ]

    def pipe(self, body: dict, __user__: dict) -> Union[str, Generator, Iterator]:
        logging.info(f"Processing request for model: {body['model']}")

        model_id = body["model"]
        payload = {**body, "model": model_id}

        if "prompt" not in payload:
            logging.error("Prompt not provided in the request body.")
            return "Error: Prompt not provided in the request body."

        try:
            # Set up the text generation pipeline
            device = 0 if os.getenv("CUDA_VISIBLE_DEVICES") else -1
            generator = pipeline(
                "text-generation",
                model=model_id,
                tokenizer=model_id,
                device=device,
            )

            set_seed(42)  # Optional: Ensure reproducibility

            response = generator(
                payload["prompt"],
                max_length=payload.get("max_tokens", 500),
                num_return_sequences=1,
                do_sample=True,
            )

            if payload.get("stream", True):
                for message in response:
                    yield message["generated_text"]
            else:
                return response
        except Exception as e:
            logging.error(f"Failed to process request: {e}")
            return f"Error: {e}"

if __name__ == "__main__":
    pipe = Pipe()
    
    # Test fetch_models
    models = pipe.fetch_models()
    print(models)
    
    # Test pipe method
    test_body = {
        "model": "gpt2",
        "prompt": "Once upon a time",
        "max_tokens": 50,
        "stream": False,
    }
    result = pipe.pipe(test_body, __user__={})
    print(result)
