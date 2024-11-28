import requests
import os
from typing import Dict, List, Optional
from huggingface_hub import hf_hub_download, snapshot_download

class HuggingFaceAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://huggingface.co/api"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def search_models(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        url = f"{self.base_url}/models"
        params = {"search": query}
        if filters:
            params.update(filters)
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def download_model(self, model_id: str, download_dir: str, progress_callback=None) -> str:
        try:
            # Create the download directory if it doesn't exist
            os.makedirs(download_dir, exist_ok=True)
            
            # Download the complete model snapshot
            local_dir = snapshot_download(
                repo_id=model_id,
                token=self.api_key,
                local_dir=os.path.join(download_dir, model_id.split('/')[-1]),
                local_dir_use_symlinks=False,
                resume_download=True,
                ignore_patterns=["*.safetensors", "*.bin"] if "text-generation" not in model_id else None,
                max_workers=4
            )
            
            if progress_callback:
                progress_callback(100)
            return local_dir
        except Exception as e:
            raise Exception(f"Failed to download model: {str(e)}")

    def get_model_info(self, model_id: str) -> Dict:
        url = f"{self.base_url}/models/{model_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def list_model_files(self, model_id: str) -> List[str]:
        url = f"{self.base_url}/models/{model_id}/tree"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_model_tags(self, model_id: str) -> List[str]:
        model_info = self.get_model_info(model_id)
        return model_info.get('tags', [])

    def get_model_downloads(self, model_id: str) -> int:
        model_info = self.get_model_info(model_id)
        return model_info.get('downloads', 0)
        
    def run_inference(self, model_id: str, inputs: str) -> Dict:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        response = requests.post(url, headers=self.headers, json={"inputs": inputs})
        response.raise_for_status()
        return response.json()
