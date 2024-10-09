import requests
import os
from typing import Dict, List, Optional
import threading

class HuggingFaceAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://huggingface.co/api"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.cancel_download = False

    def search_models(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        url = f"{self.base_url}/models"
        params = {"search": query}
        if filters:
            params.update(filters)
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def download_model(self, model_id: str, download_dir: str) -> str:
        self.cancel_download = False
        url = f"{self.base_url}/models/{model_id}/download"
        response = requests.get(url, headers=self.headers, stream=True)
        response.raise_for_status()
        
        filename = response.headers.get('Content-Disposition', '').split('filename=')[-1]
        if not filename:
            filename = f"{model_id.split('/')[-1]}.tar.gz"
        
        filepath = os.path.join(download_dir, filename)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if self.cancel_download:
                    break
                f.write(chunk)
        return filepath

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
