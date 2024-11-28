import requests
import os
from typing import Dict, List, Optional
from tqdm import tqdm
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

    def download_model(self, model_id: str, download_dir: str, progress_callback=None) -> Dict[str, str]:
        try:
            os.makedirs(download_dir, exist_ok=True)
            local_dir = os.path.join(download_dir, model_id.split('/')[-1])
            
            files = self.list_model_files(model_id)
            total_files = len(files)
            downloaded_files = []

            for idx, file in enumerate(files, 1):
                try:
                    file_path = hf_hub_download(
                        repo_id=model_id,
                        filename=file,
                        token=self.api_key,
                        local_dir=local_dir,
                        resume_download=True
                    )
                    downloaded_files.append(file_path)
                    if progress_callback:
                        progress_callback(int((idx / total_files) * 100))
                except Exception as e:
                    print(f"Error downloading {file}: {str(e)}")

            return {"local_dir": local_dir, "files": downloaded_files}
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
