import requests
import logging
import os
from typing import Dict, List, Optional
from urllib.parse import urljoin

class HuggingFaceAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://huggingface.co/api"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def search_models(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        url = f"{self.base_url}/models"
        params = {"search": query}
        if filters:
            params.update(filters)
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error searching models: {str(e)}")
            raise

    def download_model(self, model_id: str, download_dir: str) -> str:
        url = f"{self.base_url}/models/{model_id}/download"
        response = requests.get(url, headers=self.headers, stream=True)
        response.raise_for_status()
        
        filename = response.headers.get('Content-Disposition', '').split('filename=')[-1]
        if not filename:
            filename = f"{model_id.split('/')[-1]}.tar.gz"
        
        filepath = os.path.join(download_dir, filename)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return filepath

    def run_inference(self, model_id: str, input_text: str) -> Dict:
        """Run inference on a model"""
        try:
            cmd = [
                "huggingface-cli", "run-inference",
                model_id,
                "--inputs", input_text
            ]
            
            result = subprocess.run(
                cmd,
                env=self._cli_env,
                capture_output=True,
                text=True,
                check=True
            )
            
            return {"output": result.stdout.strip()}
        except Exception as e:
            logging.error(f"Error running inference: {str(e)}")
            raise
