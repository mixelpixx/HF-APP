import subprocess
import os
from typing import Dict, List, Optional
import logging
import json
import shutil

class HuggingFaceAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._cli_env = os.environ.copy()
        self._cli_env["HUGGINGFACE_TOKEN"] = api_key
        self.setup_logging()
        self.validate_api_key()

    def setup_logging(self):
        """Configure logging for the API class"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        logging.info("Logging initialized for HuggingFace API")

    def validate_api_key(self):
        """Validate API key using huggingface-cli"""
        try:
            result = subprocess.run(
                ["huggingface-cli", "whoami"],
                env=self._cli_env,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise ValueError(f"API key validation failed: {result.stderr}")
        except Exception as e:
            logging.error(f"Invalid API key or connection error: {str(e)}")
            raise ValueError("Invalid API key or unable to connect to Hugging Face API")

    def search_models(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        cmd = ["huggingface-cli", "search", query, "--filter=model"]
        try:
            if filters:
                if filters.get("task"):
                    cmd.extend(["--filter", f"task={filters['task']}"])
                if filters.get("library"):
                    cmd.extend(["--filter", f"library={filters['library']}"])
            
            result = subprocess.run(
                cmd,
                env=self._cli_env,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise Exception(f"Search failed: {result.stderr}")
                
            # Parse the output into the expected format
            models = []
            for line in result.stdout.split('\n'):
                if line.strip():
                    models.append({"id": line.strip()})
            return models
        except Exception as e:
            logging.error(f"Error processing filters: {str(e)}")
            raise ValueError("Invalid filter parameters")

    def download_model(self, model_id: str, download_dir: str, progress_callback=None) -> Dict[str, str]:
        os.makedirs(download_dir, exist_ok=True)
        local_dir = os.path.join(download_dir, model_id.replace('/', '--'))
 
        logging.info(f"Starting download of model: {model_id}")

        try:
            cmd = [
                "huggingface-cli", "download",
                model_id,
                "--local-dir", local_dir
            ]
            
            process = subprocess.Popen(
                cmd,
                env=self._cli_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                logging.error(f"Error during download: {stderr}")
                raise Exception(f"Download failed: {stderr}")
            logging.info(f"Download completed successfully to: {local_dir}")
            if progress_callback:
                progress_callback(100)
            return {"local_dir": local_dir, "success": True}
        except Exception as e:
            logging.error(f"Error during download: {str(e)}")
            if progress_callback:
                progress_callback(0)
            raise Exception(f"Download failed: {str(e)}")

    def get_model_info(self, model_id: str) -> Dict:
        cmd = ["huggingface-cli", "info", model_id]
        result = subprocess.run(
            cmd,
            env=self._cli_env,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(f"Failed to get model info: {result.stderr}")
        return json.loads(result.stdout)

    def list_model_files(self, model_id: str) -> List[str]:
        cmd = ["huggingface-cli", "files", model_id]
        result = subprocess.run(
            cmd,
            env=self._cli_env,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise Exception(f"Failed to list model files: {result.stderr}")
        return result.stdout.splitlines()

    def get_model_tags(self, model_id: str) -> List[str]:
        model_info = self.get_model_info(model_id)
        return model_info.get('tags', [])

    def get_model_downloads(self, model_id: str) -> int:
        model_info = self.get_model_info(model_id)
        return model_info.get('downloads', 0)

    def run_inference(self, model_id: str, inputs: str) -> Dict:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        response = requests.post(url, headers=self._headers, json={"inputs": inputs})
        response.raise_for_status()
        return response.json()
