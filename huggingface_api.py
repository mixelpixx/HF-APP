import subprocess
import logging
import os
import json
from typing import Dict, List, Optional

class HuggingFaceAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._cli_env = os.environ.copy()
        self._cli_env["HUGGINGFACE_TOKEN"] = api_key
        self.login()

    def login(self):
        """Login using huggingface-cli"""
        try:
            # First try logging in with the token
            login_process = subprocess.run(
                ["huggingface-cli", "login", "--token", self.api_key],
                env=self._cli_env,
                capture_output=True,
                text=True
            )
            if login_process.returncode != 0:
                raise ValueError(f"Login failed: {login_process.stderr}")
            logging.info("Successfully logged in to Hugging Face")
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
                text=True,
                check=True
            )
            
            # Parse the output and convert to list of dicts
            models = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    models.append({"id": line.strip()})
            return models
        except Exception as e:
            logging.error(f"Error searching models: {str(e)}")
            raise

    def download_model(self, model_id: str, download_dir: str, progress_callback=None) -> Dict[str, str]:
        try:
            os.makedirs(download_dir, exist_ok=True)
            local_dir = os.path.join(download_dir, model_id.split('/')[-1])
             
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

            # Handle progress tracking if needed
            if progress_callback:
                for line in process.stdout:
                    progress_callback(line)

            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise ValueError(f"Download failed: {stderr}")
            return {"local_dir": local_dir, "stdout": stdout.strip()}
        except Exception as e:
            logging.error(f"Error downloading model: {str(e)}")
            raise

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
