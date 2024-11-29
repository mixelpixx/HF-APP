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
