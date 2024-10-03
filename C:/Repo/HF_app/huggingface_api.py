import requests
import os
import time
import logging
from typing import Dict, List, Optional
from requests.exceptions import RequestException, Timeout

class HuggingFaceAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://huggingface.co/api"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.max_retries = 3
        self.timeout = 30
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _make_request(self, method, url, **kwargs):
        retries = 0
        while retries < self.max_retries:
            try:
                response = requests.request(method, url, headers=self.headers, timeout=self.timeout, **kwargs)
                response.raise_for_status()
                return response
            except Timeout:
                self.logger.warning(f"Request timed out. Retrying ({retries + 1}/{self.max_retries})...")
            except RequestException as e:
                self.logger.error(f"Request failed: {str(e)}. Retrying ({retries + 1}/{self.max_retries})...")
            retries += 1
            time.sleep(2 ** retries)  # Exponential backoff
        raise Exception(f"Failed to make request after {self.max_retries} attempts.")

    def search_models(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        url = f"{self.base_url}/models"
        params = {"search": query}
        if filters:
            params.update(filters)
        response = self._make_request("GET", url, params=params)
        return response.json()