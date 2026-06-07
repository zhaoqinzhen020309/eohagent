# This file includes classe to get response from deployed local LLM
# Now supports both custom Flask servers and Ollama API
import json
from typing import Collection
import requests


class InterfaceLocalLLM:
    """Language model that predicts continuation of provided source code.
    Supports both custom Flask servers and Ollama API.
    """

    def __init__(self, url, model_name=None):
        self._url = url
        self._model_name = model_name  # For Ollama, this should be the model name

        # Detect if using Ollama based on URL pattern
        self._is_ollama = self._detect_ollama()

        if self._is_ollama:
            print(f"Detected Ollama API at: {self._url}")
            print(f"Using model: {self._model_name}")
        else:
            print(f"Using custom Flask server at: {self._url}")

    def _detect_ollama(self):
        """Detect if the URL is for Ollama API"""
        url_lower = self._url.lower()
        # Check for Ollama indicators
        return ('11434' in self._url or
                '/api/generate' in url_lower or
                '/api/chat' in url_lower or
                'ollama' in url_lower)

    def get_response(self, content: str) -> str:
        while True:
            try:
                response = self._do_request(content)
                return response
            except Exception as e:
                print(f"Request failed: {e}, retrying...")
                continue

    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()

        if self._is_ollama:
            return self._do_ollama_request(content)
        else:
            return self._do_flask_request(content)

    def _do_ollama_request(self, content: str) -> str:
        """Handle Ollama API request"""
        # Ensure URL has proper endpoint
        url = self._url
        if not url.endswith('/api/generate') and not url.endswith('/api/chat'):
            if '/api/' not in url:
                url = url.rstrip('/') + '/api/generate'

        # Use model name if provided, otherwise use default
        model_name = self._model_name or 'deepseek-r1:70b-llama-distill-fp16'

        # Construct Ollama API request
        # 不设置temperature, top_p, top_k等参数，让Ollama使用默认值
        # 这样可以保持与原Flask服务器的行为一致（原Flask也都是None）
        data = {
            'model': model_name,
            'prompt': content,
            'stream': False
        }

        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=json.dumps(data), headers=headers, timeout=300)

        if response.status_code == 200:
            response_json = response.json()
            # Ollama returns response in 'response' field
            return response_json.get('response', '')
        else:
            raise Exception(f"Ollama API returned status code {response.status_code}: {response.text}")

    def _do_flask_request(self, content: str) -> str:
        """Handle custom Flask server request (original format)"""
        data = {
            'prompt': content,
            'repeat_prompt': 1,
            'params': {
                'do_sample': True,
                'temperature': None,
                'top_k': None,
                'top_p': None,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(self._url, data=json.dumps(data), headers=headers, timeout=300)

        if response.status_code == 200:
            response_json = response.json()
            # Flask server returns response in 'content' field as list
            return response_json['content'][0]
        else:
            raise Exception(f"Flask server returned status code {response.status_code}: {response.text}")