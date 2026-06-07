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
        self._model_name = model_name

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
        return ('11434' in self._url or
                '/api/generate' in url_lower or
                '/api/chat' in url_lower or
                'ollama' in url_lower)

    def get_response(self, content: str, temperature=None) -> str:
        """
        Get response from LLM

        Args:
            content: The prompt text
            temperature: Optional temperature parameter (0.0-2.0)
                        If None, uses the default temperature of the backend

        Returns:
            Response string from LLM
        """
        while True:
            try:
                response = self._do_request(content, temperature)
                return response
            except Exception as e:
                print(f"Request failed: {e}, retrying...")
                continue

    def _do_request(self, content: str, temperature=None) -> str:
        content = content.strip('\n').strip()

        if self._is_ollama:
            return self._do_ollama_request(content, temperature)
        else:
            return self._do_flask_request(content, temperature)

    def _do_ollama_request(self, content: str, temperature=None) -> str:
        """Handle Ollama API request"""
        url = self._url
        if not url.endswith('/api/generate') and not url.endswith('/api/chat'):
            if '/api/' not in url:
                url = url.rstrip('/') + '/api/generate'

        model_name = self._model_name or 'deepseek-r1:70b-llama-distill-fp16'

        # Construct Ollama API request
        data = {
            'model': model_name,
            'prompt': content,
            'stream': False
        }

        # Add temperature if provided
        if temperature is not None:
            data['temperature'] = float(temperature)

        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=json.dumps(data), headers=headers, timeout=300)

        if response.status_code == 200:
            response_json = response.json()
            return response_json.get('response', '')
        else:
            raise Exception(f"Ollama API returned status code {response.status_code}: {response.text}")

    def _do_flask_request(self, content: str, temperature=None) -> str:
        """Handle custom Flask server request (original format)"""
        data = {
            'prompt': content,
            'repeat_prompt': 1,
            'params': {
                'do_sample': True,
                'temperature': temperature if temperature is not None else None,
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
            return response_json['content'][0]
        else:
            raise Exception(f"Flask server returned status code {response.status_code}: {response.text}")