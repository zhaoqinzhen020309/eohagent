import http.client
import json


class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5

    def get_response(self, prompt_content, temperature=None):
        """
        Get response from remote API

        Args:
            prompt_content: The prompt text
            temperature: Optional temperature parameter (0.0-2.0)
                        If None, uses the API's default temperature

        Returns:
            Response string from LLM
        """
        # Construct base payload
        payload_dict = {
            "model": self.model_LLM,
            "messages": [
                {"role": "user", "content": prompt_content}
            ],
        }

        # Add temperature if provided
        if temperature is not None:
            payload_dict["temperature"] = float(temperature)

        payload_explanation = json.dumps(payload_dict)

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
            "x-api2d-no-cache": 1,
        }

        response = None
        n_trial = 1
        while True:
            n_trial += 1
            if n_trial > self.n_trial:
                return response
            try:
                conn = http.client.HTTPSConnection(self.api_endpoint)
                conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
                res = conn.getresponse()
                data = res.read()
                json_data = json.loads(data)
                response = json_data["choices"][0]["message"]["content"]
                break
            except:
                if self.debug_mode:
                    print("Error in API. Restarting the process...")
                continue

        return response