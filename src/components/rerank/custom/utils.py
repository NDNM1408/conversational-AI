import json

import requests


def get_response(response: requests.Response):
    data = json.loads(response.content)
    return data


def post_http_request(api_url: str, sampling_params: dict = {}) -> requests.Response:
    headers = {"Content-Type": "application/json"}
    response = requests.post(api_url, headers=headers, json=sampling_params)
    return response
