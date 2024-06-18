import requests

class Cache():
    def __init__(self, endpoint):
        self.endpoint = endpoint
    def get(self, prompt):
        json = {
            "prompt": prompt,
            "answer": ""
        }
        get_endpoint = f"{self.endpoint}/get"
        res =  requests.post(get_endpoint, json=json)
        return res.json()["answer"]
    def put(self, prompt, messages):
        json = {
            "prompt": prompt,
            "answer": messages
        }
        put_endpoint = f"{self.endpoint}/put"
        res =  requests.post(put_endpoint, json=json)

