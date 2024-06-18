import requests



class Translation():
    def __init__(self, endpoint):
        self.endpoint = endpoint
    def translate(self, messages, src_lang):
        json = {"messages": messages, "src_lang": src_lang}
        res =  requests.post(self.endpoint, json=json)
        return res.json()["res"]