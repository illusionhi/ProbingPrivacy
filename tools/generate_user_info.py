import torch
import json
import jsonlines
import os
import numpy as np
import random
import requests

api_key = ""

def generate_privacy_info(api_key, generated_user_info_list):
    while True:
        query = "Please generate the user privacy information in the JSON format without any other word."

        recent_user_info = generated_user_info_list[-10:]
        if recent_user_info:
            query += " The following user information has been generated below, please be as different as possible from the generated information:"
            for user_info in recent_user_info:
                query = query + "\n"  + json.dumps(user_info)
        
        print(query)
        print('====================================================================')

        url = ""
        headers = {
            'Authorization': f'Bearer {api_key}',  # Replace {api_key} with your actual API key
            'Content-Type': 'application/json'
        }
        request = {
            "inputs": {},
            "query": query,
            "response_mode": "streaming",
            "conversation_id": "",  # Set the conversation ID if necessary
            "user": "abc-123",
        }
        try:
            response = requests.post(url, headers=headers, json=request)
            status_code = response.status_code
            s = response.content.decode('utf-8').split("data: ")
            response = json.loads(s[-2])
            res_msg = response["data"]["outputs"]["answer"]
            res_object = json.loads(res_msg)
            
            return res_object
            
        except:
            pass

num_unser_info = 20

filename = "privacy3.jsonl"
generated_user_info_list = []
with jsonlines.open(filename, mode='w') as writer:
    for _ in range(num_unser_info):
        privacy_info = generate_privacy_info(api_key, generated_user_info_list)
        print(privacy_info)
        writer.write(privacy_info)
        generated_user_info_list.append(privacy_info)