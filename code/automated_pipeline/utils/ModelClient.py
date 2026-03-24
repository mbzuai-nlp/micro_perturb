import pandas as pd
import json
import ast
import numpy as np
import random
import os
import time
import math
import tiktoken
import base64
from openai import OpenAI
from pydantic import BaseModel

# --- Add imports for other LLM clients below as needed ---
from google import genai
from google.genai import types
from mistralai import Mistral

def article_path_to_original(manuscript_id, article_path):
    file_name = os.path.basename(article_path).split('-')[0]+'.pdf'
    return f"../../datasets/bmj/{manuscript_id}/{file_name}"

class ModelClient():
    def __init__(self, model_type, model_name, model_prefix_name, model_output_directory):
        self.model_type = model_type
        self.model_name = model_name
        self.model_prefix_name = model_prefix_name
        self.model_directory = model_output_directory

        os.makedirs(self.model_directory, exist_ok=True)
        
        if model_type == "openai":
            api_key = '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'
            self.client = OpenAI(api_key=api_key)
        elif model_type == "gemini":
            GEMINI_API_KEY = "AIzaSyAxrtFQsvJtMejfaBAHgUROLfOzVd67dwY"
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        elif model_type == "mistral":
            MISTRAL_API_KEY = "Ct8Nr8UVqsbQpYjq2yvgyoDxVrGyQnrm"
            self.client = Mistral(api_key=MISTRAL_API_KEY)
        elif model_type == "openrouter":
            OPENROUTER_API_KEY = "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"
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1",api_key=OPENROUTER_API_KEY)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
    def create_batch_file(self, df, system_prompt, message_template, batch_num, total_batches):
        model_name = self.model_name
        model_prefix_name = self.model_prefix_name

        per_batch_size = (len(df)//total_batches)+1
        for i in range(total_batches):
            if i == total_batches - 1:
                batch = df.iloc[per_batch_size*i:]
            else:
                batch = df.iloc[per_batch_size*i:per_batch_size*(i+1)]

            requests = []
            for j, row in batch.iterrows():
                cid = row['manuscript_id']


                article_path = article_path_to_original(row['manuscript_id'], row['article_file'])
                article_name = row['manuscript_id']+".pdf"
                with open(article_path, "rb") as f:
                    data = f.read()

                base64_string = base64.b64encode(data).decode("utf-8")
                
                req = {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": model_name,
                        "reasoning": {"effort": "high"},
                        "max_output_tokens": 25000, #max of ~3000 in our real human reviews, but need space for reasoning
                        "input": [
                            {
                                "role": "system",
                                "content":  system_prompt,
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_file",
                                        "filename": article_name+".pdf",
                                        "file_data": f"data:application/pdf;base64,{base64_string}",
                                    },
                                    {
                                        "type": "input_text",
                                        "text": message_template,
                                    },
                                ],
                            },
                        ]
                    }
                }

                requests.append(req)
            
            os.makedirs(self.model_directory, exist_ok=True)
            with open(f"{self.model_directory}/batch_{i}.json", "w") as f:
                for req in requests:
                    f.write(json.dumps(req) + "\n")
    
    def run_batch_file(self, batch_num):
        self.upload_batch_file(batch_num)

        self.batch_metadata = self.client.batches.create(
            input_file_id=self.batch_input_file_id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "description": f"Review generation batch {batch_num}"
            }
        )

        self.wait_for_batch()
        self.download_batch_results(batch_num)

    def upload_batch_file(self, batch_num):
        batch_input_file = self.client.files.create(
            file=open(f"{self.model_directory}/batch_{batch_num}.json", "rb"),
            purpose="batch"
        )

        self.batch_input_file_id = batch_input_file.id

    def wait_for_batch(self):
        batch = self.client.batches.retrieve(self.batch_metadata.id)
        while batch.status != "completed":
            print(f"Batch status: {batch.status}")
            time.sleep(300)
            batch = self.client.batches.retrieve(self.batch_metadata.id)

    def download_batch_results(self, batch_num):
        batch = self.client.batches.retrieve(self.batch_metadata.id)
        batch_output_file_id = batch.output_file_id
        file_response = self.client.files.content(batch_output_file_id)
        with open(f"{self.model_directory}/batch_{batch_num}_output.json", "w") as f:
            f.write(file_response.text)

    # def query(self, message, parameters={}):
    #     if self.model_type == "openai" or self.model_type == "openrouter":
    #         response = self.client.chat.completions.create(
    #             model=self.model_name,
    #             messages=message,
    #             temperature=1.0,
    #             max_completion_tokens=25000, #well above the max of ~3000 in our real human reviews
    #         )

    #         return response.choices[0].message.content

    def query(self, system_prompt, message):
        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": message,
                        },
                    ]
                }
            ],
        )

        return response.output_text
    
    def file_query(self, system_prompt, message, file_data):
        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": "manuscript.pdf",
                            "file_data": f"data:application/pdf;base64,{file_data}",
                        },
                        {
                            "type": "input_text",
                            "text": message,
                        },
                    ]
                }
            ],
        )

        return response.output_text
    
    def iterative_query(self, iter):
        for data in iter:
            if len(data) == 2:
                system_prompt, message = data
                yield self.query(system_prompt, message)
            elif len(data) == 3:
                system_prompt, message, file_data = data
                yield self.file_query(system_prompt, message, file_data)