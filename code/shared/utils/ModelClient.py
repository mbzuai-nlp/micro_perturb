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
import dill
import itertools
from pathlib import Path
from PIL import Image
import io
from openai import OpenAI, APIStatusError
from pydantic import BaseModel
from openai.types.responses import Response


# --- Add imports for other LLM clients below as needed ---
from google import genai
from google.genai import types
from mistralai import Mistral


def article_path_to_original(manuscript_id, article_path):
    file_name = os.path.basename(article_path).split('-')[0]+'.pdf'
    return f"../../datasets/bmj/{manuscript_id}/{file_name}"

def unpack_req_data(data):
    req_id, req_data = data
    system_prompt = req_data['system_prompt']
    message = req_data['message']
    file_data = None
    structured_output_format = None
    img_data = None
    reasoning_effort = None

    if 'file_data' in req_data.keys():
        file_data = req_data['file_data']

    if 'structured_output_format' in req_data.keys():
        structured_output_format = req_data['structured_output_format']

    if 'img_data' in req_data.keys():
        img_data = req_data['img_data']

    if 'reasoning_effort' in req_data.keys():
        reasoning_effort = req_data['reasoning_effort']

    return req_id, system_prompt, message, file_data, structured_output_format, img_data, reasoning_effort

def prepare_user_input(data):
    req_id, system_prompt, message, file_data, structured_output_format, img_data, reasoning_effort = unpack_req_data(data)
    user_content = []

    if file_data is not None:
        user_content.append({
            "type": "input_file",
            "filename": "manuscript.pdf",
            "file_data": f"data:application/pdf;base64,{file_data}",
        })

    if img_data is not None:
        for img in img_data:
            user_content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{img}",
                "detail": "original",
            })

    user_content.append({
        "type": "input_text",
        "text": message,
    })
    
    return user_content

class OpenAIClient():
    def __init__(self, model_name, api_key, base_url=None):
        self.model_name = model_name
        if base_url is not None:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.client_type = "openrouter"
        else:
            self.client = OpenAI(api_key=api_key)

    def query(self, system_prompt, message, file_data=None, output_format=None, img_data=None, reasoning_effort=None):
        user_content = []

        if file_data is not None:
            user_content.append({
                "type": "input_file",
                "filename": "manuscript.pdf",
                "file_data": f"data:application/pdf;base64,{file_data}",
            })

        if img_data is not None:
            for img in img_data:
                user_content.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{img}",
                    "detail": "original",
                })

        user_content.append({
            "type": "input_text",
            "text": message,
        })

        reasoning = {"effort": "none"}
        if reasoning_effort is not None:
            reasoning = {"effort": reasoning_effort}

        if output_format is not None:
            response = self.client.responses.parse(
                model=self.model_name,
                reasoning=reasoning,
                input=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_content
                    },
                ],
                text_format=output_format,
            )

            return response.output_parsed
        else:
            response = self.client.responses.create(
                model=self.model_name,
                reasoning=reasoning,
                input=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_content
                    },
                ],
            )

            return response.output_text
    
    # blocking batch query
    def batch_query(self, batch, model_output_directory):
        batch_id, batch_filepath, batch_metadata = self.add_batch(batch, model_output_directory)
        batch = self.wait_for_batch(batch_metadata)
        output_file_path = self.download_batch_results(batch_id, batch, model_output_directory)
        responses = self.process_batch_results(output_file_path)
        return responses
    
    def process_batch_results(self, output_file_path):
        responses = []
        with open(output_file_path, "r") as fp:
            for line in fp:
                api_response = json.loads(line)
                req_id = api_response["custom_id"]
                model_response = api_response["response"]

                body = model_response['body']
                #adjust these two values, otherwise pydantic validation fails
                body['store'] = 'in-memory'
                body['prompt_cache_retention'] = 'in-memory'
                response_obj = Response.model_validate(body)

                try:
                    response_text = response_obj.output_parsed
                except:
                    print('[ERROR] PROCESS BATCH RESULTS: unable to parse')
                    response_text = response_obj.output_text

                responses.append((req_id, response_text))

        return responses 
    
    def add_batch(self, batch, model_output_directory):
        batch_id, batch_file_path = self.create_batch_file(batch, model_output_directory)
        batch_input_file_id = self.upload_batch_file(batch_file_path)
        batch_metadata = self.start_batch(batch_input_file_id)
        return batch_id, batch_file_path, batch_metadata
    
    def create_batch_file(self, batch, model_output_directory):
        requests = []
        for data in batch:
            req_id, system_prompt, message, file_data, structured_output_format, img_data, reasoning_effort = unpack_req_data(data)
            
            user_content = prepare_user_input(data)

            req = {
                "custom_id": req_id,
                "method": "POST",
                "url": "/v1/responses",
            }

            req_body = {
                "model": self.model_name,
            }

            req_body["reasoning"] = {"effort": "none"}
            if reasoning_effort is not None:
                req_body["reasoning"] = {"effort": reasoning_effort}

            req_body["input"] = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]

            if structured_output_format is not None:
                schema = structured_output_format.model_json_schema()
                schema['additionalProperties'] = False

                req_body["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": structured_output_format.model_json_schema()['title'],
                        "schema": schema,
                        "strict": True,
                    }
                }

            req["body"] = req_body
            requests.append(req)

        os.makedirs(model_output_directory, exist_ok=True)
        batch_id = batch[0][0] # each req has unique id, set batch id to first req id
        batch_file_path = model_output_directory / f"batch_{batch_id}.json"
        with open(batch_file_path, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")

        return batch_id, batch_file_path

    def start_batch(self, batch_input_file_id):
        batch_metadata = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "description": f"Detection batch"
            }
        )
        return batch_metadata
    
    def upload_batch_file(self, batch_file_path):
        batch_input_file = self.client.files.create(
            file=open(batch_file_path, "rb"),
            purpose="batch"
        )

        return batch_input_file.id

    def is_batch_complete(self, batch_metadata):
        batch_id = batch_metadata.id
        batch = self.client.batches.retrieve(batch_id)
        if batch.status == "completed":
            return True
        
        if batch.status == "failed":
            return None
            #raise Exception("Batch failed...")
        
        return False

    def wait_for_batch(self, batch_metadata):
        batch_id = batch_metadata.id
        while not self.is_batch_complete(batch_metadata):
            time.sleep(300)
        
        batch = self.client.batches.retrieve(batch_id)
        return batch
    
    # Returns boolean array, None if batch failed
    def completed_batches(self, batch_metadata_list):
        completed = []
        for batch_metadata in batch_metadata_list:
            complete = self.is_batch_complete(batch_metadata)
            completed.append(complete)

        return completed
    
    def wait_for_any_batch(self, batches, download_directory):
        batch_metadata_list = [batch_data[2] for batch_data in batches]
        # continue if True or None
        #while all([not b for b in self.completed_batches(batch_metadata_list)]):
        while len([b for b in self.completed_batches(batch_metadata_list) if (b is None) or b]) == 0:
            time.sleep(180)

        completed_bool = self.completed_batches(batch_metadata_list)
        failed_batches = [batch for batch, completed in zip(batches, completed_bool) if completed is None]
        completed_data = [batch for batch, completed in zip(batches, completed_bool) if completed]
        completed_responses = []
        for batch_id, batch_filepath, batch_metadata in completed_data:
            batch_id = batch_metadata.id
            batch_res = self.client.batches.retrieve(batch_id)
            output_file_path = self.download_batch_results(batch_id, batch_res, download_directory)
            responses = self.process_batch_results(output_file_path)
            completed_responses.append(responses)

        remaining_batches = [batch for batch, completed in zip(batches, completed_bool) if (completed is not None) and (not completed)]
        return remaining_batches, failed_batches, completed_responses

    def download_batch_results(self, batch_id, batch, model_output_directory):
        batch_output_file_id = batch.output_file_id
        file_response = self.client.files.content(batch_output_file_id)
        output_file_path = model_output_directory / f"batch_{batch_id}_output.json"
        with open(output_file_path, "w") as f:
            f.write(file_response.text)
        return output_file_path

class OpenRouterClient():
    def __init__(self, model_name, api_key, base_url=None):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.client_type = "openrouter"

    def query(self, system_prompt, message, file_data=None, output_format=None, img_data=None, reasoning_effort=None):
        user_content = []

        if file_data is not None:
            user_content.append({
                "type": "input_file",
                "filename": "manuscript.pdf",
                "file_data": f"data:application/pdf;base64,{file_data}",
            })

        if img_data is not None:
            for img in img_data:
                user_content.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{img}",
                    "detail": "original",
                })

        user_content.append({
            "type": "input_text",
            "text": message,
        })

        reasoning = {"effort": "none"}
        if reasoning_effort is not None:
            reasoning = {"effort": reasoning_effort}

        if output_format is not None:
            response = self.client.responses.parse(
                model=self.model_name,
                reasoning=reasoning,
                timeout=600,
                input=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_content
                    },
                ],
                text_format=output_format,
                extra_body={
                    "provider": {
                        "order": ['alibaba', 'openai', 'anthropic', 'google-vertex/global'],
                    },
                },
            )

            return response.output_parsed
        else:
            retries = 3
            while retries > 0:
                try:
                    response = self.client.responses.create(
                        model=self.model_name,
                        reasoning=reasoning,
                        timeout=600,
                        input=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": user_content
                            },
                        ],
                        extra_body={
                            "provider": {
                                "order": ['alibaba', 'openai', 'anthropic', 'google-vertex/global'],
                            },
                        },
                    )

                    break
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError: {e}. Retrying...")
                    retries -= 1

                    if retries == 0:
                        raise e
                    
                    time.sleep(600)
                except APIStatusError as e:
                    if e.status_code == 413:
                        print('413 error: image size probably too large, skipping')
                        return ""

            return response.output_text
        
class VLLMClient(OpenAIClient):
    def __init__(self, model_name):
        with open('vllm_url.txt', 'r') as fp:
            base_url = fp.read()
            print(base_url)
        self.model_name = model_name
        api_key = ""
        #super().__init__(api_key=api_key, model_name=model_name, base_url=base_url)
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=3600.0, max_retries=0)

class GeminiClient():
    def __init__(self, model_name, api_key, model_directory):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        self.model_directory = model_directory

    def query(self, system_prompt, message, file_data=None, output_format=None, img_data=None, reasoning_effort=None):
        if file_data is not None or output_format is not None:
            raise Exception('File data and output format not implemented yet for Gemini client')

        uploaded_images = []
        if img_data is not None:
            os.makedirs(self.model_directory / "images", exist_ok=True)
            for i, img in enumerate(img_data):
                savepath = self.model_directory / "images" / f"image_{i}.jpeg"
                img_bytes = base64.b64decode(img)
                image = Image.open(io.BytesIO(img_bytes))
                image.save(savepath, format="JPEG", quality=100)
                my_file = self.client.files.upload(file=savepath)
                uploaded_images.append(my_file)


        thinking_config = None
        if reasoning_effort is not None:
            #for Gemini 3 or higher, set thinking level, otherwise set token budget
            thinking_config = None
            if self.model_name.startswith('gemini-3'):
                thinking_config = types.ThinkingConfig(thinking_level=reasoning_effort)
            else:
                if reasoning_effort != "high" and reasoning_effort != "none":
                    raise Exception("Gemini Client: Currently only support high or none for models earlier than Gemini 3")
                
                budget = 24576 if reasoning_effort == "high" else 0
                thinking_config = types.ThinkingConfig(thinking_budget=budget) # Use `0` to turn off thinking

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            thinking_config=thinking_config
        )

        contents = uploaded_images + [message]

        count = 0
        response = None
        while count < 3:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    config=config,
                    contents=contents
                )
            except:
                count += 1
                time.sleep(120)
                continue
            
            break

        if response is None:
            print("REQUEST FAILED")
            return None
        
        return response.text

class ModelClient():
    def __init__(self, model_type, model_name, model_prefix_name, model_output_directory):
        self.model_type = model_type
        self.model_name = model_name
        self.model_prefix_name = model_prefix_name
        self.model_directory = Path(model_output_directory)

        os.makedirs(self.model_directory, exist_ok=True)
        
        if model_type == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            self.client = OpenAIClient(model_name=self.model_name, api_key=api_key)
        elif model_type == "vllm":
            self.client = VLLMClient(model_name=self.model_name)
        elif model_type == "openrouter":
            OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
            self.client = OpenRouterClient(model_name=self.model_name, base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        elif model_type == "gemini":
            GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
            self.client = GeminiClient(model_name=self.model_name, api_key=GEMINI_API_KEY, model_directory=self.model_directory)
        # elif model_type == "mistral":
        #     MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
        #     self.client = Mistral(api_key=MISTRAL_API_KEY)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def query(self, system_prompt, message, file_data=None, output_format=None, img_data=None, reasoning_effort=None):
        return self.client.query(system_prompt, message, file_data=file_data, output_format=output_format, img_data=img_data, reasoning_effort=reasoning_effort)

    def batch_query(self, batch):
        return self.client.batch_query(batch, self.model_directory)
    # have this also handle batch requests so unified interface when calling

    # run the code for this function when there is an unpickling error with dill
    # def fix_cache():
        # model_prefix = "qwen_3_vl_30b_a3b_thinking"
        # def load_from_cache(filename):
        #     print(f"Loading cached responses from {filename}")
        #     with open(filename, 'rb') as f:
        #         while True:
        #             try:
        #                 # Load the next object in the stream
        #                 obj = dill.load(f)
        #                 yield obj
        #             except EOFError:
        #                 # End of file reached
        #                 break
        #             # uncomment this for UnpicklingError: but load all responses and save a new cache file, otherwise future responses won't be saved properly
        #             except:
        #                 pass

        # def save_to_cache(filename, data):
        #     print(f"Saving responses to cache {filename}")
        #     with open(filename, 'wb') as f:
        #         for obj in data:
        #             dill.dump(obj, f)

        # fn = f'{config.OUTPUT_FOLDER_PATH}responses_{TASK_ID}_{INPUT_TYPE}_{model_prefix}_cache.dill'
        # data = list(load_from_cache(fn))
        # save_to_cache(fn, data)

    def load_from_cache(self, filename):
        print(f"Loading cached responses from {filename}")
        with open(filename, 'rb') as f:
            while True:
                try:
                    # Load the next object in the stream
                    obj = dill.load(f)
                    yield obj
                except EOFError:
                    # End of file reached
                    break
                # uncomment this for UnpicklingError: but load all responses and save a new cache file, otherwise future responses won't be saved properly. fix_cache() above has the code to do this.
                # except:
                #     pass

    def _iterative_query(self, iter, processed_list):
        for data in iter:
            req_id, system_prompt, message, file_data, structured_output_format, img_data, reasoning_effort = unpack_req_data(data)
            #already processed from cache
            if req_id in processed_list:
                continue
            response = self.query(system_prompt, message, file_data=file_data, output_format=structured_output_format, img_data=img_data, reasoning_effort=reasoning_effort)
            yield req_id, response

    # def batch_query(self, batch, model_output_directory):
    #     batch_id, batch_filepath, batch_metadata = self.add_batch(batch, model_output_directory)
    #     batch = self.wait_for_batch(batch_metadata)
    #     output_file_path = self.download_batch_results(batch_id, batch, model_output_directory)
    #     responses = self.process_batch_results(output_file_path)
    #     return responses

    def retry_failed_batches(self, failed_batches):
        REDUCED_BATCH_SIZE = 3
        retried_batches = []
        
        for failed_batch_id, failed_batch_filepath, failed_batch_metadata in failed_batches:
            with open(self.model_directory / f"{failed_batch_id}_input.dill", "rb") as fp:
                failed_batch_data = dill.load(fp)
                #already reduced, still failed
                if len(failed_batch_data) == REDUCED_BATCH_SIZE:
                    print("Failed reduced")
                    with open(self.model_directory / "batch_failures.txt", "a") as fp:
                        for req_id, res in failed_batch_data:
                            fp.write(req_id + '\n')
                    continue

                reduced_batches = [failed_batch_data[i:i + REDUCED_BATCH_SIZE] for i in range(0, len(failed_batch_data), REDUCED_BATCH_SIZE)]
                for reduced_batch in reduced_batches:
                    reduced_batch_id, reduced_batch_filepath, reduced_batch_metadata = self.client.add_batch(reduced_batch, self.model_directory)
                    with open(self.model_directory / f"{reduced_batch_id}_input.dill", "wb") as fp:
                        dill.dump(reduced_batch, fp)
                    retried_batches.append((reduced_batch_id, reduced_batch_filepath, reduced_batch_metadata))

        return retried_batches

    def _batch_query(self, iter, processed_list):
        # Multiple of 15 for detection: 3x inputs, 5x categories per file
        # 250 for judge
        BATCH_SIZE = 250
        MAX_PARALLEL_BATCHES = 10

        batches = []
        batch = []
        for data in iter:
            req_id = data[0]
            #already processed from cache
            if req_id in processed_list:
                continue

            while len(batches) >= MAX_PARALLEL_BATCHES:
                remaining_batches, failed_batches, completed_responses = self.client.wait_for_any_batch(batches, self.model_directory)

                for responses in completed_responses:
                    yield from responses
                
                batches = remaining_batches

                if len(failed_batches) > 0:
                    print("failed batch")
                    batches = batches + self.retry_failed_batches(failed_batches)


            # consume data
            batch.append(data)

            #keep consuming until max parallel
            if len(batch) == BATCH_SIZE and len(batches) < MAX_PARALLEL_BATCHES:
                batch_id, batch_filepath, batch_metadata = self.client.add_batch(batch, self.model_directory)
                with open(self.model_directory / f"{batch_id}_input.dill", "wb") as fp:
                    dill.dump(batch, fp)
                batches.append((batch_id, batch_filepath, batch_metadata))
                batch = []

        # should always have capacity at this point, otherwise would still be in loop
        if len(batch) > 0:
            batch_id, batch_filepath, batch_metadata = self.client.add_batch(batch, self.model_directory)
            with open(self.model_directory / f"{batch_id}_input.dill", "wb") as fp:
                dill.dump(batch, fp)
            batches.append((batch_id, batch_filepath, batch_metadata))

        # wait for all pending batches to end
        while len(batches) > 0:
            remaining_batches, failed_batches, completed_responses = self.client.wait_for_any_batch(batches, self.model_directory)
            for responses in completed_responses:
                yield from responses
            batches = remaining_batches

            if len(failed_batches) > 0:
                batches = batches + self.retry_failed_batches(failed_batches)

    # wrapper with caching
    def iterative_query(self, iter, cache_file_path=None, batch=False, retry_cached_empty_response=False):
        # load from cache: don't adjust iterator, since cached responses may not be in same order as input
        processed_list = []
        if cache_file_path is not None:
            cache_file = cache_file_path + '.dill'

            if os.path.exists(cache_file):
                print('Loading from cache...')
                for res_data in self.load_from_cache(cache_file):
                    if res_data['response'] is None or (retry_cached_empty_response and len(res_data['response']) == 0):
                        print(f"[{res_data['meta_id']}: empty response in cache, skipping load")
                        continue
                    processed_list.append(res_data['meta_id'])
                    yield res_data['meta_id'], res_data['response']
            
                cache_fp = open(cache_file, 'ab')
            else:
                cache_fp = open(cache_file, 'wb')

        query_func = self._iterative_query

        if batch:
            query_func = self._batch_query

        for req_id, response in query_func(iter, processed_list):
            if cache_file_path is not None:
                res_dict = {
                    'meta_id': req_id,
                    'response': response,
                }
                dill.dump(res_dict, cache_fp)

            yield req_id, response

        if cache_file_path is not None:
            cache_fp.close()