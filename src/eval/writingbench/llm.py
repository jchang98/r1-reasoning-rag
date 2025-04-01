import requests
import time
from typing import Callable
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()


class ClaudeAgent(object):
    def __init__(self,
                 system_prompt: str = None, server="r1"):
        self.system_prompt = system_prompt
        if server == "r1":
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            self.url = os.getenv("DEEPSEEK_BASE_URL")
            self.model = 'deepseek-r1-250120' # Model name

        if server == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.url = os.getenv("OPENAI_BASE_URL")
            self.model = 'gpt-4o'


        self.client = OpenAI(api_key=self.api_key, base_url=self.url)
    
    def call_claude(self,
             messages: list,
             top_k: int = 20,
             top_p: float = 0.8,
             temperature: float = 0.7,
             max_length: int = 8192):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": f"{self.model}",  
            "messages": messages,  
            "max_tokens": max_length,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature
        }

        attempt = 0
        max_attempts = 5
        wait_time = 1

        while attempt < max_attempts:
            try:
                # response = requests.post(self.url, headers=headers, json=data)
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    max_completion_tokens=max_length,
                    temperature=temperature,
                    # top_p=top_p,
                )
                

                return response.choices[0].message.content.strip()

            
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt+1}: Request failed due to network error: {e}, retrying...")

            time.sleep(wait_time)
            attempt += 1

        raise Exception("Max attempts exceeded. Failed to get a successful response.")
    
    def basic_success_check(self, response):
        if not response:
            print(response)
            return False
        else:
            return True
    
    def run(self,
            prompt: str,
            top_k: int = 20,
            top_p: float = 0.8,
            temperature: float = 0.7,
            max_length: int = 8192,
            max_try: int = 5,
            success_check_fn: Callable = None):
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user","content": prompt}
        ]
        success = False
        try_times = 0

        while try_times < max_try:
            response = self.call_claude(
                messages=messages,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                max_length=max_length,
            )

            if success_check_fn is None:
                success_check_fn = lambda x: True
            
            if success_check_fn(response):
                success = True
                break
            else:
                try_times += 1
        
        return response, success
