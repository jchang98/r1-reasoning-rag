import os
import re
import json
import requests as rq
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()





class OpenAIClient(AsyncOpenAI):
    def __init__(self, model_name="gpt-4o-mini", verbose=False, record_to=None) -> None:
        super().__init__()
        self.model_name = model_name
        self.verbose = verbose
        self.record_to = record_to
        
    async def ask(self, msg, temperature=0):
        if self.verbose:
            for i in range(len(msg)):
                print(f"-----------------< {msg[i]['role']} >-----------------")
                print(msg[i]['content'])
        if self.model_name in ['o1', "o3-mini-2025-01-31"]:
            response = await self.chat.completions.create(
                model=self.model_name,
                reasoning_effort="medium",
                messages=msg,
            )
        elif self.model_name in ['deepseek-reasoner']:
            response = await self.chat.completions.create(
                model=self.model_name,
                messages=msg,
            )
        else:
            response = await self.chat.completions.create(
                model=self.model_name,
                messages=msg,
                temperature=temperature,
            )
        tokens = response.usage.total_tokens
        response = response.choices[0].message.content
        
        if self.verbose:
            print(f"-----------------< {self.model_name} >-----------------")
            print(response)
            print(f"tokens: {tokens}")
            
        if self.record_to is not None:
            try:
                with open(self.record_to, "r") as f:
                    records = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                records = []
            new_record = {
                "instruction": msg[0]['content'],
                "input": "",
                "output": response
            }
            records.append(new_record)
            with open(self.record_to, "w") as f:
                json.dump(records, f, indent=4, ensure_ascii=False)

        return response


class LocalClient():
    def __init__(self, model_type="llama-based", verbose=False):
        super().__init__()
        self.model_type = model_type
        self.verbose = verbose
    
    def ask(self, msg, temperature=0):
        if self.verbose:
            print(f"-----------------< {msg[0]['role']} >-----------------")
            print(msg[0]['content'])
        if self.model_type == "glm-based":
            payload = {
                    "model": 'llmt', 
                    "messages": msg,
                    "stop_token_ids": [151329, 151336, 151338]
            }
        elif self.model_type in ["llama-based", "qwen-based", "qwq-based"]:
            payload = {
                    "model": 'ifind-aime', 
                    "messages": msg,
                    "use_beam_search":False, 
                    "temperature": 0.0, 
                    "stream": False, 
                    "stop": ["<|im_end|>","<|end_of_text|>",'<|eot_id|>','<|endoftext|>'],
                    "ignore_eos": False
            }
        else:
            payload = {
                    "model": 'llmt', 
                    "messages": msg,
            }
        headers = {"User-Agent": "Test Client"}
        response = rq.post('http://10.244.113.149:8803/v1/chat/completions', headers=headers, json=payload, stream=False)
        response = response.json()['choices'][0]['message']["content"]

        if self.model_type in ['qwq-based']:
            response = response.split("</think>")[-1]
        if self.verbose:
            print(f"-----------------< {self.model_type} >-----------------")
            print(response)
        return response


def extract(content:str, pattern="json"):
    content = content.replace("\n\n", "\n")
    if pattern is not None:
        regex_pattern = r'```' + pattern + r'\s+(.*?)```'
        blocks = re.findall(regex_pattern, content, re.DOTALL)
        if not blocks:
            blocks = [content]
        block = blocks[0].strip()
        if pattern == "json":
            try:
                json_variable = json.loads(block)
                return json_variable
            except json.JSONDecodeError as e1:
                try: 
                    json_variable = eval(block)
                    return json_variable
                except Exception as e2:
                    raise ValueError("Cannot parse your toolcall as JSON") from e2
        else:
            return block
    else:
        return content


if __name__ == "__main__":
    import asyncio
    client = OpenAIClient(verbose=True)
    msg = [
        {"role":"user", "content":"你好"}
    ]
    asyncio.run(client.ask(msg))