import os
from dotenv import load_dotenv
load_dotenv()
from langchain_nvidia_ai_endpoints import ChatNVIDIA


import asyncio
import os
import typer
import tiktoken
from typing import Optional
# Assuming we're using OpenAI's API
from openai import OpenAI, AsyncOpenAI
import requests
import httpx


async def get_r1_ask(messages):
    url = "http://open-server.51ifind.com/standardgwapi/arsenal_hub/vtuber/ai_access/qianfan/v1/chat/completions"
    # userId和token可以login之后获取
    headers = {
        "X-Arsenal-Auth":"arsenal-tools",
        "x-ft-arsenal-auth": "L24FB1H14W54KQENSSPC4CSB2S0PPM5M",
        'Cookie': os.getenv("L20_COOKIE"),
        'userId': os.getenv("THS_DEEPSEEK_USER_ID"),
        'token': os.getenv("THS_DEEPSEEK_TOKEN") ,
        'Content-Type': 'application/json'
    }
    data = {
        "model": "deepseek-r1",
        "messages": messages,
        "stream": False
    }
    # response = requests.post(url,headers=headers,json=data,verify=False, stream=False)
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data, timeout=None)

    from dotmap import DotMap
    if response.status_code == 200:
        return DotMap(response.json())
    else:
        print("请求r1失败")
        return {}


def create_openai_client(api_key: str, base_url: Optional[str] = None) :
    return AsyncOpenAI(
        api_key=api_key, base_url=base_url or "https://api.openai.com/v1"
    )


def create_deepseek_client(
    api_key: str, base_url: Optional[str] = None
) :
    return AsyncOpenAI(
        api_key=api_key, base_url=base_url or "https://api.deepseek.com/v1"
    )



def get_ai_client(model) :
    # Decide which API key and endpoint to use
    if model.startswith("ep-") or model.startswith("deepseek"):
        service = "deepseek"
        model = "ep-20250208165153-wn9ft"
    else:
        service = "openai"


    if service.lower() == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        endpoint = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            print("[red]Missing OPENAI_API_KEY in environment[/red]")
            raise typer.Exit(1)
        client = create_openai_client(api_key=api_key, base_url=endpoint)

        return client
    elif service.lower() == "deepseek" or service.lower().startswith("ep-"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        endpoint = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        if not api_key:
            print("[red]Missing DEEPSEEK_API_KEY in environment[/red]")
            raise typer.Exit(1)
        client = create_deepseek_client(api_key=api_key, base_url=endpoint)

        return client
    else:
        print(
            "[red]Invalid service selected. Choose 'openai' or 'deepseek'.[/red]"
        )
        raise typer.Exit(1)


MIN_CHUNK_SIZE = 140


def get_token_count(model, text: str) -> int:
    """Returns the number of tokens in a given text."""

    if model.startswith("ep-") or model.startswith("deepseek"):
        service = "deepseek"
        model = "ep-20250208165153-wn9ft"
    else:
        service = "openai"

    if service.lower() == "openai":
        encoder = tiktoken.get_encoding(
            "cl100k_base"
        )  # Updated to use OpenAI's current encoding
        return len(encoder.encode(text))
    elif service.lower() == "deepseek":
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))




async def generate_completions(client, model, messages, format=None):
    if model.startswith("ep-") or model.startswith("deepseek"):
        service = "deepseek"
        model = "ep-20250208165153-wn9ft"
    else:
        service = "openai"

    
    if service == "ollama":
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat(
                model=model, messages=messages, stream=False, format=format
            ),
        )
    else:
        # Run OpenAI call in thread pool since it's synchronous
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model, messages=messages, response_format=format
            ),
        )
    return response

r1 = get_ai_client("ep-20250208165153-wn9ft")

# r1 = ChatNVIDIA(model="ep-20250208165153-wn9ft",
#                 api_key=os.getenv("DEEPSEEK_API_KEY"), 
#                 base_url=os.getenv("DeepSEEK_BASE_URL"),
#                 temperature=0.6,
#                 top_p=0.7,
#                 max_tokens=4096)


if __name__ == "__main__":
    message = [
        {
            "role": "user",
            "content": "马克思是谁"
        }
    ]
    res = asyncio.run(get_r1_ask(message))
    print(res)