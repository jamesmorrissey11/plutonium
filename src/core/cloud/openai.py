import openai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential


def stream_response(user_message, model_name):
    collected_chunks = []
    collected_messages = []
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": user_message}],
        temperature=0,
        stream=True,
    )
    for i, chunk in enumerate(response):
        collected_chunks.append(chunk)
        chunk_message = chunk["choices"][0]["delta"]
        collected_messages.append(chunk_message)


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
