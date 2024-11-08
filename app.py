import os
from typing import Generator

from flask import Flask, request, stream_with_context, Response
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

app = Flask(__name__)


@app.get("/")
def handle_home_get():
    return "OK", 200


def generate_chat_completion(content: str) -> str:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ]
    )
    return completion.choices[0].message.content


@app.post("/")
def handle_home_post():
    body = request.get_json()
    response = generate_chat_completion(content=body["content"])
    return response, 200


def generate_chat_completion_stream(content: str) -> Generator[str, None, None]:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ],
        stream=True
    )
    for chunk in completion:
        if chunk.choices[0].delta.content != None:
            yield chunk.choices[0].delta.content


@app.post("/stream")
def handle_stream_post():
    body = request.get_json()

    def generate():
        for chunk in generate_chat_completion_stream(content=body["content"]):
            yield chunk

    return Response(stream_with_context(generate()), content_type='application/json')


if __name__ == "__main__":
    app.run(
        debug=True
    )
