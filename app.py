from typing import List, Optional, Union
from starlette.concurrency import run_in_threadpool
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import uvicorn
from fastapi.middleware.gzip import GZipMiddleware
import pydantic
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
import os
import gzip
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

verbose = bool(os.environ.get('VERBOSE', ''))


class GZipRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        content_encoding = request.headers.get('Content-Encoding', '').lower()
        if (verbose):
            print("content_encoding", content_encoding)
        if 'gzip' in content_encoding:
            try:
                body = await request.body()
                content_length = int(
                    request.headers.get('Content-Length', '0'))
                if len(body) != content_length:
                    return JSONResponse(
                        content={"error": "Invalid Content-Length header"},
                        status_code=400,
                    )
                body = gzip.decompress(body)
                request._body = body
                if (verbose):
                    print("content_length", content_length)
                    print("gzip decompressed body:", body)
            except ValueError:
                return JSONResponse(
                    content={"error": "Invalid Content-Length header"},
                    status_code=400,
                )
            except Exception as e:
                print(e)
                return JSONResponse(
                    content={"error": "Failed to decompress gzip content"},
                    status_code=400,
                )

        response = await call_next(request)
        return response


router = APIRouter()

DEFAULT_MODEL_NAME = "./models/paraphrase-multilingual-MiniLM-L12-v2"
# DEFAULT_MODEL_NAME = "D:\\models\\deepseek\\all-MiniLM-L12-v2"
E5_EMBED_INSTRUCTION = "passage: "
E5_QUERY_INSTRUCTION = "query: "
BGE_EN_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def create_app(args):
    initialize_embeddings(args)
    app = FastAPI(
        title="Open Text Embeddings API",
        version="1.0.4",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(GZipRequestMiddleware)

    # handling gzip response only
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    app.include_router(router)

    return app


class CreateEmbeddingRequest(BaseModel):
    model: Optional[str] = Field(
        description="The model to use for generating embeddings.", default=None)
    input: Union[str, List[str]] = Field(description="The input to embed.")
    dimensions: Optional[int] = Field(
        description="The number of dimensions the resulting output embeddings should have.",
        default=None)
    user: Optional[str] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input": "The food was delicious and the waiter...",
                }
            ]
        }
    }


class Embedding(BaseModel):
    object: str
    embedding: List[float]
    index: int


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class CreateEmbeddingResponse(BaseModel):
    object: str
    data: List[Embedding]
    model: str
    usage: Usage


embeddings = None
tokenizer = None
model = None


def str_to_bool(s):
    map = {'true': True, 'false': False, '1': True, '0': False}
    if s.lower() not in map:
        raise ValueError("Cannot convert {} to a bool".format(s))
    return map[s.lower()]


def initialize_embeddings(args):
    global embeddings
    global tokenizer
    global model
    if "DEVICE" in os.environ:
        device = os.environ["DEVICE"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # model_name = os.environ.get("MODEL")
    model_name = args.model_path
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
    print("Loading model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)


def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # 通常取 [CLS] 标记的输出作为句子嵌入
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings


class CustomEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [encode_text(text).flatten() for text in texts]

    def embed_query(self, text):
        return encode_text(text).flatten()


def _create_embedding(input: Union[str, List[str]]):
    global embeddings, model, tokenizer
    model_name = os.environ.get("MODEL")
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
    model_name_short = model_name.split("/")[-1]

    embeddings = CustomEmbeddings()
    if isinstance(input, str):
        tokens = embeddings.embed_query(input)
        return CreateEmbeddingResponse(data=[Embedding(embedding=tokens.tolist(),
                                                       object="embedding", index=0)],
                                       model=model_name_short, object='list',
                                       usage=Usage(prompt_tokens=len(tokens), total_tokens=len(tokens)))
    else:
        data = [Embedding(embedding=embedding, object="embedding", index=i)
                for i, embedding in enumerate(embeddings.embed_documents(input))]
        total_tokens = 0
        for text in input:
            total_tokens += len(tokenizer.tokenize(text))
        return CreateEmbeddingResponse(data=data, model=model_name_short, object='list',
                                       usage=Usage(prompt_tokens=total_tokens, total_tokens=total_tokens))


@router.post(
    "/v1/embeddings",
    response_model=CreateEmbeddingResponse,
)
async def create_embedding(
        request: CreateEmbeddingRequest
):
    if pydantic.__version__ > '2.0.0':
        return await run_in_threadpool(
            _create_embedding, **request.model_dump(exclude={"user", "model", "model_config", "dimensions"})
        )
    else:
        return await run_in_threadpool(
            _create_embedding, **request.dict(exclude={"user", "model", "model_config", "dimensions"})
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run the application with custom model path and port.')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the application on')
    args = parser.parse_args()

    app = create_app(args)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port
    )
