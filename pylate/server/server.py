import argparse
from typing import List

import batched
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pylate import models

app = FastAPI()


class EmbeddingRequest(BaseModel):
    """PyDantic model for the requests sent to the server.

    Parameters
    ----------
    input
        The input(s) to encode.
    is_query
        A boolean indicating if the input is a query or a document.
    model
        The name model to use for encoding.
    """

    input: List[str] | str
    is_query: bool = True
    model: str = "lightonai/colbertv2.0"


class EmbeddingResponse(BaseModel):
    """PyDantic model for the server answer to a call.

    Parameters
    ----------
    data
        A list of dictionaries containing the embeddings ("embedding" key) and the type of the object ("object" key, is always embedding).
    model
        The name of the model used for encoding.
    usage
        An approximation of the number of tokens used to generate the embeddings (computed by splitting the input sequences on spaces).
    """

    data: List[dict]
    model: str
    usage: dict


def wrap_encode_function(model, **kwargs):
    def wrapped_encode(sentences):
        return model.encode(sentences, **kwargs)

    return wrapped_encode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FastAPI ColBERT serving server with specified host, port, and model."
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8002, help="Port to run the server on"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lightonai/colbertv2.0",
        help="Model to serve, can be an HF model or a path to a model",
    )
    return parser.parse_args()


args = parse_args()

# We need to load the model here so it is shared for every request
model = models.ColBERT(args.model)
# We cannot create the function on the fly as the batching require to use the same function (memory address)
model.encode_query = batched.aio.dynamically(wrap_encode_function(model, is_query=True))
model.encode_document = batched.aio.dynamically(
    wrap_encode_function(model, is_query=False)
)


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    if request.model != args.model:
        raise HTTPException(
            status_code=400,
            detail=f"Model not supported, the loaded model is {args.model}, but the request is for {request.model}",
        )
    try:
        if request.is_query:
            embeddings = await model.encode_query(
                request.input,
            )
        else:
            embeddings = await model.encode_document(
                request.input,
            )

        # Format response
        data = [
            {"object": "embedding", "embedding": embedding.tolist(), "index": i}
            for i, embedding in enumerate(embeddings)
        ]

        # Calculate token usage (approximate)
        total_tokens = sum(len(text.split()) for text in request.input)

        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
