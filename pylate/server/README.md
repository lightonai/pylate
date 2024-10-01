# Serve the embeddings of a PyLate model
The ```server.py``` script allows to create a FastAPI server to serve the embeddings of a PyLate model.
To use it, simply run ```python server.py```
You can then send requests to the API like so:
```
curl -X POST http://localhost:8002/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Query 1", "Query 2"],
    "model": "lightonai/colbertv2.0",
    "is_query": false
  }'
```
If you want to encode queries, simply set ```ìs_query``` to ```True```.

Note that the server leverages [batched](https://github.com/mixedbread-ai/batched), so you can do batch processing by sending multiple separate calls and it will create batches dynamically to fill up the GPU.

For now, the server only support one loaded model, which you can define by using the ```--model``` argument when launching the server.

