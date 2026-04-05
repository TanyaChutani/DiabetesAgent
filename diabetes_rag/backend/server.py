from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()

print("Loading model ONCE...")

llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.8,
    enforce_eager=True
)

class Req(BaseModel):
    prompt: str

@app.post("/generate")
def generate(req: Req):
    params = SamplingParams(max_tokens=256, temperature=0.7)

    out = llm.generate([req.prompt], params)
    return {"text": out[0].outputs[0].text}