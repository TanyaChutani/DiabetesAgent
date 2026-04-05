from vllm import LLM, SamplingParams

#  multi-GPU auto
llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.2",

    tensor_parallel_size=2,   # use both GPUs 

    gpu_memory_utilization=0.7,   # lower
    enforce_eager=True,           #  disables CUDA graphs

    max_num_seqs=4                #  LIMIT BATCH
)
sampling_params = SamplingParams(
    temperature=0.2,
    top_p=0.9,
    max_tokens=256
)

def generate(prompts):
    outputs = llm.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]