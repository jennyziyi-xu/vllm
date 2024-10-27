from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="/vision/u/jennyxu6/systems/Llama-3.2-3B",
    dtype="float16",
    tensor_parallel_size=1,
    speculative_model="/vision/u/jennyxu6/systems/Llama-3.2-1B",
    num_speculative_tokens=5,
    max_model_len=26000
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")