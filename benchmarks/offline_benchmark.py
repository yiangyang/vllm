import os
from vllm import LLM, SamplingParams
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, default=None, help="Model name")
parser.add_argument("--input_prompt", type=str, default=None, help="Input Prompt, default prompt.json")
parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size, default tp=1 (1 GPU)")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size, default 1")
parser.add_argument("--input_tokens", type=str, default="32", help="max input tokens, default 32")
parser.add_argument("--output_tokens", type=int, default=32, help="max output tokens")
parser.add_argument("--temperature", type=float, default=0.8, help="set temperature")
parser.add_argument("--top_p", type=float, default=0.95, help="set top_p")
parser.add_argument("--iteration", type=int, default=1, help="Inference benchmark iterations")


args = parser.parse_args()

if args.input_prompt is not None:
    input_prompt = args.input_prompt
else:
    with open("prompt.json", "r") as json_file:
        prompt_pool = json.load(json_file)["llama"]
        if args.input_tokens in prompt_pool:
            input_prompt = prompt_pool[args.input_tokens]
        else:
            raise SystemExit("[ERROR] Plese use --input_tokens if you want custom input.")

input_prompt_list = []
for _ in range(args.batch_size):
    input_prompt_list.append(input_prompt)

# prompts = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"
# Create a sampling params object.
sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.output_tokens)

# Create an LLM.
llm = LLM(model=args.model_name, tensor_parallel_size=args.tensor_parallel_size)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
print("Warmup the model...")
llm.generate(input_prompt_list, sampling_params)
print("\nBegin benchmark...")
total_input_tokens = 0
total_output_tokens = 0
total_time_list = []
ttft_list = []
next_token_latency_list = []
throughput_without_1st_list = []
for i in range(args.iteration):
    batch_input_tokens = 0
    batch_output_tokens = 0
    batch_token_latency_list = []
    print("Inference benchmark iteration", i+1, ":")
    outputs = llm.generate(input_prompt_list, sampling_params)
    for output in outputs:
        prompt = output.prompt
        input_tokens = len(output.prompt_token_ids)
        output_tokens = len(output.outputs[0].token_ids)
        batch_input_tokens += input_tokens
        batch_output_tokens += output_tokens
        batch_first_token_time = output.metrics.first_token_time
        batch_finished_time = output.metrics.finished_time
        batch_next_token_time = batch_finished_time - batch_first_token_time
        next_token_latency = batch_next_token_time * 1000 / (output_tokens - 1)
        batch_token_latency_list.append(next_token_latency)
        generated_text = output.outputs[0].text
        print("Prompt: {}".format(prompt))
        print("Generated text: {}\n".format(generated_text))
    
    first_scheduled_time = outputs[len(outputs) - 1].metrics.first_scheduled_time
    first_token_time = outputs[len(outputs) - 1].metrics.first_token_time
    finished_time = outputs[len(outputs) - 1].metrics.finished_time
    ttft = first_token_time - first_scheduled_time
    total_time = finished_time - first_scheduled_time
    without_1st_time = finished_time - first_token_time
    throughput_without_1st = (batch_output_tokens - 1) / without_1st_time
    
    total_input_tokens += batch_input_tokens
    total_output_tokens += batch_output_tokens
    total_time_list.append(total_time)
    ttft_list.append(ttft)
    next_token_latency_list.append(sum(batch_token_latency_list) / len(batch_token_latency_list))
    throughput_without_1st_list.append(throughput_without_1st)

# print benchmark result
print("=" * 20, "Benchmark Result", "=" * 20)
print("{:<40} {:<10}".format("Benchmark inferences:", args.iteration))
print("{:<40} {:<10}".format("Batch size:", args.batch_size))
print("{:<40} {:<10}".format("Total input tokens:", total_input_tokens))
print("{:<40} {:<10}".format("Total generated tokens:", total_output_tokens))
print("{:<40} {:<10.2f}".format("Inference Total Time (ms):", sum(total_time_list) * 1000))
print("{:<40} {:<10.2f}".format("First Token Latency (ms):", 1000 * sum(ttft_list) / len(ttft_list)))
print("{:<40} {:<10.2f}".format("Next Token Avg Latency (ms):", sum(next_token_latency_list) / len(next_token_latency_list)))
print("{:<40} {:<10.2f}".format("Throughput without 1st token (tok/s):", sum(throughput_without_1st_list) / len(throughput_without_1st_list)))
print("=" * 58, "\n" * 2)
