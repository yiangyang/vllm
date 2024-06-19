# Benchmarking vLLM

## Downloading the ShareGPT dataset

You can download the dataset by running:
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Using the xFasterTransformer dataset

For compare inference performance with intel CPU and GPU, use the save dataset from xFasterTransformer/benchmark/prompt.json

## Uasge for the benchmark_offline.py script

```
python3 offline_inference.py -h

options:
  -h, --help                        show this help message and exit
  --model_name MODEL_NAME           Model name
  --input_prompt INPUT_PROMPT       Input Prompt, default prompt.json
  --tensor_parallel_size TENSOR_PARALLEL_SIZE
                                    Tensor parallel size, default tp=1 (1 GPU)
  --batch_size BATCH_SIZE           Batch size, default 1
  --input_tokens INPUT_TOKENS       max input tokens, default 32
  --output_tokens OUTPUT_TOKENS     max output tokens
  --temperature TEMPERATURE         set temperature
  --top_p TOP_P                     set top_p
  --iteration ITERATION             Inference benchmark iterations

Example:
  python3 offline_inference.py --model_name llama2-7b-hf --input_tokens 128 --output_tokens 128 --iteration 50

Output:
  ==================== Benchmark Result ====================
  Benchmark inferences:                    50
  Batch size:                              1
  Total input tokens:                      6400
  Total generated tokens:                  6400
  Inference Total Time (ms):               126117.94
  First Token Latency (ms):                26.12
  Next Token Avg Latency (ms):             19.66
  Throughput without 1st token (tok/s):    50.88
  ==========================================================



```
  
