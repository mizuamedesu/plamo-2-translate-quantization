
https://huggingface.co/pfnet/plamo-2-translate

多分8bit量子化は11GB、4bit量子化は8GBあれば動くと思われる。
以下はBitsAndBytesでフルサイズ、8bit量子化、4bit量子化の結果

```
========================================
Device: cuda
Model: pfnet/plamo-2-translate
Test sentences: 10

==================================================
Benchmarking: No Quantization
==================================================
Loading model with quantization: None
tokenizer_config.json: 100%|███████████████| 1.43k/1.43k [00:00<00:00, 12.8MB/s]
tokenization_plamo.py: 100%|███████████████| 16.9k/16.9k [00:00<00:00, 40.4MB/s]
A new version of the following files was downloaded from https://huggingface.co/pfnet/plamo-2-translate:
- tokenization_plamo.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
tokenizer.jsonl: 100%|█████████████████████| 10.6M/10.6M [00:00<00:00, 54.3MB/s]
special_tokens_map.json: 100%|█████████████████| 587/587 [00:00<00:00, 4.86MB/s]
config.json: 100%|█████████████████████████| 1.18k/1.18k [00:00<00:00, 12.3MB/s]
modeling_plamo.py: 100%|███████████████████| 66.8k/66.8k [00:00<00:00, 12.7MB/s]
A new version of the following files was downloaded from https://huggingface.co/pfnet/plamo-2-translate:
- modeling_plamo.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
model.safetensors.index.json: 100%|████████| 37.2k/37.2k [00:00<00:00, 26.0MB/s]
model-00004-of-00004.safetensors: 100%|█████| 4.49G/4.49G [00:27<00:00, 163MB/s]
model-00001-of-00004.safetensors: 100%|█████| 4.77G/4.77G [00:29<00:00, 163MB/s]
model-00003-of-00004.safetensors: 100%|█████| 4.83G/4.83G [00:30<00:00, 161MB/s]
model-00002-of-00004.safetensors: 100%|█████| 4.96G/4.96G [00:30<00:00, 164MB/s]
Fetching 4 files: 100%|███████████████████████████| 4/4 [00:30<00:00,  7.60s/it]
Loading checkpoint shards: 100%|██████████████████| 4/4 [00:03<00:00,  1.08it/s]
generation_config.json: 100%|██████████████████| 132/132 [00:00<00:00, 1.42MB/s]
Model loading time: 44.36s
GPU memory usage: 18430.70MB
Running inference on 10 sentences...
[ 1] The weather is beautiful today -> 33.447s
[ 2] Machine learning is revolution -> 0.552s
[ 3] I love reading books in the li -> 0.489s
[ 4] The cat is sleeping on the sof -> 0.489s
[ 5] Artificial intelligence will c -> 0.549s
[ 6] She enjoys cooking traditional -> 0.489s
[ 7] The mountains look magnificent -> 0.496s
[ 8] Technology connects people aro -> 0.551s
[ 9] Music has the power to heal th -> 0.594s
[10] Education is the key to succes -> 0.561s

Results for No Quantization:
  Average inference time: 3.822s
  Total inference time: 38.220s
  Memory usage: 18430.70MB
  Model loading time: 44.36s

==================================================
Benchmarking: 8-bit Quantization
==================================================
Loading model with quantization: BitsAndBytesConfig {
  "_load_in_4bit": false,
  "_load_in_8bit": true,
  "bnb_4bit_compute_dtype": "float32",
  "bnb_4bit_quant_storage": "uint8",
  "bnb_4bit_quant_type": "fp4",
  "bnb_4bit_use_double_quant": false,
  "llm_int8_enable_fp32_cpu_offload": false,
  "llm_int8_has_fp16_weight": false,
  "llm_int8_skip_modules": null,
  "llm_int8_threshold": 6.0,
  "load_in_4bit": false,
  "load_in_8bit": true,
  "quant_method": "bitsandbytes"
}

Loading checkpoint shards: 100%|██████████████████| 4/4 [00:21<00:00,  5.26s/it]
Model loading time: 26.76s
GPU memory usage: 10145.92MB
Running inference on 10 sentences...
[ 1] The weather is beautiful today -> 0.690s
[ 2] Machine learning is revolution -> 1.146s
[ 3] I love reading books in the li -> 1.029s
[ 4] The cat is sleeping on the sof -> 1.028s
[ 5] Artificial intelligence will c -> 1.150s
[ 6] She enjoys cooking traditional -> 1.030s
[ 7] The mountains look magnificent -> 1.148s
[ 8] Technology connects people aro -> 1.268s
[ 9] Music has the power to heal th -> 1.146s
[10] Education is the key to succes -> 1.153s

Results for 8-bit Quantization:
  Average inference time: 1.079s
  Total inference time: 10.788s
  Memory usage: 10145.92MB
  Model loading time: 26.76s

==================================================
Benchmarking: 4-bit NF4 Quantization
==================================================
Loading model with quantization: BitsAndBytesConfig {
  "_load_in_4bit": true,
  "_load_in_8bit": false,
  "bnb_4bit_compute_dtype": "bfloat16",
  "bnb_4bit_quant_storage": "uint8",
  "bnb_4bit_quant_type": "nf4",
  "bnb_4bit_use_double_quant": true,
  "llm_int8_enable_fp32_cpu_offload": false,
  "llm_int8_has_fp16_weight": false,
  "llm_int8_skip_modules": null,
  "llm_int8_threshold": 6.0,
  "load_in_4bit": true,
  "load_in_8bit": false,
  "quant_method": "bitsandbytes"
}

Loading checkpoint shards: 100%|██████████████████| 4/4 [00:17<00:00,  4.42s/it]
Model loading time: 23.43s
GPU memory usage: 6120.78MB
Running inference on 10 sentences...
[ 1] The weather is beautiful today -> 0.399s
[ 2] Machine learning is revolution -> 0.668s
[ 3] I love reading books in the li -> 0.599s
[ 4] The cat is sleeping on the sof -> 0.598s
[ 5] Artificial intelligence will c -> 0.667s
[ 6] She enjoys cooking traditional -> 0.530s
[ 7] The mountains look magnificent -> 0.661s
[ 8] Technology connects people aro -> 0.660s
[ 9] Music has the power to heal th -> 0.658s
[10] Education is the key to succes -> 0.657s

Results for 4-bit NF4 Quantization:
  Average inference time: 0.610s
  Total inference time: 6.096s
  Memory usage: 6120.78MB
  Model loading time: 23.43s

============================================================
QUANTIZATION PERFORMANCE SUMMARY
============================================================
Memory Reduction (8-bit): 45.0%
Memory Reduction (4-bit): 66.8%
Inference Speedup (8-bit): 3.54x
Inference Speedup (4-bit): 6.27x

================================================================================
DETAILED BENCHMARK RESULTS
================================================================================
         Configuration Avg Inference Time (s) Memory Usage (MB) Model Loading Time (s)
       No Quantization                  3.822           18430.7                  44.36
    8-bit Quantization                  1.079           10145.9                  26.76
4-bit NF4 Quantization                  0.610            6120.8                  23.43

============================================================
SAMPLE TRANSLATIONS
============================================================

Input: The weather is beautiful today.
No Quantization     : 今日は天気が美しい。
8-bit Quantization  : 今日は天気が美しい。
4-bit NF4 Quantization: 今日は天気が美しい。

Input: Machine learning is revolutionizing technology.
No Quantization     : 機械学習はテクノロジーに革命をもたらしている。
8-bit Quantization  : 機械学習はテクノロジーに革命をもたらしている。
4-bit NF4 Quantization: 機械学習はテクノロジーに革命をもたらしている。

Input: I love reading books in the library.
No Quantization     : 私は図書館で本を読むのが好きだ。
8-bit Quantization  : 私は図書館で本を読むのが好きだ。
4-bit NF4 Quantization: 私は図書館で本を読むのが大好きです。
```
