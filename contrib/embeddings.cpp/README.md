# embeddings.cpp

[ggml](https://github.com/ggerganov/ggml) inference of BERT neural net architecture with pooling and normalization from embedding models including [SentenceTransformers (sbert.net)](https://sbert.net/), [BGE series](https://huggingface.co/BAAI/bge-base-en-v1.5) and others.
High quality sentence embeddings in pure C++ (with C API).

This repo is a fork of original [bert.cpp](https://github.com/skeskinen/bert.cpp).

In this fork, we have added support for:

+ Multilingual tokenizer inlcuding Asian languages and latin languages.
+ Support real batch inference.
+ Support current SOTA embedding model [BGE series](https://huggingface.co/BAAI/bge-base-en-v1.5).


## Description
The main goal of `bert.cpp` is to run the BERT model using 4-bit integer quantization on CPU

* Plain C/C++ implementation without dependencies
* Inherit support for various architectures from ggml (x86 with AVX2, ARM, etc.)
* Choose your model size from 32/16/4 bits per model weigth
* all-MiniLM-L6-v2/BGE with 4bit quantization is only 14MB. Inference RAM usage depends on the length of the input
* Sample cpp server over tcp socket and a python test client
* Benchmarks to validate correctness and speed of inference

## Limitations & TODO

+ Update to the latest ggml lib and gguf format.
+ Current memory management is not good enough, should be improved using lastest ggml api.

## Usage

### Checkout the ggml submodule
```sh
git submodule update --init --recursive
```

### Get models
```sh
pip install -r requirements.txt
cd models
python download-repo.py BAAI/bge-base-en-v1.5 # or any other model
sh run_conversions.sh bge-base-en-v1.5
```

### Test tokenizer

In this fork we support multilingual tokenizer, you can test different model's tokenzier by:

```sh
bash test_tokenizer.sh bge-base-en-v1.5
```

This script will tokenize the content in `models/test_prompts.txt` with both huggingface tokenizer and this tokenizer, and compare the results. You can add more content in the `models/test_prompts.txt` to test more cases. Note that orignal `bge-small-zh-v1.5` tokenizer (not this repo) is some problematic, refer to this [issue](https://github.com/xyzhang626/embeddings.cpp/issues/1) for more details.

### Build
To build the dynamic library for usage from e.g. Python:
```sh
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
make
cd ..
```

To build the native binaries, like the example server, with static libraries, run:
```sh
mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
make
cd ..
```
### Run the python dynamic library example
```sh
python3 examples/sample_dylib.py models/all-MiniLM-L6-v2/ggml-model-f16.bin

# bert_load_from_file: loading model from '../models/all-MiniLM-L6-v2/ggml-model-f16.bin' - please wait ...
# bert_load_from_file: n_vocab = 30522
# bert_load_from_file: n_max_tokens   = 512
# bert_load_from_file: n_embd  = 384
# bert_load_from_file: n_intermediate  = 1536
# bert_load_from_file: n_head  = 12
# bert_load_from_file: n_layer = 6
# bert_load_from_file: f16     = 1
# bert_load_from_file: ggml ctx size =  43.12 MB
# bert_load_from_file: ............ done
# bert_load_from_file: model size =    43.10 MB / num tensors = 101
# bert_load_from_file: mem_per_token 450 KB
# Loading texts from sample_client_texts.txt...
# Loaded 1738 lines.
# Starting with a test query "Should I get health insurance?"
# Closest texts:
# 1. Can I sign up for Medicare Part B if I am working and have health insurance through an employer?
#  (similarity score: 0.4790)
# 2. Will my Medicare premiums be higher because of my higher income?
#  (similarity score: 0.4633)
# 3. Should I sign up for Medicare Part B if I have Veterans' Benefits?
#  (similarity score: 0.4208)
# Enter a text to find similar texts (enter 'q' to quit): poaching
# Closest texts:
# 1. The exotic animal trade is enormous , and it continues to spiral out of control .
#  (similarity score: 0.2825)
# 2. " PeopleSoft management entrenchment tactics continue to destroy the value of the company for its shareholders , " said Deborah Lilienthal , an Oracle spokeswoman .
#  (similarity score: 0.2709)
# 3. " I 've stopped looters , run political parties out of abandoned buildings , caught people with large amounts of cash and weapons , " Williams said .
#  (similarity score: 0.2672)
```

### Start sample server
```sh
./build/bin/server -m models/all-MiniLM-L6-v2/ggml-model-q4_0.bin --port 8085

# bert_model_load: loading model from 'models/all-MiniLM-L6-v2/ggml-model-q4_0.bin' - please wait ...
# bert_model_load: n_vocab = 30522
# bert_model_load: n_ctx   = 512
# bert_model_load: n_embd  = 384
# bert_model_load: n_intermediate  = 1536
# bert_model_load: n_head  = 12
# bert_model_load: n_layer = 6
# bert_model_load: f16     = 2
# bert_model_load: ggml ctx size =  13.57 MB
# bert_model_load: ............ done
# bert_model_load: model size =    13.55 MB / num tensors = 101
# Server running on port 8085 with 4 threads
# Waiting for a client
```
### Run sample client
```sh
python3 examples/sample_client.py 8085
# Loading texts from sample_client_texts.txt...
# Loaded 1738 lines.
# Starting with a test query "Should I get health insurance?"
# Closest texts:
# 1. Will my Medicare premiums be higher because of my higher income?
#  (similarity score: 0.4844)
# 2. Can I sign up for Medicare Part B if I am working and have health insurance through an employer?
#  (similarity score: 0.4575)
# 3. Should I sign up for Medicare Part B if I have Veterans' Benefits?
#  (similarity score: 0.4052)
# Enter a text to find similar texts (enter 'q' to quit): expensive
# Closest texts:
# 1. It is priced at $ 5,995 for an unlimited number of users tapping into the single processor , or $ 195 per user with a minimum of five users .
#  (similarity score: 0.4597)
# 2. The new system costs between $ 1.1 million and $ 22 million , depending on configuration .
#  (similarity score: 0.4547)
# 3. Each hull will cost about $ 1.4 billion , with each fully outfitted submarine costing about $ 2.2 billion , Young said .
#  (similarity score: 0.4078)
```

### Converting models to ggml format
Converting models is similar to llama.cpp. Use models/convert-to-ggml.py to make hf models into either f32 or f16 ggml models. Then use ./build/bin/quantize to turn those into Q4_0, 4bit per weight models.

There is also models/run_conversions.sh which creates all 4 versions (f32, f16, Q4_0, Q4_1) at once.
```sh
cd models
# Clone a model from hf
python download-repo.py USERNAME/MODEL_NAME
# Run conversions to 4 ggml formats (f32, f16, Q4_0, Q4_1)
sh run_conversions.sh MODEL_NAME
```

## Benchmarks
Running MTEB (Massive Text Embedding Benchmark) with bert.cpp vs. [sbert](https://sbert.net/)(cpu mode) gives comparable results between the two, with quantization having minimal effect on accuracy and eval time being similar or better than sbert with batch_size=1 (bert.cpp doesn't support batching).

See [benchmarks](benchmarks) more info.

### BGE_base_en_v1.5

| Data Type | STSBenchmark | eval time | 
|-----------|-----------|------------|
| f32 | 0.8530 | 20.04 | 
| f16 | 0.8530 | 21.82 | 
| q4_0 | 0.8509 | 18.78 | 
| q4_0-batchless | 0.8509 | 35.97 |
| q4_1 | 0.8568 | 18.77 |
| sbert | 0.8464 | 7.52 | 
| sbert-batchless | 0.8464 | 64.58 | 

Note that the absolute value is not comparable to the original repo, as the test machine is different.
