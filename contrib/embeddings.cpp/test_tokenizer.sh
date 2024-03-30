mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
make
cd ..
python examples/test_hf_tokenizer.py $1
build/bin/test_tokenizer -m models/$1/ggml-model-q4_0.bin