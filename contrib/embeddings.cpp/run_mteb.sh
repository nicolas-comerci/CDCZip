mkdir build
cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
make
pwd
cd ../benchmarks
CUDA_VISIBLE_DEVICES="" python run_mteb.py 