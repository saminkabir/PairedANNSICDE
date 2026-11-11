# Fast Adaptive Similarity Search through Variance‑Aware Quantization

# Requirements
- CMake 3.13 or newer
- g++ 7.5 or newer

# Build
```
# With testing
mkdir build && cd build
cmake .. -DBUILD_TESTING=ON
make

# With debug
cmake -B build . -D=DCMAKE_BUILD_TYPE=Debug -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF
make
# Without avx
cmake -B build . -D=DCMAKE_BUILD_TYPE=Debug -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DOPTIMIZATION_LEVEL=nonavx


# With optimization level option {O2: full, O3: aggressive (default)}
cmake .. -DOPTIMIZATION_LEVEL=full
make

# Without optimization
cmake .. -DOPTIMIZATION_LEVEL=generic
make
```

# Run
```
# Testing
make test

# Driver (temporary)
./main
```
