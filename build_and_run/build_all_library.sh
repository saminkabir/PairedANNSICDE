cd VAQ22/
mkdir build
cmake -B build . -D=DCMAKE_BUILD_TYPE=Debug -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF
cd build
make
cd ../../
cd flatnav-main/
./bin/build.sh -e
cd faiss-1.7.3/
cmake -B build -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_GPU=OFF
make -C build demo_ivfpq_indexing
make -C build demo_pq_opq_imi
make -C build demo_hnsw
make -C build demo_nsg
pip install scann
cd QALSH_Mem/methods/
make -j
cd ../../
conan install . --output-folder=build --build=missing
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
make -j
make
cd DB-LSH/dbLSH/
make
