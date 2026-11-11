#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>

#include <faiss/IndexNSG.h>
#include <faiss/index_io.h>
#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;

float* fvecs_read(char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    size_t n = sz / ((d + 1) * 4);
    *d_out = d;
    *n_out = n;

    float* x = new float[n * (d + 1)];
    fread(x, sizeof(float), n * (d + 1), f);
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

int* ivecs_read(char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

void replace_all(std::string &str, const std::string &from, const std::string &to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
}

int main(int argc, char* argv[]) {
    omp_set_num_threads(1);

    std::string dataset_name = argv[1];
    size_t d = std::stoi(argv[2]);
    size_t nt = std::stoi(argv[3]);
    size_t nq = std::stoi(argv[4]);
    int K = std::stoi(argv[5]);        // out-degree
    int search_L = std::stoi(argv[6]); // search list size
    int GK = std::stoi(argv[7]);       // graph build parameter

    std::string d_path="/data/[username]/similarity-search/dataset/[dataset]/base.fvecs";
    std::string q_path="/data/[username]/similarity-search/dataset/[dataset]/query.fvecs";
    std::string g_path="/data/[username]/similarity-search/dataset/[dataset]/groundtruth.ivecs";
    replace_all(d_path,"[dataset]",dataset_name);
    replace_all(q_path,"[dataset]",dataset_name);
    replace_all(g_path,"[dataset]",dataset_name);

    size_t read_d, read_nt;
    float* xt = fvecs_read((char*) d_path.c_str(), &read_d, &read_nt);
    assert(read_d == d);
    assert(read_nt == nt);

    // ------------------- Build NSG index -------------------
    faiss::IndexNSGFlat index(d, K);
    index.metric_type = faiss::METRIC_L2;
    index.nsg.search_L = search_L;

    // Step 1: train (build initial neighbor graph)
    auto start_training = std::chrono::high_resolution_clock::now();
    index.train(nt, xt);

    // Step 2: add vectors
    index.add(nt, xt);

    // Step 3: refine graph
    // NSG::build() requires (n, data, knn_graph, GK)
    // Pass nullptr for knn_graph to auto-generate from inner index
    index.build(nt, xt, nullptr, GK);
    auto end_training = std::chrono::high_resolution_clock::now();

    std::cout << "Index built. ntotal = " << index.ntotal << std::endl;
    delete[] xt;

    // ------------------- Load queries -------------------
    size_t read_dq, read_nq;
    float* xq = fvecs_read((char*) q_path.c_str(), &read_dq, &read_nq);
    assert(read_dq == d);
    assert(read_nq == nq);

    // ------------------- Load ground truth -------------------
    size_t k;
    faiss::Index::idx_t* gt;
    {
        size_t nq2;
        int* gt_int = ivecs_read((char*) g_path.c_str(), &k, &nq2);
        assert(nq2 == nq);
        gt = new faiss::Index::idx_t[k * nq];
        for (int i = 0; i < k * nq; i++)
            gt[i] = gt_int[i];
        delete[] gt_int;
    }

    // ------------------- Search -------------------
    faiss::Index::idx_t* I = new faiss::Index::idx_t[nq * k];
    float* D = new float[nq * k];

    auto start_searching = std::chrono::high_resolution_clock::now();
    index.search(nq, xq, k, D, I);
    auto end_searching = std::chrono::high_resolution_clock::now();

    // ------------------- Evaluate recall -------------------
    int matched = 0;
    for (int i = 0; i < nq; i++) {
        for (int ii = 0; ii < k; ii++) {
            int gt_nn = gt[i * k + ii];
            for (int j = 0; j < k; j++) {
                if (I[i * k + j] == gt_nn)
                    matched++;
            }
        }
    }

    printf("Recall@k = %.4f\n", matched / float(nq * k));

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_training - start_training);
    std::cout << "Construction time: " << duration.count() << " ms\n";

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_searching - start_searching);
    std::cout << "Search time: " << duration.count() << " ms\n";

    delete[] I;
    delete[] D;
    delete[] xq;
    delete[] gt;

    return 0;
}
