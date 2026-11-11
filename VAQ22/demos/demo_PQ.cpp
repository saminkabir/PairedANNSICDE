/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/time.h>

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>
#include <faiss/IndexPQ.h>
#include <iostream>
#include <vector>

using namespace std;
/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
 *
 * and unzip it to the sudirectory sift1M.
 **/

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/


 float* readFVecsFromExternal(char* filepath, size_t* N, size_t* maxRow) {
    FILE *infile = fopen(filepath, "rb");
    if (infile == NULL) {
        std::cout << "File not found" << std::endl;
        return nullptr;  // Return nullptr if the file is not found
    }
    int a = *maxRow;
    int b = *N;
    float* x = new float[a * b]();  // Initialize with zeros
    int rowCt = 0;
    int dimen;
    while (fread(&dimen, sizeof(int), 1, infile) == 1) {
        std::vector<float> v(dimen);
        if (fread(v.data(), sizeof(float), dimen, infile) != dimen) {
            std::cout << "Error when reading" << std::endl;
            break;
        }

        for (int i = 0; i < b; i++) {
            if (i < dimen) {
                x[rowCt * b + i] = v[i];
            } else {
                x[rowCt * b + i] = 0;  // Padding with 0 if the dimension is smaller than maxRow
            }
        }
        rowCt++;
    }

    if (fclose(infile)) {
        std::cout << "Could not close data file" << std::endl;
    }

    return x;
}


 float* fvecs_read(char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void replace_all(std::string &str, std::string &from, std::string &to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Move past the replaced substring
    }
}

int main(int argc, char* argv[]) {
    size_t d = 128;         // dimension
    int nbits = 8; 
    std::string dataset_name = argv[1];
    int nsub = std::stoi(argv[2]);
    int _d = std::stoi(argv[3]);
    size_t nt= std::stoi(argv[4]);
    size_t nq= std::stoi(argv[5]);
    std::string d_path="/data/[username]/similarity-search/dataset/[dataset]/base.fvecs";
    std::string q_path="/data/[username]/similarity-search/dataset/[dataset]/query.fvecs";
    std::string g_path="/data/[username]/similarity-search/dataset/[dataset]/groundtruth.ivecs";
    replace_all(d_path,"[dataset]",dataset_name);
    replace_all(q_path,"[dataset]",dataset_name);
    replace_all(g_path,"[dataset]",dataset_name);
    // Ensure that _d is divisible by nsub, or pad it
    if (_d % nsub != 0) {
        // Round up to the next multiple of nsub
        d = ((int)(_d / nsub) + 1) * nsub;
    }
    

    
    float* xt = readFVecsFromExternal(d_path.c_str(), &d, &nt);
    faiss::IndexPQ index(d, nsub, nbits);


    index.train(nt, xt);
    index.add(nt, xt);

    delete[] xt;
    
    float* xq;
    {
        size_t d2;
        xq = readFVecsFromExternal(q_path.c_str(), &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
    }

    size_t k;         // nb of results per query in the GT
    faiss::Index::idx_t* gt; // nq * k matrix of ground-truth nearest-neighbors

    {
        // load ground-truth and convert int to long
        size_t nq2;
        int* gt_int = ivecs_read(g_path.c_str(), &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new faiss::Index::idx_t[k * nq];
        for (int i = 0; i < k * nq; i++) {
            gt[i] = gt_int[i];
        }
        delete[] gt_int;
    }
    faiss::Index::idx_t* I = new faiss::Index::idx_t[nq * k];
    float* D = new float[nq * k];
    index.search(nq, xq, k, D, I);
    // evaluate result by hand.
    int n_1 = 0, n_10 = 0, n_100 = 0;
    for (int i = 0; i < nq; i++) {
        int gt_nn = gt[i * k];
        for (int j = 0; j < k; j++) {
            if (I[i * k + j] == gt_nn) {
                if (j < 1)
                    n_1++;
                if (j < 10)
                    n_10++;
                if (j < 100)
                    n_100++;
            }
        }
    }
    printf("R@1 = %.4f\n", n_1 / float(nq));
    printf("R@10 = %.4f\n", n_10 / float(nq));
    printf("R@100 = %.4f\n", n_100 / float(nq));

    delete[] I;
    delete[] D;
    delete[] xq;
    delete[] gt;
    return 0;

}
