<!-- <h1 align="center">PairedANNS</h1> -->
<h2 align="center">PairedANNS: Enhancing Approximate Nearest Neighbor Search via Index Pairing</h2>
Similarity search is a core database operation, and large-scale deployments rely on approximate nearest neighbor search (ANNS) methods from four main families: graph, quantization, hashing, and tree methods. Each family offers distinct recall--search--construction trade-offs. Prior evaluations provide limited insight into how the results of different ANNS methods relate, or how stable they remain across datasets and parameter choices. Motivated by these gaps, we conduct an empirical study of representative ANNS methods on 20 real-world datasets that reveals substantial behavioral differences between methods and complementarity between their results. Guided by these observations, we investigate the effectiveness of combining ANNS results from different methods, and systematically compare the performance of ANNS index pairings against standalone methods. We demonstrate that index pairings attain accuracy comparable to, and often exceeding, that of tuned standalone indices under identical search and construction latency budgets, limiting the need for per-dataset parameter tuning. Furthermore, we demonstrate that shared vector storage substantially reduces storage overhead and improves query efficiency, with limited impact on construction time.
<h3 id="algorithms">⚙️ Algorithms</h3>

| Method | Folder
|:------|:-------------|
| [FLATNAV](https://github.com/BlaiseMuhirwa/flatnav) | flatnav-main |
| [NSG](https://github.com/facebookresearch/faiss) | faiss-1.7.3 |
| [HNSW](https://github.com/facebookresearch/faiss) | faiss-1.7.3 |
| [VAQ](https://github.com/TheDatumOrg/VAQ) | VAQ |
| [OPQ](https://github.com/facebookresearch/faiss) | faiss-1.7.3 |
| [PQ](https://github.com/facebookresearch/faiss) | faiss-1.7.3 |
| [DB-LSH](https://github.com/Jacyhust/DB-LSH) | DB-LSH |
| [PM-LSH](https://github.com/Jacyhust/PM-LSH) | PM-LSH |
| [QALSH](https://github.com/HuangQiang/QALSH_Mem) | QALSH_Mem |
| [SCANN](https://github.com/google-research/google-research/tree/master/scann) | scann |
| [ANNOY](https://github.com/spotify/annoy) | annoy |
| [IVFPQ](https://github.com/facebookresearch/faiss) | faiss-1.7.3 |


Each folder contains a README file providing detailed instructions on building and executing the library.

### Datasets

The datasets used in this study are available in a [Google Drive](https://drive.google.com/drive/folders/1olm5AcZ1h7hBEuWEk3UwOWkxs6IXoewB?usp=sharing).
