"""FlatNav graph built directly on faiss storage (no vector copy).

Run with:  pytest -q python-bindings/unit_tests/test_faiss_shared_storage.py
Requires faiss (faiss-cpu or a local faiss build with Python bindings).
"""
import gc
import os
import tempfile

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")
import flatnav  # noqa: E402

N, NQ, D, M, K = 20000, 200, 64, 32, 10
EF_C, EF_S = 100, 256  # uniform random 64-d data is hard; ef 256 -> recall@10 ~0.95


def faiss_xb_view(flat):
    """Zero-copy numpy view of a faiss.IndexFlat's vectors."""
    return faiss.rev_swig_ptr(flat.get_xb(), flat.ntotal * flat.d).reshape(flat.ntotal, flat.d)


def recall(I, gt):
    return np.mean([len(set(a) & set(b)) / K for a, b in zip(I, gt)])


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    xb = rng.random((N, D), dtype=np.float32)
    xq = rng.random((NQ, D), dtype=np.float32)
    return xb, xq


@pytest.fixture(scope="module")
def shared(data):
    xb, xq = data
    storage = faiss.IndexFlatL2(D)
    storage.add(xb)
    index = flatnav.index.create(
        distance_type="l2", dim=D, dataset_size=N, max_edges_per_node=M, external_storage=storage
    )
    index.set_num_threads(min(4, os.cpu_count() or 1))
    index.add_from_storage(ef_construction=EF_C)
    _, gt = storage.search(xq, K)  # exact kNN from the *same* storage
    return storage, index, gt


def test_zero_copy_and_recall(data, shared):
    storage, index, gt = shared
    assert index.uses_external_storage
    # FlatNav reads exactly faiss' buffer.
    assert index.storage_data_ptr == faiss_xb_view(storage).ctypes.data
    # Graph-only allocation: (M links + label) * 4 bytes per node.
    assert index.graph_memory_bytes == N * (4 * M + 4)
    _, I = index.search(data[1], K, EF_S)
    assert recall(I, gt) > 0.9


def test_matches_owned_flatnav(data, shared):
    xb, xq = data
    storage, index, gt = shared
    owned = flatnav.index.create(distance_type="l2", dim=D, dataset_size=N, max_edges_per_node=M)
    owned.set_num_threads(min(4, os.cpu_count() or 1))
    owned.add(xb, ef_construction=EF_C)
    assert owned.graph_memory_bytes == N * (4 * D + 4 * M + 4)
    _, I_own = owned.search(xq, K, EF_S)
    _, I_sh = index.search(xq, K, EF_S)
    assert abs(recall(I_own, gt) - recall(I_sh, gt)) < 0.02


def test_reads_faiss_memory_live(data, shared):
    xb, _ = data
    storage, index, _ = shared
    view = faiss_xb_view(storage)
    q = xb[7:8].copy()
    D0, I0 = index.search(q, 1, EF_S)
    assert I0[0, 0] == 7 and D0[0, 0] == 0
    view[7] += 1.0  # mutate faiss' storage in place
    try:
        D1, I1 = index.search(q, K, EF_S)
        hit = np.where(I1[0] == 7)[0]
        assert len(hit) == 0 or np.isclose(D1[0, hit[0]], D, rtol=1e-4)
    finally:
        view[7] -= 1.0


def test_share_storage_of_faiss_hnsw(data):
    xb, xq = data
    hnsw = faiss.IndexHNSWFlat(D, 16)
    hnsw.add(xb)
    index = flatnav.index.create("l2", D, N, M, external_storage=hnsw)  # uses hnsw.storage
    index.add_from_storage(ef_construction=EF_C)
    flat = faiss.downcast_index(hnsw.storage)
    assert index.storage_data_ptr == faiss_xb_view(flat).ctypes.data
    _, gt = flat.search(xq, K)
    _, I = index.search(xq, K, EF_S)
    assert recall(I, gt) > 0.9


def test_index_keeps_storage_alive(data):
    xb, xq = data

    def build():
        storage = faiss.IndexFlatL2(D)
        storage.add(xb[:2000])
        idx = flatnav.index.create("l2", D, 2000, 16, external_storage=storage)
        idx.add_from_storage(ef_construction=50)
        return idx  # `storage` goes out of scope here

    idx = build()
    gc.collect()
    _, I = idx.search(xb[:5], 1, 50)
    assert list(I[:, 0]) == [0, 1, 2, 3, 4]


def test_save_load_graph_only(data, shared):
    _, xq = data
    storage, index, _ = shared
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "graph.index")
        index.save(path)
        # vectors are NOT in the file: graph only
        assert os.path.getsize(path) < N * (4 * M + 4) + 4096
        loaded = flatnav.index.IndexL2Float.load_index(path)
    assert loaded.uses_external_storage
    with pytest.raises(RuntimeError):
        loaded.search(xq, K, EF_S)
    loaded.attach_storage(storage)
    _, I_a = index.search(xq, K, EF_S)
    _, I_b = loaded.search(xq, K, EF_S)
    np.testing.assert_array_equal(I_a, I_b)


def test_incremental_add_after_faiss_grows(data):
    xb, xq = data
    storage = faiss.IndexFlatL2(D)
    storage.add(xb[:5000])
    idx = flatnav.index.create("l2", D, N, M, external_storage=storage)
    idx.add_from_storage(ef_construction=EF_C)
    storage.add(xb[5000:])  # may reallocate faiss' buffer
    idx.add_from_storage(ef_construction=EF_C)  # re-attaches, then inserts rows 5000..N-1
    assert idx.storage_data_ptr == faiss_xb_view(storage).ctypes.data
    _, gt = storage.search(xq, K)
    _, I = idx.search(xq, K, EF_S)
    assert recall(I, gt) > 0.9


def test_guards(data):
    xb, _ = data
    storage = faiss.IndexFlatL2(D)
    storage.add(xb[:1000])
    idx = flatnav.index.create("l2", D, 1000, 16, external_storage=storage)
    with pytest.raises(RuntimeError):
        idx.add(xb[:10], ef_construction=50)  # vectors must go into faiss storage
    ip = faiss.IndexFlatIP(D)
    ip.add(xb[:1000])
    with pytest.raises(ValueError):
        flatnav.index.create("l2", D, 1000, 16, external_storage=ip)  # metric mismatch
    with pytest.raises(ValueError):
        flatnav.index.create("l2", D, 1000, 16, external_storage=np.asfortranarray(xb[:1000]))
    with pytest.raises(ValueError):
        flatnav.index.create("l2", D, 1000, 16, external_storage=xb[:1000].astype(np.float64))
    # Plain numpy buffers work too (zero-copy).
    arr = np.ascontiguousarray(xb[:1000])
    np_idx = flatnav.index.create("l2", D, 1000, 16, external_storage=arr)
    np_idx.add_from_storage(ef_construction=50)
    assert np_idx.storage_data_ptr == arr.ctypes.data


def test_inner_product_storage(data):
    xb, xq = data
    xbn = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    xqn = xq / np.linalg.norm(xq, axis=1, keepdims=True)
    storage = faiss.IndexFlatIP(D)
    storage.add(xbn)
    idx = flatnav.index.create("angular", D, N, M, external_storage=storage)
    idx.set_num_threads(min(4, os.cpu_count() or 1))
    idx.add_from_storage(ef_construction=EF_C)
    _, gt = storage.search(xqn, K)
    _, I = idx.search(xqn, K, EF_S)
    assert recall(I, gt) > 0.9


def test_simd_level_reported():
    # "AVX2" for the default FLATNAV_SIMD=avx2 build; never AVX-512 unless
    # the extension was built with FLATNAV_SIMD=native.
    assert flatnav.simd_level in {"AVX2", "AVX", "SSE", "scalar", "AVX-512"}
    if os.environ.get("FLATNAV_SIMD", "avx2") == "avx2":
        assert flatnav.simd_level != "AVX-512"
