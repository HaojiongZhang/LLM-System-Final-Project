import numpy as np

from minitorch.paged_attention import BlockManager


def test_block_manager_writes_and_gathers_across_blocks():
    manager = BlockManager(
        num_layers=1,
        num_blocks=4,
        block_size=2,
        n_head=2,
        head_dim=3,
        backend=None,
    )
    manager.allocate_seq(seq_id=7)

    expected_k = []
    expected_v = []
    for pos in range(3):
        k_vec = np.full((2, 3), pos + 1, dtype=np.float32)
        v_vec = np.full((2, 3), 10 + pos, dtype=np.float32)
        expected_k.append(k_vec)
        expected_v.append(v_vec)
        manager.write_kv(0, 7, pos, k_vec, v_vec)

    gathered_k, gathered_v = manager.gather_kv(0, 7, 3)

    assert gathered_k.shape == (1, 2, 3, 3)
    assert gathered_v.shape == (1, 2, 3, 3)
    np.testing.assert_allclose(gathered_k[0], np.stack(expected_k, axis=1))
    np.testing.assert_allclose(gathered_v[0], np.stack(expected_v, axis=1))
    assert len(manager.block_tables[7]) == 2


def test_block_manager_releases_sequence_blocks():
    manager = BlockManager(
        num_layers=1,
        num_blocks=3,
        block_size=2,
        n_head=1,
        head_dim=2,
        backend=None,
    )
    manager.allocate_seq(seq_id=1)
    manager.write_kv(
        0,
        1,
        2,
        np.ones((1, 2), dtype=np.float32),
        np.ones((1, 2), dtype=np.float32),
    )

    assert len(manager.free_blocks) == 1

    manager.free_seq(seq_id=1)

    assert sorted(manager.free_blocks) == [0, 1, 2]
    assert 1 not in manager.block_tables
    assert 1 not in manager.seq_lengths
