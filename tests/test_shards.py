"""Shard resume: a shard with DONE is skipped."""
import pandas as pd

from advice_audit.local import shards


def test_done_shard_skipped():
    df = pd.DataFrame({"x": range(10)})
    shards.write_shards(df, "m", "extract", 4)
    assert shards.n_shards("m", "extract") == 3
    calls = []

    def fn(d, out):
        calls.append(len(d))
        return {"n": len(d)}

    assert shards.run_shard("m", "extract", 1, fn) is True
    assert shards.is_done("m", "extract", 1)
    assert shards.run_shard("m", "extract", 1, fn) is False  # skipped
    assert calls == [4]
    assert len(shards.shard_dirs("m", "extract")) == 1
