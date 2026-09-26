"""CLI 冒烟测试。作者：晨星"""
import json
import os
import tempfile

from cli import main


def test_cli_smoke():
    tmp = tempfile.mktemp(suffix=".json")
    try:
        rc = main([
            "--dataset", "synthetic",
            "--n-users", "80", "--n-items", "40", "--n-ratings", "700",
            "--backend", "numpy",
            "--factors", "16", "--iterations", "8",
            "--k", "5,10", "--seed", "42", "--out", tmp, "--quiet",
        ])
        assert rc == 0
        assert os.path.exists(tmp)
        with open(tmp, encoding="utf-8") as f:
            data = json.load(f)
        # 系统行可用、字段完整、benchmark.json 真实可解析
        sys_row = next(r for r in data["rows"] if r["backend"] in ("implicit", "numpy"))
        assert sys_row["available"] is True
        assert "recall@10" in sys_row["metrics"]
        assert data["n_train"] > 0 and data["n_test"] > 0
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
