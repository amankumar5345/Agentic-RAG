"""Guards for unchanged A0-A7 results and A8 isolation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from litho_eval.runner import a8_cache_dir

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_A0_A7 = {
    "A0.json": "3459a7b77a051604c5bc733f2fda9640ccf417f8f114b42861f1ebcccdb1c637",
    "A1.json": "8d7cbb026fd19cb1100aa78296594b11b3f70e70905b98b677be4c4f06b13123",
    "A2.json": "cf5ec5a90f61ab264c13123d311c92774aebecd07c5751f649b234be99acd4e3",
    "A3.json": "1c3ac1f5c26e62e28fc44d2eaea4a9a02a7982b5d161d27226afadceab21fd64",
    "A4.json": "c4f22fb7f8285b35901cc73efaba72336d20f812e7fff25d5b8f4172e5c2c349",
    "A5.json": "5b36f7278f636a72825f74d60b62b5062921d94940e257b6ca6ab505c06c3ba2",
    "A6.json": "b42e7c16e96d0115b91634975b777d18aa0c04cfda3b943d77b4952ec68bbbbd",
    "A7.json": "cc75f88606e8bb25db2261a896fe600da7d929489317dd37ef29a61b1bfa3f3d",
}


def test_a0_a7_results_are_byte_identical():
    for name, expected in EXPECTED_A0_A7.items():
        actual = hashlib.sha256((REPO_ROOT / "results" / name).read_bytes()).hexdigest()
        assert actual == expected, f"{name} changed; A0-A7 must stay byte-identical"


def test_a8_has_its_own_cache_namespace(tmp_path):
    assert a8_cache_dir(tmp_path) == tmp_path / "A8"
    assert a8_cache_dir(tmp_path) != tmp_path
