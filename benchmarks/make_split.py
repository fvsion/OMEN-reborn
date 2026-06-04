#!/usr/bin/env python3
"""Split a password corpus into disjoint train/test sets by content hash.

Each password is bucketed by ``md5(password) % 10``: bucket 0 is the held-out
test set, buckets 1-9 are training. Because the bucket is a deterministic
function of the password, the same password can never appear on both sides — the
test set is guaranteed disjoint from training (no leakage), which is what makes
the crack-rate benchmark a fair generalisation test.

    python make_split.py rockyou.txt train.txt test.txt
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

TEST_BUCKET = 0
NUM_BUCKETS = 10


def split(source: Path, train_out: Path, test_out: Path) -> tuple[int, int]:
    n_train = n_test = 0
    with (
        source.open("r", encoding="utf-8", errors="replace") as src,
        train_out.open("w", encoding="utf-8") as train_f,
        test_out.open("w", encoding="utf-8") as test_f,
    ):
        for line in src:
            pw = line.rstrip("\n")
            if not pw:
                continue
            digest = hashlib.md5(pw.encode("utf-8", "replace")).digest()[0]
            if digest % NUM_BUCKETS == TEST_BUCKET:
                test_f.write(pw + "\n")
                n_test += 1
            else:
                train_f.write(pw + "\n")
                n_train += 1
    return n_train, n_test


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: make_split.py CORPUS TRAIN_OUT TEST_OUT", file=sys.stderr)
        return 2
    n_train, n_test = split(Path(argv[0]), Path(argv[1]), Path(argv[2]))
    print(f"train={n_train:,}  test={n_test:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
