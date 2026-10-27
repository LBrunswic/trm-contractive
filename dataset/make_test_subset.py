"""Create a fixed, seeded subset of a Sudoku test split (one example per puzzle, one puzzle per group).

Subsets of different sizes built with the same seed are nested (prefixes of one permutation),
so test1k ⊂ test10k ⊂ test25k. Only a `test/` split is written; point `data_paths_test` at it.
"""
import argparse
import hashlib
import json
import os
import shutil

import numpy as np


def make_subset(source: str, output: str, n: int, seed: int = 0):
    src = os.path.join(source, "test")
    with open(os.path.join(src, "dataset.json")) as f:
        meta = json.load(f)

    inputs = np.load(os.path.join(src, "all__inputs.npy"), mmap_mode="r")
    labels = np.load(os.path.join(src, "all__labels.npy"), mmap_mode="r")
    puzzle_identifiers = np.load(os.path.join(src, "all__puzzle_identifiers.npy"))
    puzzle_indices = np.load(os.path.join(src, "all__puzzle_indices.npy"))
    group_indices = np.load(os.path.join(src, "all__group_indices.npy"))

    total = inputs.shape[0]
    assert meta["sets"] == ["all"]
    assert np.array_equal(puzzle_indices, np.arange(total + 1)) and np.array_equal(group_indices, np.arange(total + 1)), \
        "expects one example per puzzle and one puzzle per group (Sudoku layout)"
    assert n <= total

    selected = np.sort(np.random.default_rng(seed).permutation(total)[:n])

    out = os.path.join(output, "test")
    os.makedirs(out, exist_ok=True)
    sub_inputs = np.ascontiguousarray(inputs[selected])
    np.save(os.path.join(out, "all__inputs.npy"), sub_inputs)
    np.save(os.path.join(out, "all__labels.npy"), np.ascontiguousarray(labels[selected]))
    np.save(os.path.join(out, "all__puzzle_identifiers.npy"), puzzle_identifiers[selected].astype(np.int32))
    np.save(os.path.join(out, "all__puzzle_indices.npy"), np.arange(n + 1, dtype=np.int32))
    np.save(os.path.join(out, "all__group_indices.npy"), np.arange(n + 1, dtype=np.int32))

    meta.update(total_groups=n, total_puzzles=n)
    with open(os.path.join(out, "dataset.json"), "w") as f:
        json.dump(meta, f)

    if os.path.exists(os.path.join(source, "identifiers.json")):
        shutil.copy(os.path.join(source, "identifiers.json"), output)
    np.save(os.path.join(output, "subset_indices.npy"), selected)
    with open(os.path.join(output, "subset.json"), "w") as f:
        json.dump(dict(source=os.path.abspath(source), n=n, seed=seed,
                       sha256_inputs=hashlib.sha256(sub_inputs.tobytes()).hexdigest()), f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    make_subset(args.source, args.output, args.n, args.seed)
