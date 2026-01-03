import os
import numpy as np
from constants import VOCAB_SIZE

def analyze_npy_cache(path):
    total_tokens = 0
    valid_tokens = 0
    invalid_tokens = 0

    if not os.path.exists(path):
        print(f"Error: cache directory {path} doesn't exist.")
        return None

    print("Analyzing tokens from .npy cache...")

    for file in os.listdir(path):
        if not file.endswith(".npy"):
            continue

        file_path = os.path.join(path, file)

        try:
            tokens = np.load(file_path)
            tokens = tokens.flatten()

            total_tokens += tokens.size
            valid_tokens += np.sum(tokens < VOCAB_SIZE)
            invalid_tokens += np.sum(tokens >= VOCAB_SIZE)

            print(f"Analyzed cache file: {file}")

        except Exception as e:
            print(f"Error loading cache file {file}: {e}")

    print("Analysis complete.")
    print(f"Total tokens: {total_tokens}")
    print(f"Valid tokens (< VOCAB_SIZE): {valid_tokens}")
    print(f"Invalid tokens (>= VOCAB_SIZE): {invalid_tokens}")

    return {
        "total_tokens": total_tokens,
        "valid_tokens": valid_tokens,
        "invalid_tokens": invalid_tokens,
    }

def main():
    stats = analyze_npy_cache("../training/data_cache")

    if stats is None:
        print("Cache analysis failed.")
        return

    print("\n===== CACHE SUMMARY =====")
    print(f"Total tokens     : {stats['total_tokens']}")
    print(f"Valid tokens     : {stats['valid_tokens']}")
    print(f"Invalid tokens   : {stats['invalid_tokens']}")

    if stats["total_tokens"] > 0:
        invalid_ratio = stats["invalid_tokens"] / stats["total_tokens"] * 100
        print(f"Invalid ratio    : {invalid_ratio:.2f}%")

if __name__ == "__main__":
    main()
