import os
import numpy as np
import hashlib
from music21 import converter

from processing.extractors.notes_and_tempo_extractor import extract_notes_and_tempos
from processing.extractors.chord_extractor import extract_chords
from processing.event_mapper import group_items, create_list_of_events
from processing.tokenizer import event_to_int
from constants import  CACHE_DIR


def get_cache_filename(file_path):
    hash_name = hashlib.md5(file_path.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{hash_name}.npy")


def create_token_sequence_from_directory(
        folder_path,
        use_cache=True,
        make_cache=True,
        clean_cache=False,
):
    all_tokens = []

    if not os.path.exists(folder_path):
        print(f"Error: directory {folder_path} doesn't exist.")
        return None

    if clean_cache:
        if os.path.exists(CACHE_DIR):
            print(f"Cleaning cache in {CACHE_DIR}...")
            for file in os.listdir(CACHE_DIR):
                file_path = os.path.join(CACHE_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("Cache cleared.")

    if use_cache or make_cache:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

    print("Processing of files is ongoing...")

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not file.endswith((".mid", ".midi")):
                continue

            full_path = os.path.join(root, file)
            cache_path = get_cache_filename(full_path)

            if use_cache and os.path.exists(cache_path):
                try:
                    file_tokens = np.load(cache_path)
                    all_tokens.extend(file_tokens)
                    print(f"Loaded from cache: {file}")
                    continue
                except:
                    pass

            try:
                score = converter.parse(full_path)
                print(f"Processing {file}...")
                notes, tempos = extract_notes_and_tempos(score)
                chords = extract_chords(score)
                groups = group_items(notes, tempos, chords)
                events = create_list_of_events(groups)
                tokens = [event_to_int(ev) for ev in events]
                all_tokens.extend(tokens)

                if make_cache:
                    np_tokens = np.array(tokens, dtype=np.uint16)
                    np.save(cache_path, np_tokens)

            except Exception as e:
                print(f"Error in file {file}: {e}")

    print(f"Success: {len(all_tokens)} tokens have been loaded")
    return all_tokens


def create_token_sequence_from_npy_cache(path=CACHE_DIR):
    all_tokens = []

    if not os.path.exists(path):
        print(f"Error: cache directory {path} doesn't exist.")
        return None

    print("Loading tokens from .npy cache...")

    for file in os.listdir(path):
        if not file.endswith(".npy"):
            continue

        file_path = os.path.join(path, file)

        try:
            tokens = np.load(file_path)
            tokens = tokens.flatten()
            all_tokens.extend(tokens)
            print(f"Loaded cache file: {file}")
        except Exception as e:
            print(f"Error loading cache file {file}: {e}")

    if len(all_tokens) == 0:
        print("Warning: no tokens were loaded from cache.")
        return None

    all_tokens = np.array(all_tokens, dtype=np.uint16)

    print(f"Success: {len(all_tokens)} tokens loaded from cache")
    return all_tokens
