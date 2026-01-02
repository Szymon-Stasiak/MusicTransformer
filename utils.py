def fit_to_boundary(val, down, up):
    if val < down: return down
    if val > up: return up
    return val


def get_position(val):
    return val % 16


ROOT_MAP = {
    "C": 0,
    "C#": 1, "Db": 1, "D-": 1,
    "D": 2,
    "D#": 3, "Eb": 3, "E-": 3,
    "E": 4, "Fb": 4,
    "F": 5, "E#": 5,
    "F#": 6, "Gb": 6, "G-": 6,
    "G": 7,
    "G#": 8, "Ab": 8, "A-": 8,
    "A": 9,
    "A#": 10, "Bb": 10, "B-": 10,
    "B": 11, "Cb": 11
}

QUALITY_LIST = [
    "maj", "min", "dim", "aug", "7",
    "maj7", "min7", "m7b5", "dim7", "other"
]

QUALITY_MAP = {q: i for i, q in enumerate(QUALITY_LIST)}
OTHER_QUALITY_ID = QUALITY_MAP["other"]

NUM_QUALITIES = len(QUALITY_LIST)


def get_chord_id(root_name: str, quality_name: str) -> int:
    root_id = get_root_id(root_name)
    quality_id = get_quality_id(quality_name)
    return root_id * NUM_QUALITIES + quality_id

def get_root_id(root_name: str) -> int:
    return ROOT_MAP[root_name]

def get_quality_id(quality_name: str) -> int:
    return QUALITY_MAP.get(quality_name, OTHER_QUALITY_ID)
