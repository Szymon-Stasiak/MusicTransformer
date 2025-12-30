from constants import QUANTIZATION_RESOLUTION as RESOLUTION


def quantize_items_16th(items):
    for item in items:

        q_start = int(round(item.start * RESOLUTION))
        q_end = int(round(item.end * RESOLUTION))

        if q_end <= q_start:
            q_end = q_start + 1

        item.start = q_start
        item.end = q_end

    return items


def quantize_tempo_16th(items):
    for item in items:
        q_start = int(round(item.start * RESOLUTION))

        item.start = q_start

    return items


def fit_to_boundary(val, down, up):
    if val < down: return down
    if val > up: return up
    return val


def get_position(val):
    return val % 16


ROOT_MAP = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11
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


# todo move to better place
# def create_midi_from_remi(note_items, tempo_items, output_path):
#
#     s = stream.Stream()
#
#     for item in tempo_items:
#         tm = tempo.MetronomeMark(number=item.velocity)
#         s.insert(item.start, tm)
#
#     for item in note_items:
#         n = note.Note(item.pitch)
#
#         n.volume.velocity = int(item.velocity)
#
#         duration = item.end - item.start
#         if duration <= 0:
#             duration = 0.25
#
#         n.quarterLength = duration
#
#         s.insert(item.start, n)
#
#     s.write('midi', fp=output_path)
#     print(f"Saved MIDI: {output_path}")
def main():
    ids = set()

    for root_name in ROOT_MAP.keys():
        for quality_name in QUALITY_LIST:
            chord_id = get_chord_id(root_name, quality_name)
            chord_name = f"{root_name}{quality_name}"
            print(f"{chord_name:8s} -> {chord_id}")
            ids.add(chord_id)

    print("\nLiczba unikalnych ID:", len(ids))
    print("Zakres ID:", min(ids), "-", max(ids))


if __name__ == "__main__":
    main()
