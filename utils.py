import numpy as np
from remi_item import RemiItem

RESOLUTION = 4


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


def group_items(notes, tempo, chords=None):
    if chords is None:
        chords = []

    items = notes + tempo + chords
    items.sort(key=lambda x: x.start)

    BAR_STEPS = 16

    if not items:
        return []

    max_time = items[-1].end
    downbeats = np.arange(0, max_time + BAR_STEPS, BAR_STEPS)

    groups = []

    for db_start, db_end in zip(downbeats[:-1], downbeats[1:]):
        insiders = [
            item for item in items
            if item.start >= db_start and item.start < db_end
        ]
        overall = [db_start] + insiders + [db_end]

        filtered = []
        tempo_seen = False
        for elem in overall:
            if isinstance(elem, RemiItem) and elem.name == "Tempo":
                if not tempo_seen:
                    filtered.append(elem)
                    tempo_seen = True
            else:
                filtered.append(elem)

        groups.append(filtered)

    return groups

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
