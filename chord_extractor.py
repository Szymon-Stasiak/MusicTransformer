from music21 import converter
from remi_item import RemiItem
from collections import defaultdict
from utils import quantize_items_16th


def chords_from_midi(filepath):
    try:
        score = converter.parse(filepath)
    except Exception as e:
        print(f"Error parsing: {e}")
        return []

    try:
        s_chords = score.chordify()
    except Exception:
        return []

    if not s_chords.hasPartLikeStreams():
        try:
            s_chords.makeMeasures(inPlace=True)
        except:
            pass

    aggregated_items = []

    measures = s_chords.recurse().getElementsByClass('Measure')

    for m in measures:
        abs_start = m.offset
        abs_end = m.offset + m.duration.quarterLength

        chord_durations = defaultdict(float)

        for c in m.flat.getElementsByClass('Chord'):
            duration = c.quarterLength

            try:
                name = f"{c.root().name}:{c.quality}"
            except:
                name = "Unknown"

            chord_durations[name] += duration

        if chord_durations:
            best_chord = max(chord_durations, key=chord_durations.get)
        else:
            best_chord = "Rest"

        item = RemiItem(
            name="Chord",
            start=float(abs_start),
            end=float(abs_end),
            velocity=None,
            pitch=best_chord
        )
        aggregated_items.append(item)

    return fulfil_and_clear_items(quantize_items_16th(aggregated_items))


def fulfil_and_clear_items(items):
    for c in items:
        c.pitch = clean_chord_pitch(c.pitch)

    to_add = []
    for i in range(len(items) - 1):
        if (items[i].end > items[i + 1].start):
            items[i].end = items[i + 1].start
        elif (items[i].end < items[i + 1].start):
            item = RemiItem(
                name="Chord",
                start=items[i].end,
                end=items[i + 1].start,
                velocity=None,
                pitch="Rest"
            )
            to_add.append(item)

    if len(to_add) > 0:
        items.extend(to_add)
        items.sort(key=lambda x: x.start)

    items = [item for item in items if item.start != item.end]

    i = 0
    while i < len(items) - 1:
        if items[i].pitch ==items[i + 1].pitch :
            items[i].end = items[i + 1].end
            del items[i + 1]
        else:
            i += 1

    return items


def clean_chord_pitch(pitch_str):
    if pitch_str == 'Rest':
        return pitch_str

    if ':' in pitch_str:
        root, quality = pitch_str.split(':')
        if quality == 'other':
            return f"{root}:major"

    return pitch_str
def print_chords(chords):
    print(f"Generated {len(chords)} aggregated chords (by measure).")
    for c in chords:
       print(vars(c))


if __name__ == '__main__':
    chords = chords_from_midi('./data/train/000.midi')
    print_chords(chords)
