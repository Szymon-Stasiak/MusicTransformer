from entities.remi_item import RemiItem
from collections import defaultdict
from processing.quantizator import quantize_items_16th
from music21 import converter


CHORD_QUALITY_MAPPING = {
    "major": "maj",
    "minor": "min",
    "dominant": "7",
    "major-seventh": "maj7",
    "minor-seventh": "min7",
    "diminished": "dim",
    "augmented": "aug",
    "half-diminished": "m7b5",
    "diminished-seventh": "dim7"
}


def get_chord_root(c):
    try:
        if c.root():
            return c.root().name
        return "unknown"
    except:
        return "unknown"


def get_chord_quality(c):
    try:
        q = c.quality
        return CHORD_QUALITY_MAPPING.get(q, "other")
    except:
        return "other"


def chords_from_midi(score):
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
        abs_start = float(m.offset)
        abs_end = abs_start + float(m.duration.quarterLength)

        chord_stats = defaultdict(float)

        for c in m.flatten().getElementsByClass('Chord'):
            duration = float(c.quarterLength)

            root = get_chord_root(c)
            quality = get_chord_quality(c)

            chord_stats[(root, quality)] += duration

        if chord_stats:
            best_root, best_quality = max(chord_stats, key=chord_stats.get)
        else:
            best_root = "unknown"
            best_quality = "other"

        if(best_root=="unknown" ):
            continue
        item = RemiItem(
            name="Chord",
            start=abs_start,
            end=abs_end,
            descriptor=best_quality,
            value=best_root
        )
        aggregated_items.append(item)

    return fulfil_and_clear_items(quantize_items_16th(aggregated_items))


def fulfil_and_clear_items(items):
    items.sort(key=lambda x: x.start)

    for i in range(len(items) - 1):
        if items[i].end > items[i + 1].start:
            items[i].end = items[i + 1].start

    items = [item for item in items if item.start < item.end]

    i = 0
    while i < len(items) - 1:
        cur = items[i]
        nxt = items[i + 1]

        if (cur.end == nxt.start and
                cur.value == nxt.value and
                cur.descriptor == nxt.descriptor):

            cur.end = nxt.end
            del items[i + 1]
        else:
            i += 1

    return items

def print_chords(chords):
    print(f"Generated {len(chords)} aggregated chords (by measure).")
    for c in chords:
        print(vars(c))


if __name__ == '__main__':
    score = converter.parse('../data/train/000.midi')
    chords = chords_from_midi(score)
    print_chords(chords)
