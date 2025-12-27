from music21 import converter
from remi_item import RemiItem
from utils import quantize_items


def chords_from_midi(filepath, ticks=120):
    score = converter.parse(filepath)
    chord_items = []

    chords = score.chordify()
    for i, c in enumerate(chords.recurse().getElementsByClass('Chord')):
        start_tick = int(c.offset * ticks)
        end_tick = int((c.offset + c.quarterLength) * ticks)

        try:
            root = c.root().name
            quality = c.quality
            simple_name = f"{root} {quality}" if quality else root
        except:
            simple_name = "Unknown"

        chord_item = RemiItem(
            name=f"Chord_{i}",
            start=start_tick,
            end=end_tick,
            velocity=None,
            pitch=simple_name
        )
        chord_items.append(chord_item)

    return quantize_items(chord_items)


# test
if __name__ == '__main__':
    chords = chords_from_midi('./data/train/000.midi')
