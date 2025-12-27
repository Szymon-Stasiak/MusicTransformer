from music21 import converter, tempo, note, chord, stream, meter
from remi_item import RemiItem
from utils import quantize_items


def notes_and_tempo_in_ticks(filepath, ticks=120):
    score = converter.parse(filepath)

    remi_notes = []
    for i, part in enumerate(score.parts):
        for n in part.recurse().notes:
            if isinstance(n, chord.Chord):
                for p in n.pitches:
                    start_tick = int(n.offset * ticks)
                    end_tick = int((n.offset + n.quarterLength) * ticks)
                    rn = RemiItem(
                        name=f"Instrument_{i}",
                        start=start_tick,
                        end=end_tick,
                        velocity=getattr(n.volume, 'velocity', None),
                        pitch=p.midi
                    )
                    remi_notes.append(rn)
            elif isinstance(n, note.Note):
                start_tick = int(n.offset * ticks)
                end_tick = int((n.offset + n.quarterLength) * ticks)
                rn = RemiItem(
                    name=f"Instrument_{i}",
                    start=start_tick,
                    end=end_tick,
                    velocity=getattr(n.volume, 'velocity', None),
                    pitch=n.pitch.midi
                )
                remi_notes.append(rn)

    remi_notes.sort(key=lambda x: x.start)
    remi_notes = quantize_items(remi_notes, ticks)

    tempo_items = []
    for t in score.flat.getElementsByClass(tempo.MetronomeMark):
        start_tick = int(t.offset * ticks)
        tempo_item = RemiItem(
            name="Tempo",
            start=start_tick,
            end=None,
            velocity=None,
            pitch=t.number
        )
        tempo_items.append(tempo_item)
    tempo_items.sort(key=lambda x: x.start)

    return remi_notes, tempo_items


def get_items_from_midi_file(filepath):
    note_items, tempo_items = notes_and_tempo_in_ticks(filepath)
    for note in note_items:
        print(vars(note))
    for tempo in tempo_items:
        print(vars(tempo))
    return note_items, tempo_items


# test
if __name__ == '__main__':
    get_items_from_midi_file('./data/train/000.midi')
