from music21 import converter, note, chord, tempo
from entities.remi_item import RemiItem
from processing.quantizator import quantize_items_16th, quantize_tempo_16th


def parse_midi_to_remi(score):
    flat_score = score.flatten()

    notes_list = []
    tempo_list = []

    for element in flat_score:
        if isinstance(element, note.Note):
            start = element.offset
            end = element.offset + element.duration.quarterLength
            vel = element.volume.velocity if element.volume.velocity else 64

            item = RemiItem("Note", start, end, vel, element.pitch.midi)
            notes_list.append(item)

        elif isinstance(element, chord.Chord):
            start = element.offset
            end = element.offset + element.duration.quarterLength
            vel = element.volume.velocity if element.volume.velocity else 64

            for p in element.pitches:
                item = RemiItem("Note", start, end, vel, p.midi)
                notes_list.append(item)

        elif isinstance(element, tempo.MetronomeMark):
            bpm = element.number
            item = RemiItem("Tempo", element.offset, None, None, int(bpm))
            tempo_list.append(item)

    return quantize_items_16th(notes_list), quantize_tempo_16th(tempo_list)


def get_items_from_midi_file(filepath):
    note_items, tempo_items = parse_midi_to_remi(filepath)
    return note_items, tempo_items


# test
if __name__ == '__main__':
    score = converter.parse('../data/train/000.midi')
    note_items, tempo_items = get_items_from_midi_file(score)

    for n in note_items:
        print(vars(n))

    for t in tempo_items:
        print(vars(t))
