from pretty_midi import PrettyMIDI
import numpy as np


class RemiItem:
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch


def notes_and_tempo_in_midi_ticks(midi_data: PrettyMIDI):
    remi_notes = []
    for i, instrument in enumerate(midi_data.instruments):
        for note in instrument.notes:
            start_tick = midi_data.time_to_tick(note.start)
            end_tick = midi_data.time_to_tick(note.end)

            rn = RemiItem(
                name=f"Instrument_{i}",
                start=start_tick,
                end=end_tick,
                velocity=note.velocity,
                pitch=note.pitch
            )
            remi_notes.append(rn)

    tempo_changes = []
    times, tempos = midi_data.get_tempo_changes()
    for t, bpm in zip(times, tempos):
        tick = midi_data.time_to_tick(t)
        tempo = RemiItem(
            name="Tempo",
            start=tick,
            end=None,
            velocity=None,
            pitch=bpm
        )
        tempo_changes.append(tempo)
    remi_notes.sort(key=lambda x: x.start)
    tempo_changes.sort(key=lambda x: x.start)
    return quantize_items(remi_notes), tempo_changes


def quantize_items(items, ticks=120):
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items


def get_items_from_midi_file(filepath):
    midi_data = PrettyMIDI(filepath)
    note_items, tempo_items = notes_and_tempo_in_midi_ticks(midi_data)
    for note in note_items:
        print(vars(note))
    for tempo in tempo_items:
        print(vars(tempo))
    return note_items, tempo_items


# test
if __name__ == '__main__':
    get_items_from_midi_file('./data/train/000.midi')
