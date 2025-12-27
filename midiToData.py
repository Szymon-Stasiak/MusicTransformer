from pretty_midi import PrettyMIDI


class RemiNote:
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

class RemiTempo:
    def __init__(self, name, start, pitch):
        self.name = name
        self.start = start
        self.pitch = pitch


def notes_and_tempo_in_midi_ticks(midi_data: PrettyMIDI):
    remi_notes = []
    for i, instrument in enumerate(midi_data.instruments):
        for note in instrument.notes:
            start_tick = midi_data.time_to_tick(note.start)
            end_tick = midi_data.time_to_tick(note.end)

            rn = RemiNote(
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
        tempo = RemiTempo(
            name="Tempo",
            start=tick,
            pitch=bpm
        )
        tempo_changes.append(tempo)

    remi_notes.sort(key=lambda x: x.start)
    tempo_changes.sort(key=lambda x: x.start)


    return remi_notes, tempo_changes

if __name__ == '__main__':
    midi_data = PrettyMIDI('./data/evaluation/000.midi')
    note_items, tempo_items = notes_and_tempo_in_midi_ticks(midi_data)
    for note in note_items:
        print(vars(note))
    for tempo in tempo_items:
        print(vars(tempo))
