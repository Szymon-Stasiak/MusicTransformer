from music21 import stream, note, tempo
from entities.remi_item import RemiItem
import random


def save_tokens_to_txt(tokens, filename):
    with open(f"{filename}.txt", "w") as f:
        f.write(" ".join(map(str, tokens)))
    print(f"Saved tokens to {filename}.txt")


def create_remi_from_tokens(tokens):
    if not tokens:
        print("No tokens provided for REMI creation.")
        return [], []
    items = []
    current_time = -16.0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # PAD SKIP
        # SOS SKIP
        # EOS SKIP
        # BAR
        if token == 3:
            current_time += 16.0
            i += 1
            continue
        # POSITION
        elif 4 <= token <= 19:
            position_index = token - 4
            current_time = (current_time // 16) * 16 + (position_index)
            i += 1
            continue
        # TEMPO
        elif 20 <= token <= 220:
            bpm = token - 20 + 30
            tempo_item = RemiItem(
                type="Tempo",
                start=current_time,
                end=None,
                descriptor=None,
                value=bpm
            )
            items.append(tempo_item)
            i += 1
            continue
        # CHORD SKIP
        # NOTE
        elif 341 <= token <= 372 and i + 2 < len(tokens):
            value = (token - 341) * 4
            note_on_token = tokens[i + 1]
            duration_token = tokens[i + 2]

            if 373 <= note_on_token <= 500 and 501 <= duration_token <= 564:
                pitch = note_on_token - 373
                duration = (duration_token - 501) + 1
                note_item = RemiItem(
                    type="Note",
                    start=current_time,
                    end=current_time + duration,
                    descriptor=value,
                    value=pitch
                )
                items.append(note_item)
                i += 3
                continue
        i += 1
    return items


def create_midi_from_remi_items(items, output_path='output.mid'):
    s = stream.Score()
    p = stream.Part()

    for item in items:
        offset = item.start / 4.0

        if item.type == "Tempo":
            tm = tempo.MetronomeMark(number=item.value)
            p.insert(offset, tm)

        elif item.type == "Note":
            n = note.Note(item.value)
            n.volume.velocity = item.value

            duration_steps = item.end - item.start
            n.quarterLength = duration_steps / 4.0

            p.insert(offset, n)

    s.append(p)
    s.write('midi', fp=output_path)
    print(f"Saved: {output_path}")


def create_midi_humanized(items, output_path='humanized.mid'):
    s = stream.Score()
    p = stream.Part()

    for item in items:
        human_timing = random.uniform(-0.025, 0.025)

        raw_offset = (item.start / 4.0) + human_timing

        offset = max(0.0, raw_offset)

        if item.type == "Note":
            n = note.Note(item.value)

            base_vel = getattr(item, 'velocity', 64)

            human_vel = random.randint(-10, 10)
            final_vel = max(1, min(127, base_vel + human_vel))
            n.volume.velocity = final_vel

            duration_steps = item.end - item.start
            n.quarterLength = duration_steps / 4.0

            p.insert(offset, n)

        elif item.type == "Tempo":
            p.insert(offset, tempo.MetronomeMark(number=item.value))

    s.append(p)
    s.write('midi', fp=output_path)
    print(f"Saved (humanized): {output_path}")


def create_midis_from_tokens(tokens, remi_output_path='remi_items.txt',
                             midi_output_path='output', ):
    humanized_midi_output_path = f"{midi_output_path}_humanized.mid"
    midi_output_path = f"{midi_output_path}.mid"
    items = create_remi_from_tokens(tokens)
    save_tokens_to_txt(tokens, remi_output_path)
    create_midi_from_remi_items(items, midi_output_path)
    create_midi_humanized(items, humanized_midi_output_path)
