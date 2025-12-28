from chord_extractor import chords_from_midi
from notes_and_tempo_extractor import get_items_from_midi_file
from utils import group_items

if __name__ == '__main__':
    note_items, tempo_items = get_items_from_midi_file('./data/train/000.midi')
    chords = chords_from_midi('./data/train/000.midi')
    groups = group_items(note_items, tempo_items, chords)

    print(f"\n{'=' * 20} After grouping {'=' * 20}\n")

    for i, bar in enumerate(groups):
        bar_start = bar[0]
        bar_end = bar[-1]

        items = bar[1:-1]

        print(f"╔══ BAR {i + 1} (Sixteenths: {bar_start} -> {bar_end}) ══")

        if not items:
            print("║  (Empty bar)")

        for item in items:
            rel_pos = item.start - bar_start

            if item.name == "Tempo":
                print(f"║  ⏱️  TEMPO : {item.pitch} BPM")

            elif item.name == "Chord":
                print(f"║  🎹 CHORD : {item.pitch} (Pos: {rel_pos})")

            elif item.name == "Note":
                duration = item.end - item.start
                print(
                    f"║  🎵 Note  : Pos: {rel_pos:<2} | Pitch: {item.pitch:<3} | Vel: {item.velocity:<3} | Dur: {duration}")

            else:
                print(f"║  ?  {item.name}: {item}")

        print("╚" + "═" * 45 + "\n")
