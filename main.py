from processing.data_loader import create_token_sequence_from_directory

if __name__ == '__main__':
    groups = create_token_sequence_from_directory('./data/train', use_cache=True, make_cache=True, clean_cache=False)
    # events = create_list_of_events(groups)
    #
    # for e in events:
    #     print( "Event: ", e.name, " | Value: ", e.val)

    # print(f"\n{'=' * 20} After grouping {'=' * 20}\n")
    #
    # print(f"Total bars: {len(groups)}\n")
    #
    # for i, bar in enumerate(groups):
    #     bar_start = bar[0]
    #     bar_end = bar[-1]
    #
    #     items = bar[1:-1]
    #
    #     print(f"╔══ BAR {i + 1} (Sixteenths: {bar_start} -> {bar_end}) ══")
    #
    #     if not items:
    #         print("║  (Empty bar)")
    #
    #     for item in items:
    #         rel_pos = item.start - bar_start
    #
    #         if item.name == "Tempo":
    #             print(f"║  ⏱️  TEMPO : {item.pitch} BPM")
    #
    #         elif item.name == "Chord":
    #             print(f"║  🎹 CHORD : {item.pitch} (Pos: {rel_pos})")
    #
    #         elif item.name == "Note":
    #             duration = item.end - item.start
    #             print(
    #                 f"║  🎵 Note  : Pos: {rel_pos:<2} | Pitch: {item.pitch:<3} | Vel: {item.velocity:<3} | Dur: {duration}")
    #
    #         else:
    #             print(f"║  ?  {item.name}: {item}")
    #
    #     print("╚" + "═" * 45 + "\n")
