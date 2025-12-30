from constants import SOS, EOS, BAR, POSITION, TEMPO, CHORD, VELOCITY, NOTE_ON, DURATION, PAD

VOCAB_OFFSETS = {
        PAD: 0,                # Empty space (padding) - optional, but recommended
        SOS: 1,                # Start of Sequence
        EOS: 2,                # End of Sequence
        BAR: 3,                # 1 token
        POSITION: 4,           # 16 tokens (indices 4-19)
        TEMPO: 20,             # 201 tokens (indices 20-220)
        CHORD: 221,            # 120 tokens (12 roots * 10 qualities) (indices 221-340)
        VELOCITY: 341,         # 32 tokens (indices 341-372)
        NOTE_ON: 373,          # 128 tokens (indices 373-500)
        DURATION: 501          # 64 tokens (indices 501-564)
    }


def event_to_int(event):
    try:
        if (event.type == None):
            print("TUUUUUU")
            print(vars(event))

        start_index = VOCAB_OFFSETS.get(event.type)

        if event.type in [SOS, EOS, BAR]:
            return start_index

        return start_index + event.value

    except Exception as e:
        print(f"Error in event_to_int with event {vars(event)}: {e}")
