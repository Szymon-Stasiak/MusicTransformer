VOCAB_OFFSETS = {
        "PAD": 0,         # Empty space (padding) - optional, but recommended
        "SOS": 1,         # Start of Sequence
        "EOS": 2,         # End of Sequence
        "BAR": 3,         # 1 token
        "POS": 4,         # 16 tokens (indices 4-19)
        "TEMPO": 20,      # 201 tokens (indices 20-220)
        "CHORD": 221,     # 120 tokens (12 roots * 10 qualities) (indices 221-340)
        "VEL": 341,       # 32 tokens (indices 341-372)
        "NOTE": 373,      # 128 tokens (indices 373-500)
        "DUR": 501        # 64 tokens (indices 501-564)
    }

def event_to_int(event):


    start_index = VOCAB_OFFSETS.get(event.type)

    if event.type in ["SOS", "EOS", "BAR"]:
        return start_index


    return start_index + event.value
