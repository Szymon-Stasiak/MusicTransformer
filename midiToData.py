def read_header(midi_data):
    if midi_data[0:4] != b'MThd':
        raise ValueError("Invalid MIDI file: missing 'MThd' header")
    length = int.from_bytes(midi_data[4:8], byteorder='big')
    format = int.from_bytes(midi_data[8:10], byteorder='big')
    tracks_quantity = int.from_bytes(midi_data[10:12], byteorder='big')
    division = int.from_bytes(midi_data[12:14], byteorder='big')
    return [length, format, tracks_quantity, division]


def print_header(header):
    length, format, tracks, division = header

    print(f"Header length: {length} bytes")

    if format == 0:
        format_str = "Single track"
    elif format == 1:
        format_str = "Multiple track"
    elif format == 2:
        format_str = "Multiple song"
    else:
        format_str = "Unknown format"
    print(f"Format: {format} ({format_str})")

    print(f"Number of tracks: {tracks}")

    if division & 0x8000:
        smpte_format = -(division >> 8)
        ticks_per_frame = division & 0xFF
        print(f"Time division: SMPTE {smpte_format} fps, {ticks_per_frame} ticks per frame")
    else:
        print(f"Time division: {division} ticks per beat")


def parse_midi(midi_data):
    header = read_header(midi_data)
    print_header(header)


    return [header]


if __name__ == "__main__":
    with open("./data/evaluation/000.midi", "rb") as f:
        data = f.read()
    parse_midi(data)
