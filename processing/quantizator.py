from constants import QUANTIZATION_RESOLUTION as RESOLUTION


def quantize_items_16th(items):
    for item in items:

        q_start = int(round(item.start * RESOLUTION))
        q_end = int(round(item.end * RESOLUTION))

        if q_end <= q_start:
            q_end = q_start + 1

        item.start = q_start
        item.end = q_end

    return items


def quantize_tempo_16th(items):
    for item in items:
        q_start = int(round(item.start * RESOLUTION))

        item.start = q_start

    return items
