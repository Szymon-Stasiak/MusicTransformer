import numpy as np
from remi_item import RemiItem
from chord_extractor import chords_from_midi
from notes_and_tempo_extractor import get_items_from_midi_file
from music21 import converter
from event import Event
from utils import fit_to_boundary, get_position, get_chord_id
from constants import BAR, EOS, SOS, TEMPO, POSITION, NOTE_ON, DURATION, CHORD, VElOCITY


def group_items(notes, tempo, chords=None):
    if chords is None:
        chords = []

    items = notes + tempo + chords
    items.sort(key=lambda x: x.start)

    BAR_STEPS = 16

    if not items:
        return []

    max_time = items[-1].end
    downbeats = np.arange(0, max_time + BAR_STEPS, BAR_STEPS)

    groups = []

    for db_start, db_end in zip(downbeats[:-1], downbeats[1:]):
        insiders = [
            item for item in items
            if item.start >= db_start and item.start < db_end
        ]
        overall = insiders

        filtered = []
        tempo_seen = False
        for elem in overall:
            if isinstance(elem, RemiItem) and elem.name == "Tempo":
                if not tempo_seen:
                    filtered.append(elem)
                    tempo_seen = True
            else:
                filtered.append(elem)

        groups.append(filtered)

    return groups


def create_list_of_groups(filepath):
    try:
        score = converter.parse(filepath)
    except Exception as e:
        print(f"Error parsing: {e}")
        return []

    note_items, tempo_items = get_items_from_midi_file(score)
    chords = chords_from_midi(score)
    return group_items(note_items, tempo_items, chords)


def create_list_of_events(groups):
    events = []

    events.append(Event(SOS, None))

    for g in groups:
        events.append(Event(BAR, None))
        for i in range(len(g)):
            if (i == 0 or g[i].start != g[i - 1].start):
                events.append(Event(POSITION, get_position(g[i].start)))
            if (g[i].name == "Tempo"):
                events.append(Event(TEMPO, fit_to_boundary(g[i].value, 30, 230)))
            if (g[i].name == "Chord"):
                events.append(Event(CHORD, get_chord_id(g[i].value, g[i].descriptor)))
            if (g[i].name == "Note"):
                events.append(Event(VElOCITY, fit_to_boundary(g[i].descriptor // 4, 0, 31)))
                events.append(Event(NOTE_ON, fit_to_boundary(g[i].value, 0, 127)))
                events.append(Event(DURATION, fit_to_boundary(g[i].end - g[i].start, 1, 64)))
    events.append(Event(EOS, None))
    return events
