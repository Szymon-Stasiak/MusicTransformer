import numpy as np
from entities.remi_item import RemiItem
from entities.event import Event
from utils import fit_to_boundary, get_position, get_chord_id
from constants import BAR, EOS, SOS, TEMPO, POSITION, NOTE_ON, DURATION, CHORD, VELOCITY


def group_items(notes, tempo, chords=None):
    if chords is None:
        chords = []

    items = notes + tempo + chords
    items.sort(key=lambda x: x.start)

    BAR_STEPS = 16

    if not items:
        return []

    max_time = max(item.start for item in items)
    downbeats = np.arange(0, max_time + BAR_STEPS, BAR_STEPS)

    groups = []

    for db_start, db_end in zip(downbeats[:-1], downbeats[1:]):
        insiders = [
            item for item in items
            if db_start <= item.start < db_end
        ]

        filtered = []
        tempo_seen = False
        for elem in insiders:
            if isinstance(elem, RemiItem) and elem.type == "Tempo":
                if not tempo_seen:
                    filtered.append(elem)
                    tempo_seen = True
            else:
                filtered.append(elem)

        groups.append(filtered)

    return groups


def create_list_of_events(groups):
    events = []

    events.append(Event(SOS, None))

    for g in groups:
        events.append(Event(BAR, None))
        for i in range(len(g)):
            if (i == 0 or g[i].start != g[i - 1].start):
                events.append(Event(POSITION, get_position(g[i].start)))
            if (g[i].type == "Tempo"):
                events.append(Event(TEMPO, fit_to_boundary(g[i].value, 30, 230)))
            if (g[i].type == "Chord"):
                events.append(Event(CHORD, get_chord_id(g[i].value, g[i].descriptor)))
            if (g[i].type == "Note"):
                events.append(Event(VELOCITY, fit_to_boundary(g[i].descriptor // 4, 0, 31)))
                events.append(Event(NOTE_ON, fit_to_boundary(g[i].value, 0, 127)))
                events.append(Event(DURATION, fit_to_boundary(g[i].end - g[i].start, 1, 64)))
    events.append(Event(EOS, None))
    return events
