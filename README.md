# Music Transformer – REMI-based MIDI Modeling

This repository contains an implementation inspired by the paper:

> **Pop Music Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Compositions**  
> Yu-Siang Huang, Yi-Hsuan Yang  
> https://arxiv.org/abs/2002.00212

The goal of this project is to build a **Transformer-based model for symbolic music generation**, operating on MIDI-derived event sequences using a **REMI-like representation**.

---

## MIDI Analysis and REMI Representation

Before implementing the Transformer model, the MIDI files were analyzed in order to extract musically meaningful information and convert it into a format suitable for sequence modeling.

For MIDI parsing and music-theoretical analysis, the **music21** library was used.  
music21 provides advanced functionalities for:

- note extraction
- chord recognition
- tempo change detection
- metric structure analysis (bars and beats)

These features were essential for converting raw MIDI data into a structured event-based representation.

---

## REMI Design

The symbolic representation is inspired by **REMI (REvamped MIDI-derived events)** and is based on a **linear sequence of discrete musical events**, which can be directly modeled using an autoregressive Transformer.

The REMI representation consists of three main event types:

- **Note** – represents pitch and duration of individual notes  
- **Chord** – provides harmonic context  
- **Tempo** – represents tempo changes to preserve rhythmic structure  

Each event is encoded as a **separate token**, forming a single sequential stream.

---

## Temporal Representation

Time is represented using **musical units instead of absolute time**.

- The base unit of time is a **quarter note**
- Each quarter note is subdivided into **4 equal steps**
- This results in **16 time steps per bar** (16th-note resolution in 4/4 meter)

This beat-based discretization:

- reduces vocabulary size
- improves rhythmic consistency
- allows the model to learn musically meaningful temporal patterns

Bar boundaries are explicitly represented to preserve metrical structure.

---

## Model Architecture

todo later
---

##  Requirements

- Python 3.6+
- PyTorch 1.0+
- music21
- numpy

Install the required dependencies using:

```bash
pip install -r requirements.txt
