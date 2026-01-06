# Music Transformer-XL – REMI-based MIDI Modeling

This repository contains a Transformer-based model for **symbolic music generation**, operating on MIDI-derived event sequences using a **REMI-like representation**.

The project is **inspired by** the paper:

> **Pop Music Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Compositions**  
> Yu-Siang Huang, Yi-Hsuan Yang  
> https://arxiv.org/abs/2002.00212

**Important note:**  
While this work draws conceptual inspiration from the Pop Music Transformer paper—particularly the idea of beat-based event modeling—it **does not implement the original method one-to-one**. Several components of the representation, data processing pipeline, and model design have been **modified, simplified, or reinterpreted** to better suit the goals of this project and practical experimentation.

As a result, this repository should be viewed as a **REMI-inspired Transformer-XL approach**, rather than a faithful reproduction of the original paper’s architecture or training setup.

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

The model is based on the **Transformer-XL** architecture, which is designed to model long-term dependencies using **segment-level recurrence**. By reusing hidden states from previous segments, the model can preserve musical context across long sequences, making it well-suited for symbolic music generation.

---

### Event-Based Music Representation

Music is represented as a sequence of **discrete event tokens**, where each token corresponds to a specific musical concept. All tokens are mapped into a single continuous vocabulary with fixed index ranges assigned to each category.

The vocabulary consists of the following token groups:

- **Padding (PAD)** A special token used for sequence padding during batching.

- **Start of Sequence (SOS)** Indicates the beginning of a musical sequence.

- **End of Sequence (EOS)** Marks the end of a musical sequence.

- **Bar** A single token used to denote the start of a new bar (measure), providing high-level rhythmic structure.

- **Position** 16 tokens representing discrete positions within a bar, allowing the model to learn rhythmic timing.

- **Tempo** 201 tokens encoding tempo values, enabling tempo changes within a sequence.

- **Chord** 120 tokens representing harmonic content, defined as combinations of 12 pitch-class roots and 10 chord qualities.

- **Velocity** 32 tokens representing note velocity, capturing expressive dynamics.

- **Note On (Pitch)** 128 tokens corresponding to MIDI pitch values.

- **Duration** 64 tokens encoding note duration values.

The total vocabulary size is **565 tokens**.

---

### Sequence Structure

Musical sequences follow an autoregressive, event-based format. A typical sequence begins with a start token and ends with an end token, with musical events in between such as bar markers, positions, tempo changes, harmonic context, and note-level information.

The model is trained to predict the next token in the sequence given all previous tokens, including contextual information preserved in the Transformer-XL memory.

---

### Embeddings and Attention mechanism

Each token is mapped to a learned embedding vector. The architecture explicitly utilizes:

- **Relative Multi-Head Attention:** Unlike standard attention mechanisms, this model uses relative attention to better capture the distance between musical events.
- **Positional Embeddings:** The model incorporates relative positional encodings within the attention mechanism. This allows the model to generalize motifs and rhythmic patterns regardless of their absolute position in the sequence (translation invariance), which is crucial for musical structure.
- **Causal (Masked) Self-Attention:** Ensures the model can only attend to past events, preventing information leakage from the future.

---

## Configuration & Hyperparameters

The model is configured with specific parameters optimized for learning symbolic music structures.

### Data & Training Configuration
* **Sequence Size:** 512
* **Recurrence Length:** 1024 (`2 * SEQUENCE_SIZE`) - Allows the model to reference past context.
* **Batch Size:** 16
* **Epochs:** 50
* **Learning Rate:** 0.0001
* **Data Path:** `../data/train/`
* **Cache Directory:** `data_cache`

### Architecture Parameters
* **Vocabulary Size:** 565
* **Embedding Dimension ($d_{model}$):** 512
* **Number of Layers:** 12
* **Attention Heads:** 8
* **Feed-Forward Dimension ($d_{ff}$):** 2048
* **Dropout:** 0.1

---

## Training Objective

### Loss Function
The model is trained using **Cross Entropy Loss**.

Cross Entropy is the standard objective function for autoregressive language modeling and multi-class classification problems.
* **Discrete Nature:** Our music generation task involves predicting the next token from a fixed, discrete vocabulary of 565 classes (Pithes, Durations, Tempos, etc.).
* **Probability Distribution:** The model outputs a probability distribution (via Softmax) over the vocabulary. Cross Entropy effectively minimizes the divergence between this predicted distribution and the actual ground-truth token (one-hot encoding).
* **Optimization:** It provides a strong gradient signal for the model to learn to assign high probabilities to the correct musical events while suppressing incorrect ones.

---

## Usage

To generate new musical sequences using the trained model, run the inference script. This script loads the model weights and generates symbolic music output based on the learned probabilities.
Use **inference/generate.py** for music generation. Remember to adjust the parameters as needed.

## Dataset

The model is trained on the **MAESTRO (MIDI and Audio Edited for Synchronous TRacks and Organization) Dataset v3.0.0**. This dataset serves as a benchmark for piano performance generation, containing over 200 hours of virtuosic piano performances from the *International Piano-e-Competition*.

### Dataset Characteristics
* **Source:** Real-world performances captured on Yamaha Disklavier pianos, ensuring high-fidelity capture of key presses and pedal movements.
* **Format:** The project utilizes the **MIDI files** from the dataset, which provide precise symbolic representations of the music.
* **Repertoire:** A vast collection of classical piano music (e.g., Chopin, Liszt, Rachmaninoff, Debussy).
* **Expressiveness:** Unlike quantized sheet music, this dataset captures **human expression**, including dynamics (velocity) and micro-timing (rubato), allowing the model to learn and generate expressive, human-like performances.

Official webpage: https://magenta.tensorflow.org/datasets/maestro

## Training Details

The training process was conducted using parallel processing on a single high-performance GPU. Model was trained using parameters specified in the **Configuration & Hyperparameters** section above.

* **Hardware:** NVIDIA Tesla P100 GPU
* **Training Duration:** Approximately **40 hours** (48 minutes per epoch for 50 epochs)

---

##  Requirements

- Python 3.6+
- PyTorch 1.0+
- music21
- numpy

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Music Generation (Inference)

The `generate.py` script allows you to compose new music using your trained Transformer model. 

It features a built-in **Logit/Grammar Masking** mechanism. This ensures the generated MIDI files are syntactically valid by preventing common errors (like generating a Note-On event without a subsequent Duration) and blocking the model from ending the track too early.

### Quick Start

To generate music using the default settings (looking for `model_epoch_31.pth` in the current directory), simply run:

```bash
python generate.py
```

###  Command Line Arguments

You can customize the generation process using the following flags:

| Flag | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--checkpoint` | `-c` | `./model_epoch_31.pth` | Path to the trained model weights file (`.pth`). |
| `--output` | `-o` | `generated_music` | Output filename (without extension). |
| `--length` | `-l` | `500` | Number of tokens to generate (length of the piece). |
| `--temperature` | `-t` | `1.0` | Sampling temperature. <br>`<1.0`: More conservative/repetitive.<br>`>1.0`: More chaotic/creative. |
| `--top_k` | - | `40` | Limits sampling to the `K` most likely tokens (removes noise/bad notes). |
| `--top_p` | - | `0.9` | Nucleus Sampling. Selects from the smallest set of tokens whose cumulative probability is 90%. |
| `--simple` | - | `False` | **Debug Mode.** Disables Grammar Masking. Uses the raw model output without rules (not recommended). |


### Temperature (`--temperature`)

Controls the randomness of token sampling during generation.

- `< 1.0` – More conservative and repetitive output  
- `> 1.0` – More creative, diverse, and potentially chaotic output  

---

## Usage Examples

### 1️ High Quality Generation (Recommended)

Uses grammar masking to ensure a valid MIDI structure.

```bash
python generate.py --checkpoint model_epoch_50.pth --output my_new_song
```
### 2️ Long & Creative Piece

Increases the sampling temperature for more expressive and diverse musical ideas  
and generates a longer sequence (1000 tokens).

```bash
python generate.py -c model_epoch_50.pth -o jazz_session -l 1000 -t 1.2
```
### 3️ Safe & Conservative

Uses a lower sampling temperature to favor high-probability tokens.

This mode produces more stable and predictable music with fewer structural issues,  
but the output may be more repetitive and less adventurous.

```bash
python generate.py -t 0.8 -o safe_melody
```

### 4 Raw Model Test (No Constraints)

Disables grammar masking and uses the raw model output without any structural rules.

This mode is intended for debugging and experimentation, allowing you to observe  
how the model behaves without the safety constraints. The generated output may  
contain invalid or incomplete MIDI structures.

```bash
python generate.py --simple
```

## How Grammar Masking Works

During the generation loop, the script actively applies rule-based constraints  
to the model’s output in order to guarantee a syntactically valid MIDI sequence.

- **Token Order Enforcement**  
  A `NOTE_ON` token is only allowed **after a `VELOCITY` token**.  
  This ensures that every note has an explicitly defined velocity before it starts.

- **Note Structure Enforcement**  
  When the model generates a `NOTE_ON` token, the next token is forced to be  
  `DURATION`. This prevents hanging notes and incomplete note definitions.

- **EOS (End of Sequence) Blocking**  
  The model is prevented from selecting the `EOS` token until the requested  
  `--length` is reached, ensuring the generated track has the desired length.

These rules act as a safety layer on top of the model’s predictions, improving  
the musical validity of the output without retraining the model.

---

## Outputs

After generation completes, the following files are produced:

- `filename.mid` – The generated, playable MIDI file  
- `filename.txt` – A plain text file containing the generated token sequence  
  (useful for debugging and analysis)  
- `filename_humanized.mid` – A humanized version of the MIDI file with slight  
  timing shifts to make it sound more natural and played by a human

---

