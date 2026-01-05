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

## Results

Below is a sample composition generated by the model, demonstrating its ability to maintain rhythm, structure, and harmony over time.

#todolater

[🎵 **Download/Listen to Generated Sample (MIDI)**](path/to/your/generated_file.mid)

## Conclusion
#todolater

##  Requirements

- Python 3.6+
- PyTorch 1.0+
- music21
- numpy

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

