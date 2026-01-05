import torch
import torch.nn.functional as F
import os
from models.transformer_xl import MusicTransformerXL
from constants import VOCAB_SIZE, N_LAYERS, D_MODEL, N_HEAD, D_FF
from processing.decoder import save_tokens_to_txt
from processing.decoder import create_midis_from_tokens

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpoint_path):
    model = MusicTransformerXL(
        vocab_size=VOCAB_SIZE,
        num_layers=N_LAYERS,
        d_model=D_MODEL,
        num_heads=N_HEAD,
        d_ff=D_FF,
        dropout=0.0
    ).to(DEVICE)

    print(f"Loading model from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate(model, prime_sequence=[1], gen_len=512, temperature=1.0):
    model.eval()

    current_seq = torch.tensor(prime_sequence, dtype=torch.long).unsqueeze(0).to(DEVICE)
    mems = None
    generated_tokens = list(prime_sequence)

    print(f"Generating {gen_len} tokens with temperature {temperature}...")

    with torch.no_grad():
        for _ in range(gen_len):
            logits, mems = model(current_seq, mems=mems)

            last_logit = logits[0, -1, :] / temperature
            probs = F.softmax(last_logit, dim=-1)

            next_token = torch.multinomial(probs, 1).item()

            generated_tokens.append(next_token)

            current_seq = torch.tensor([[next_token]], dtype=torch.long).to(DEVICE)

    return generated_tokens


if __name__ == "__main__":
    CHECKPOINT = "../training/checkpoints/model_epoch_4.pth"
    OUTPUT_FILE = "generated_music"

    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint {CHECKPOINT} not found! Train model first.")
        exit()

    model = load_model(CHECKPOINT)

    tokens = generate(model, gen_len=500, temperature=1.0)

    print("Saving generated tokens...")
    save_tokens_to_txt(tokens, OUTPUT_FILE)

    print(f"Saving outputs to {OUTPUT_FILE}...")
    create_midis_from_tokens(tokens, OUTPUT_FILE)
    print("Done!")
