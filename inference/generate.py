import argparse
import torch
import torch.nn.functional as F
import os
import sys

from models.transformer_xl import MusicTransformerXL
from constants import VOCAB_SIZE, N_LAYERS, D_MODEL, N_HEAD, D_FF
from processing.decoder import save_tokens_to_txt
from processing.decoder import create_midis_from_tokens
from inference.logit_masking import generate_logit_masking

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
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
    except Exception as e:
        print(f"ERROR: Could not load checkpoint from {checkpoint_path}")
        print(f"Details: {e}")
        sys.exit(1)

    model.eval()
    return model


def generate_simple(model, prime_sequence=[1], gen_len=512, temperature=1.0):

    model.eval()

    current_seq = torch.tensor(prime_sequence, dtype=torch.long).unsqueeze(0).to(DEVICE)
    mems = None
    generated_tokens = list(prime_sequence)

    print(f"Generating {gen_len} tokens (Simple Mode) with temperature {temperature}...")

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
    parser = argparse.ArgumentParser(description="Music Transformer Generator")

    parser.add_argument('--checkpoint', '-c', type=str, default='./model_epoch_31.pth',
                        help='Path to the model checkpoint file (.pth)')

    parser.add_argument('--output', '-o', type=str, default='generated_music',
                        help='Output filename (without extension)')

    parser.add_argument('--length', '-l', type=int, default=500,
                        help='Number of tokens to generate')

    parser.add_argument('--temperature', '-t', type=float, default=1.0,
                        help='Sampling temperature (higher = more random)')

    parser.add_argument('--simple', action='store_true',
                        help='Use simple generation WITHOUT grammar masking (not recommended)')

    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-K filtering parameter (only for masking mode)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-P nucleus sampling parameter (only for masking mode)')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found! Train model first or check path.")
        sys.exit(1)

    model = load_model(args.checkpoint)

    if args.simple:
        tokens = generate_simple(
            model,
            gen_len=args.length,
            temperature=args.temperature
        )
    else:
        tokens = generate_logit_masking(
            model,
            gen_len=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )

    print(f"Saving {len(tokens)} tokens to {args.output}.txt...")
    save_tokens_to_txt(tokens, args.output)

    print(f"Converting to MIDI: {args.output}.mid...")
    create_midis_from_tokens(tokens, args.output)

    print("Done! Enjoy your music.")
