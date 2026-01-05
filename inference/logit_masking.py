import torch
import torch.nn.functional as F
from constants import SOS, EOS, BAR, POSITION, TEMPO, CHORD, VELOCITY, NOTE_ON, DURATION, PAD

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TOKEN_RANGES = {
    PAD: range(0, 1),
    SOS: range(1, 2),
    EOS: range(2, 3),
    BAR: range(3, 4),
    POSITION: range(4, 20),
    TEMPO: range(20, 221),
    CHORD: range(221, 341),
    VELOCITY: range(341, 373),
    NOTE_ON: range(373, 501),
    DURATION: range(501, 565)
}


def apply_grammar_mask(logits, last_token):
    mask = torch.full_like(logits, float('-inf'))

    def allow(range_name):
        r = TOKEN_RANGES[range_name]
        mask[r] = logits[r]

    if last_token in TOKEN_RANGES[VELOCITY]:
        allow(NOTE_ON)
    elif last_token in TOKEN_RANGES[NOTE_ON]:
        allow(DURATION)
    elif last_token in TOKEN_RANGES[DURATION]:
        allow(VELOCITY)
        allow(POSITION)
        allow(BAR)
        allow(EOS)
        allow(TEMPO)
        allow(CHORD)
    elif last_token in TOKEN_RANGES[BAR]:
        allow(POSITION)
        allow(EOS)
    elif last_token in TOKEN_RANGES[POSITION]:
        allow(TEMPO)
        allow(CHORD)
        allow(VELOCITY)
    elif last_token in TOKEN_RANGES[TEMPO]:
        allow(CHORD)
        allow(VELOCITY)
    elif last_token in TOKEN_RANGES[CHORD]:
        allow(VELOCITY)
        allow(POSITION)
    elif last_token in TOKEN_RANGES[SOS]:
        allow(BAR)
    else:
        return logits

    if torch.all(mask == float('-inf')):
        return logits

    return mask


def generate_logit_masking(model, prime_sequence=[1], gen_len=512, temperature=1.0, top_k=40, top_p=0.9):
    model.eval()

    if prime_sequence[0] == 0:
        prime_sequence = [1]

    current_seq = torch.tensor(prime_sequence, dtype=torch.long).unsqueeze(0).to(DEVICE)
    mems = None
    generated_tokens = list(prime_sequence)

    print(f"Generating {gen_len} tokens with Grammar Masking (EOS Blocked)...")

    with torch.no_grad():
        for i in range(gen_len):
            logits, mems = model(current_seq, mems=mems)
            last_logit_1d = logits[0, -1, :]
            last_token_id = generated_tokens[-1]
            last_logit_masked = apply_grammar_mask(last_logit_1d, last_token_id)

            if i < gen_len - 1:
                last_logit_masked[TOKEN_RANGES["EOS"]] = float('-inf')

            last_logit_masked = last_logit_masked / temperature
            last_logit_2d = last_logit_masked.unsqueeze(0)

            if top_k > 0:
                top_k_val = torch.topk(last_logit_2d, top_k)[0][..., -1, None]
                indices_to_remove = last_logit_2d < top_k_val
                last_logit_2d[indices_to_remove] = float('-inf')

            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(last_logit_2d, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                last_logit_2d[indices_to_remove] = float('-inf')

            probs = F.softmax(last_logit_2d, dim=-1)

            if torch.isnan(probs).any():
                probs = torch.zeros_like(probs)
                probs[0, TOKEN_RANGES["BAR"][0]] = 1.0

            next_token = torch.multinomial(probs, 1).item()
            generated_tokens.append(next_token)
            current_seq = torch.tensor([[next_token]], dtype=torch.long).to(DEVICE)

    return generated_tokens
