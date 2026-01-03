import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import multiprocessing
from torch.amp import GradScaler, autocast
from constants import LEARNING_RATE, EPOCHS, VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEAD, D_FF, DROPOUT, DATA_PATH, \
    BATCH_SIZE, SEQUENCE_SIZE
from processing.data_loader import create_token_sequence_from_directory, create_token_sequence_from_npy_cache
from models.transformer_xl import MusicTransformerXL

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def batchify(data, batch_size, device):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.long)

    n_batch = data.size(0) // batch_size
    data = data[:n_batch * batch_size]
    data = data.view(batch_size, -1).contiguous()
    return data.to(device)


def get_batch(source, i, seq_len):
    seq_len = min(seq_len, source.size(1) - 1 - i)
    data = source[:, i:i + seq_len]
    target = source[:, i + 1:i + 1 + seq_len]
    return data, target


def train():
    print(f"Loading data from {DATA_PATH}...")

    # dataset_obj = create_event_sequence_from_directory("C:/Users/stszy/Downloads/maestro-v3.0.0-midi",
    #                                                use_cache=True, make_cache=True, clean_cache=False)
    dataset_obj = create_token_sequence_from_npy_cache()

    if dataset_obj is None:
        print("Error: Failed to load dataset.")
        return

    raw_data = dataset_obj.data
    print(f"Total tokens loaded: {len(raw_data)}")

    print("Batchifying data (creating parallel streams)...")
    train_data = batchify(raw_data, BATCH_SIZE, DEVICE)

    print("Initializing model...")
    model = MusicTransformerXL(
        vocab_size=VOCAB_SIZE,
        num_layers=N_LAYERS,
        d_model=D_MODEL,
        num_heads=N_HEAD,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler('cuda' if DEVICE.type == 'cuda' else 'cpu')

    print(f"Starting training on {DEVICE} for {EPOCHS} epochs with Batch Size {BATCH_SIZE}...")
    model.train()

    stream_len = train_data.size(1)

    for epoch in range(EPOCHS):
        total_loss = 0
        mems = None


        progress_bar = tqdm(range(0, stream_len - 1, SEQUENCE_SIZE), desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for i in progress_bar:
            x, y = get_batch(train_data, i, SEQUENCE_SIZE)

            optimizer.zero_grad()

            with autocast('cuda' if DEVICE.type == 'cuda' else 'cpu'):
                logits, mems = model(x, mems=mems)

                mems = [m.detach() for m in mems]

                loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(progress_bar)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

        save_dir = "checkpoints"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train()
