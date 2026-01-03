import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing
from torch.amp import GradScaler, autocast

from constants import LEARNING_RATE, EPOCHS, VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEAD, D_FF, DROPOUT, DATA_PATH, BATCH_SIZE
from processing.data_loader import create_event_sequence_from_directory, create_event_sequence_from_npy_cache
from models.transformer_xl import MusicTransformerXL

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def train():
    print(f"Loading data from {DATA_PATH}...")
    # dataset = create_event_sequence_from_directory(DATA_PATH,
    #                                                use_cache=True, make_cache=True, clean_cache=False)
    dataset = create_event_sequence_from_npy_cache()
    cpu_count = multiprocessing.cpu_count()

    num_workers = max(2, min(8, cpu_count))

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers,
                            pin_memory=DEVICE.type == 'cuda')

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

    for epoch in range(EPOCHS):
        total_loss = 0
        mems = None
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with autocast('cuda' if DEVICE.type == 'cuda' else 'cpu'):
                logits, mems = model(x, mems=mems)

                if mems is not None:
                    mems = [m.detach() for m in mems]

                loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

        save_path = f"/kaggle/working/checkpoints/model_epoch_{epoch + 1}.pth"
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")

        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train()
