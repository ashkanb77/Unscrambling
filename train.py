import torch
from tqdm import tqdm
from utils import read_dataset, collate_fn
from dataset import UnscramblingDataset
from argparse import ArgumentParser
from model import UnscramblingModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch import nn
from transformers import get_linear_schedule_with_warmup
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time


parser = ArgumentParser()

parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate')
parser.add_argument('--tokenizer', type=str, default='m3hrdadfi/bert2bert-fa-wiki-summary', help='tokenizer')
parser.add_argument(
    '--train_dataset', type=str,
    default='dataset/train_informal.csv', help='train sentences'
)
parser.add_argument(
    '--val_dataset', type=str,
    default='dataset/val_informal.csv', help='test sentences'
)
parser.add_argument('--model_name', type=str, default='m3hrdadfi/bert2bert-fa-wiki-summary', help='dataset directory')

args = parser.parse_args()

experiment = str(int(time.time()))
writer = SummaryWriter('runs/' + experiment)

train_sentences, val_sentences = read_dataset(args.train_dataset)

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

train_dataset = UnscramblingDataset(train_sentences)
val_dataset = UnscramblingDataset(val_sentences)


train_dataloader = DataLoader(train_dataset, collate_fn=lambda data: collate_fn(data, tokenizer),
                              batch_size=args.batch_size)
val_dataloader = DataLoader(val_dataset, collate_fn=lambda data: collate_fn(data, tokenizer),
                            batch_size=args.batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = UnscramblingModule(args.model_name)
model.to(device)

print(f"You are using {device}")
print()

# training config: optimizer, scheduler and criterion
optimizer = AdamW(model.parameters(), lr=args.learning_rate)

total_steps = len(train_dataloader) * args.epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0,
    num_training_steps=total_steps
)


def train_epoch(model, dataloader, optimizer, scheduler, epoch):
    losses = []
    model.train()

    with tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Train Epoch {epoch + 1}/{args.epochs}")

            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(**batch)
            loss = output['loss']
            losses.append(loss.item())

            # optimize
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            tepoch.set_postfix(loss=loss.item())

    loss = np.mean(losses)
    writer.add_scalar('Loss/train', loss, epoch)
    return loss


def eval_model(model, dataloader, epoch):
    losses = []
    model.eval()

    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Validation Epoch {epoch + 1}/{args.epochs}")

                batch = {k: v.to(device) for k, v in batch.items()}

                output = model(**batch)
                loss = output['loss']

                losses.append(loss.item())
                tepoch.set_postfix(loss=loss.item())

        loss = np.mean(losses)
        writer.add_scalar('Loss/validation', loss, epoch)
        return loss


losses = []
val_losses = []

best_loss = 100

for epoch in range(args.epochs):

    # train one epoch
    train_loss = train_epoch(
        model,
        train_dataloader,
        optimizer,
        scheduler, epoch
    )

    print(f'Train loss {train_loss:0.4f}')

    # evaluate
    val_loss = eval_model(
        model, val_dataloader, epoch
    )

    print(f'Validation loss {val_loss:0.4f}')
    print()

    # save history
    losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_loss:  # save model if its accuracy is bigger than best model accuracy
        torch.save(model.state_dict(), experiment + '.pth')
        best_loss = val_loss

print(f"Best Loss is {best_loss:0.4f}")