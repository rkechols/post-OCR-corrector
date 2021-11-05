import argparse
import os
import sys

import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from corpus.corrector_dataset import CorrectorDataset
from model.positional_encoding import PositionalEncoding
from util.data_functions import collate_sequences


class NeuralCorrector(pl.LightningModule):
    def __init__(self, alphabet_size: int, d_model: int = 512, max_len: int = 512):
        super().__init__()
        self.unk_index = alphabet_size
        self.bookend_index = alphabet_size + 1
        self.pad_index = alphabet_size + 2
        self.vocab_size = alphabet_size + 3
        self.embedding_src = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_index)
        self.embedding_tgt = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_index)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            # TODO: other hyperparameters
            norm_first=True
        )
        self.linear_stack = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.vocab_size)
        )
        self.criterion = nn.CrossEntropyLoss()  # TODO: label smoothing? label weights?
        self.max_len = max_len

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        # TODO: max sequence length?
        batch_size = x.shape[1]  # 0 = sequence, 1 = batch
        x_padding_mask = torch.where(x == -1, True, False)  # get padding mask for the input sequence
        x[x_padding_mask] = self.pad_index  # convert any -1 to the actual padding index
        x = self.positional_encoding(self.embedding_src(x.detach()))  # detach is so we don't need to back-prop through the data prep
        # put the input sequence into the encoder to get the "context"/"memory" sequence
        x = self.transformer.encoder(x, src_key_padding_mask=torch.permute(x_padding_mask, (1, 0)))
        # make a sequence to go in the decoder, gradually growing. starts with just a bookend
        sequence = torch.full((1, batch_size), self.bookend_index, dtype=torch.long, device=device)
        terminated = torch.zeros(batch_size, device=device).bool()  # keep track of which sequences have finished
        # make a padding mask that will grow with the sequence
        sequence_padding_mask = torch.zeros((batch_size, 1), device=device).bool()  # the mask is transposed the whole time
        while True:
            sequence_embed = self.positional_encoding(self.embedding_tgt(sequence))  # turn indices into embeddings
            new_thing = self.transformer.decoder(sequence_embed, x, tgt_key_padding_mask=sequence_padding_mask)[-1, :]  # only take the last token
            new_thing = self.linear_stack(new_thing)
            new_thing = torch.argmax(new_thing, dim=-1)
            # update which sequences are done
            terminated = terminated + (new_thing == self.bookend_index)  # '+' with bool tensors is element-wise 'or'
            if torch.all(terminated):
                break  # all pieces of the batch are done
            # if a sequence is done, force it to be only padding
            new_thing[terminated] = self.pad_index
            # actually add the new thing to the sequence
            sequence = torch.cat([sequence, new_thing.unsqueeze(0)], dim=0)
            if sequence.shape[0] > self.max_len:  # no more room; stop generating
                break  # TODO: do we want to do a sliding window to keep generating?
            # also update the padding mask, remembering that the mask is transposed
            sequence_padding_mask = torch.cat([sequence_padding_mask, torch.where(new_thing == self.pad_index, True, False).unsqueeze(1)], dim=1)
        sequence = sequence[1:, :]  # chop off the starting bookend
        sequence = torch.where(sequence == self.pad_index, -1, sequence)  # convert any padding to -1
        return sequence

    def training_step(self, batch, batch_idx) -> Tensor:
        # TODO: max sequence length?
        x, y = batch  # 0 = sequence, 1 = batch
        batch_size = x.shape[1]
        device = x.device
        # get padding masks
        x_padding_mask = torch.where(x == -1, True, False)
        y_padding_mask = torch.where(y == -1, True, False)
        # convert any -1 to the actual padding index
        x[x_padding_mask] = self.pad_index
        y[y_padding_mask] = self.pad_index
        # stick a "start token" at the beginning of the target sequence
        y_in = torch.cat([torch.full((1, batch_size), self.bookend_index, device=device), y], dim=0)
        # make padding masks
        x_padding_mask = torch.permute(x_padding_mask, (1, 0))  # also permute since that's what torch wants
        y_padding_mask = torch.permute(torch.cat([torch.zeros((1, batch_size), device=device).bool(), y_padding_mask], dim=0), (1, 0))
        # turn indices into embeddings
        x = self.positional_encoding(self.embedding_src(x.detach()))  # detach is so we don't need to back-prop through the data prep
        y_in = self.positional_encoding(self.embedding_tgt(y_in.detach()))
        # make a mask so it can't cheat when learning how to sequentially generate
        y_mask = nn.Transformer.generate_square_subsequent_mask(y_in.shape[0]).to(device)
        # pass the sequences through the main part of the model
        y_hat = self.transformer(x, y_in, tgt_mask=y_mask, src_key_padding_mask=x_padding_mask, tgt_key_padding_mask=y_padding_mask)
        y_hat = self.linear_stack(y_hat)
        # add a bookend token to the end of each sequence in y, so the loss is teaching it to end with a bookend
        y_target = torch.cat([y, torch.full((1, batch_size), self.pad_index, device=device)], dim=0)
        for i in range(batch_size):  # for each sequence
            for j in range(y_target.shape[0] - 2, -1, -1):  # go from the last slot toward the start (but skip the added padding)
                if y_target[j, i] != self.pad_index:  # looking for something that isn't padding
                    y_target[j + 1, i] = self.bookend_index  # add a bookend immediately after the last non-padding (the first one we see)
                    break
            else:  # the whole thing is padding? put the bookend at the start, I guess
                y_target[0, i] = self.bookend_index
        # transpose dimensions since cross entropy expects shape (batch, class, ...)
        y_hat = torch.permute(y_hat, (1, 2, 0))
        y_target = torch.permute(y_target, (1, 0))
        # get the loss
        loss = self.criterion(y_hat, y_target)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=4e-3)
        return optimizer


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("corpus_dir", type=str, help="File path to the directory containing the corpus to collect chars from.")
    arg_parser.add_argument("--cuda", type=int, default=None, help="Index of the CUDA device (GPU) to use.")
    arg_parser.add_argument("--cpu-limit", type=int, default=None, help="Max number of CPU processors to use.")
    args = arg_parser.parse_args()
    corpus_dir = args.corpus_dir
    cuda_index = args.cuda
    cpu_limit = args.cpu_limit

    if cuda_index is None:
        device_ = torch.device("cpu")
    elif cuda_index >= (cuda_count := torch.cuda.device_count()) or cuda_index < 0:
        print(f"WARNING: provided cuda index '{cuda_index}' is not valid (available count = {cuda_count}); defaulting to CPU", file=sys.stderr)
        device_ = torch.device("cpu")
    else:
        device_ = torch.device(f"cuda:{cuda_index}")

    if cpu_limit is None:  # use all we've got
        cpus = max(os.cpu_count(), 1)
    else:
        cpus = min(max(cpu_limit, 1), os.cpu_count())  # clip the provided number between 1 and os.cpu_count(), inclusive
    if cpus == 1:
        cpus = 0  # DataLoader expects 0 if we're not doing extra workers

    dataset = CorrectorDataset(corpus_dir, split="validation", tensors_out=True)
    dataloader = DataLoader(dataset, batch_size=3, num_workers=cpus, collate_fn=collate_sequences)
    model = NeuralCorrector(dataset.alphabet_size)
    model.to(device_)
    for batch_ in dataloader:
        batch_ = tuple(t.to(device_) for t in batch_)
        # try both functions
        print("starting to run things through the model...")
        loss_ = model.training_step(batch_, 0)
        print(f"{loss_=}")
        output = model(batch_[0])
        print(f"{output=}")
        # interpret output as a string
        print("\nGenerated outputs from untrained model:")
        alphabet = dataset.alphabet
        for i_ in range(output.shape[1]):
            sequence = list()
            for j_ in range(output.shape[0]):
                char_index = output[j_, i_].item()
                if char_index == -1:
                    break
                sequence.append(alphabet[char_index])
            sequence_str = "".join(sequence)
            print(f"\nOutput sequence {i_}: {sequence_str}")
            print(f"(Length: {len(sequence_str)})")
        break
