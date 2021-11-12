import argparse
import os
import sys
from math import ceil
from typing import List, Tuple

import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from corpus.corrector_dataset import CorrectorDataset
from model.positional_encoding import PositionalEncoding
from util import INT_EMPTY, UNK
from util.data_functions import collate_sequences, collate_single_column, get_alphabet, text_to_tensor


class NeuralCorrector(pl.LightningModule):
    def __init__(self, data_dir: str, cpus: int = None, max_len: int = 512,
                 d_model: int = 512,
                 n_head: int = 8,
                 n_layers: int = 6,
                 d_linear: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 label_smoothing: float = 0.0,
                 lr: float = 4e-3,
                 batch_size: int = 8):
        super().__init__()
        self.data_dir = data_dir
        self.alphabet = get_alphabet(data_dir)
        alphabet_size = len(self.alphabet)
        # unk_index MUST be the first one after the valid alphabet indices because of how util.data_functions.text_to_tensor works
        self.unk_index = alphabet_size
        self.bookend_index = alphabet_size + 1
        self.pad_index = alphabet_size + 2
        self.vocab_size = alphabet_size + 3
        if cpus is None:
            self.cpus = os.cpu_count()
        else:
            self.cpus = cpus
        self.max_len = max_len
        self.embedding_src = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_index)
        self.embedding_tgt = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_index)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=d_linear,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            norm_first=True
        )
        self.linear_stack = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.vocab_size)
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.lr = lr
        self.batch_size = batch_size

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[0] > self.max_len:
            print(f"WARNING: truncating input sequence to length {self.max_len}", file=sys.stderr)
            x = x[:self.max_len, :]
        in_length, batch_size = x.shape
        device = self.device
        x_padding_mask = torch.where(x == INT_EMPTY, True, False)  # get padding mask for the input sequence
        x[x_padding_mask] = self.pad_index  # convert any INT_EMPTY to the actual padding index
        x = self.positional_encoding(self.embedding_src(x.detach()))  # detach is so we don't need to back-prop through the data prep
        # put the input sequence into the encoder to get the "context"/"memory" sequence
        x = self.transformer.encoder(x, src_key_padding_mask=torch.permute(x_padding_mask, (1, 0)))
        # make a sequence to go in the decoder, gradually growing. starts with just a bookend
        sequence = torch.full((1, batch_size), self.bookend_index, dtype=torch.long, device=device)
        terminated = torch.zeros(batch_size, device=device).bool()  # keep track of which sequences have finished
        # make a padding mask that will grow with the sequence
        sequence_padding_mask = torch.zeros((batch_size, 1), device=device).bool()  # the mask is transposed the whole time
        while sequence.shape[0] <= 2 * in_length:  # stop generating when it gets unreasonably long
            # turn indices into embeddings and generate one more token
            if sequence.shape[0] > self.max_len:  # too long; give only the last `max_len` tokens
                sequence_embed = self.positional_encoding(self.embedding_tgt(sequence[-self.max_len:, :]))
                new_thing = self.transformer.decoder(sequence_embed, x, tgt_key_padding_mask=sequence_padding_mask[:, -self.max_len:])[-1, :]  # only take the last token
            else:
                sequence_embed = self.positional_encoding(self.embedding_tgt(sequence))
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
            # also update the padding mask, remembering that the mask is transposed
            sequence_padding_mask = torch.cat([sequence_padding_mask, torch.where(new_thing == self.pad_index, True, False).unsqueeze(1)], dim=1)
        sequence = sequence[1:, :]  # chop off the starting bookend
        sequence = torch.where(sequence == self.pad_index, INT_EMPTY, sequence)  # convert any padding to INT_EMPTY
        return sequence

    def tensor_to_texts(self, t: Tensor) -> List[str]:
        to_return = list()
        for i in range(t.shape[1]):  # each sequence in the batch
            sequence = list()
            for j in range(t.shape[0]):
                char_index = t[j, i].item()
                if char_index == INT_EMPTY:
                    break
                elif char_index == self.unk_index:
                    char = UNK
                else:
                    try:
                        char = self.alphabet[char_index]
                    except IndexError:
                        print(f"ERROR - unknown char index: {char_index} (max expected is {self.unk_index})", file=sys.stderr)
                        char = UNK
                sequence.append(char)
            sequence_str = "".join(sequence)
            to_return.append(sequence_str)
        return to_return

    def correct(self, texts: List[str]) -> List[str]:
        to_return = list()
        next_text = 0
        n = len(texts)
        self.eval()
        with torch.no_grad():
            while next_text < n:
                batch_texts = texts[next_text:(next_text + self.batch_size)]
                longest = max(len(x) for x in batch_texts)
                out_texts_chunks = [list() for _ in range(len(batch_texts))]
                for chunk_num in range(ceil(longest / self.max_len)):
                    chunk_start = chunk_num * self.max_len
                    chunk_end = chunk_start + self.max_len
                    in_chunks = [text[chunk_start:chunk_end] for text in batch_texts]  # get the relevant chunk of each text in the batch
                    batch_tensors = [text_to_tensor(text, self.alphabet) for text in in_chunks]  # turn each chunk into a tensor
                    out = self(collate_single_column(batch_tensors).to(self.device))  # stack the tensors into a batch tensor and put the batch through the model
                    out = self.tensor_to_texts(out)  # convert the tensor to a list of output strings
                    for i, (in_text, out_text) in enumerate(zip(in_chunks, out)):  # put each chunk into the appropriate list with other chunks of the same sequence
                        if in_text != "":  # but skip ones where the input has already ran out
                            out_texts_chunks[i].append(out_text)
                to_return += ["".join(chunks) for chunks in out_texts_chunks]  # turn each list of chunks into a full output sequence
                next_text += self.batch_size
        return to_return

    def forward_with_target(self, x: Tensor, y: Tensor) -> Tensor:
        if x.shape[0] > self.max_len:
            print(f"WARNING: truncating input sequence to length {self.max_len}", file=sys.stderr)
            x = x[:self.max_len, :]
        if y.shape[0] >= self.max_len:
            print(f"WARNING: truncating target sequence to length {self.max_len - 1}", file=sys.stderr)
            y = y[:(self.max_len - 1), :]
        batch_size = x.shape[1]
        device = self.device
        # get padding masks
        x_padding_mask = torch.where(x == INT_EMPTY, True, False)
        y_padding_mask = torch.where(y == INT_EMPTY, True, False)
        # convert any INT_EMPTY to the actual padding index
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
        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss = self.forward_with_target(*batch)
        self.log("ptl/train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss = self.forward_with_target(*batch)
        self.log("ptl/val_loss", loss)
        return loss

    def train_dataloader(self) -> DataLoader:
        dataset_train = CorrectorDataset(self.data_dir, split="train", tensors_out=True)
        return DataLoader(dataset_train, batch_size=self.batch_size, num_workers=self.cpus, collate_fn=collate_sequences)

    def val_dataloader(self) -> DataLoader:
        dataset_val = CorrectorDataset(self.data_dir, split="validation", tensors_out=True)
        return DataLoader(dataset_val, batch_size=self.batch_size, num_workers=self.cpus, collate_fn=collate_sequences)

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("corpus_dir", type=str, help="File path to the directory containing the corpus to collect chars from.")
    arg_parser.add_argument("--cuda", type=int, default=None, help="Index of the CUDA device (GPU) to use.")
    arg_parser.add_argument("--cpu-limit", type=int, default=None, help="Max number of CPU processors to use.")
    arg_parser.add_argument("--try-long", action="store_true", help="Try running a long sequence through the model")
    args = arg_parser.parse_args()
    corpus_dir = args.corpus_dir
    cuda_index = args.cuda
    cpu_limit_ = args.cpu_limit
    try_long = args.try_long

    if cuda_index is None:
        device_ = torch.device("cpu")
    elif cuda_index >= (cuda_count := torch.cuda.device_count()) or cuda_index < 0:
        print(f"WARNING: provided cuda index '{cuda_index}' is not valid (available count = {cuda_count}); defaulting to CPU", file=sys.stderr)
        device_ = torch.device("cpu")
    else:
        device_ = torch.device(f"cuda:{cuda_index}")

    if cpu_limit_ is None:  # use all we've got
        cpus_ = os.cpu_count()
    else:
        cpus_ = min(max(cpu_limit_, 1), os.cpu_count())  # clip the provided number between 1 and os.cpu_count(), inclusive
    if cpus_ == 1:
        cpus_ = 0  # DataLoader expects 0 if we're not doing extra workers

    model = NeuralCorrector(corpus_dir, cpus_)
    model.to(device_)
    model.eval()

    print("using untrained model as plain corrector...")
    print("-------")
    test_in = [
        "This is a thing. " * 32,
        "Super short sentence, nothing else.",
        "This one is going to be crazy long. " * 40
    ]
    test_out = model.correct(test_in)
    for in_str, out_str in zip(test_in, test_out):
        print(f"INPUT:  {in_str}")
        print(f"OUTPUT: {out_str}")
        print("-------")

    dataset = CorrectorDataset(corpus_dir, split="validation", tensors_out=True)
    dataloader = DataLoader(dataset, batch_size=3, num_workers=cpus_, collate_fn=collate_sequences)

    with torch.no_grad():
        for batch_ in dataloader:
            if try_long and batch_[0].shape[0] <= model.max_len and batch_[1].shape[0] < model.max_len:
                continue  # find a sequence that's too long to make sure we don't get an error
            batch_ = (batch_[0].to(device_), batch_[1].to(device_))
            print(f"input tensor shape/type: {batch_[0].shape}/{batch_[0].dtype}")
            print(f"output tensor shape/type: {batch_[1].shape}/{batch_[1].dtype}")
            # try both functions
            print("starting to run raw tensors through the model...")
            loss_ = model.training_step(batch_, 0)
            print(f"{loss_=}")
            output = model(batch_[0])
            print(f"{output=}")
            # interpret output as a batch of strings
            print("\nGenerated outputs from untrained model:")
            texts_out = model.tensor_to_texts(output)
            for i_, s in enumerate(texts_out):
                print(f"\nOutput sequence {i_}: {s}")
                print(f"(Length: {len(s)})")
            break
