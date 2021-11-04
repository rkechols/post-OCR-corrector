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

    def forward(self, x: Tensor) -> Tensor:
        # TODO: max sequence length?
        batch_size = x.shape[1]  # 0 = sequence, 1 = batch
        # get padding mask
        x_padding_mask = torch.where(x == -1, True, False)
        # convert any -1 to the actual padding index
        x[x_padding_mask] = self.pad_index
        x = self.positional_encoding(self.embedding_src(x.detach()))  # detach is so we don't need to back-prop through the data prep
        x = self.transformer.encoder(x, src_key_padding_mask=x_padding_mask)
        # make a sequence to go in the decoder, gradually growing
        sequence = torch.full((1, batch_size), self.bookend_index, dtype=torch.long)
        terminated = torch.zeros(batch_size).bool()  # keep track of which sequences have finished
        while True:
            sequence_padding_mask = 1  # TODO
            sequence_embed = self.positional_encoding(self.embedding_tgt(sequence))
            new_thing = self.transformer.decoder(sequence_embed, x, tgt_key_padding_mask=sequence_padding_mask)[-1, :]  # only take the last token
            new_thing = self.linear_stack(new_thing)
            new_thing = torch.argmax(new_thing, dim=-1)
            terminated = terminated or (new_thing == self.bookend_index)
            if terminated.all():
                break  # all pieces of the batch are done
            sequence = torch.cat([sequence, new_thing.unsqueeze(0)], dim=0)
        sequence = torch.where(sequence == self.bookend_index, self.pad_index, sequence)
        # trim padding from the end (if the last thing in all sequences is padding, chop off the las thing from each sequence)
        while sequence.shape[0] > 0 and torch.all(sequence[-1, :] == self.pad_index):
            sequence = sequence[:-1, :]
        # convert any padding to -1
        sequence = torch.where(sequence == self.pad_index, -1, sequence)
        return sequence

    def training_step(self, batch, batch_idx) -> Tensor:
        # TODO: max sequence length?
        x, y = batch  # 0 = sequence, 1 = batch
        batch_size = x.shape[1]
        # get padding masks
        x_padding_mask = torch.where(x == -1, True, False)
        y_padding_mask = torch.where(y == -1, True, False)
        # convert any -1 to the actual padding index
        x[x_padding_mask] = self.pad_index
        y[y_padding_mask] = self.pad_index
        # stick a "start token" at the beginning of the target sequence
        bookend_tensor = torch.full((1, batch_size), self.bookend_index)
        y_in = torch.cat([bookend_tensor, y], dim=0)
        x_padding_mask = torch.permute(x_padding_mask, (1, 0))  # also permute since that's what torch wants
        y_padding_mask = torch.permute(torch.cat([torch.zeros((1, batch_size)).bool(), y_padding_mask], dim=0), (1, 0))
        # turn indices into embeddings
        x = self.positional_encoding(self.embedding_src(x.detach()))  # detach is so we don't need to back-prop through the data prep
        y_in = self.positional_encoding(self.embedding_tgt(y_in.detach()))
        # make a mask so it can't cheat when learning how to sequentially generate
        y_mask = nn.Transformer.generate_square_subsequent_mask(y_in.shape[0])
        # pass the sequences through the main part of the model
        y_hat = self.transformer(x, y_in, tgt_mask=y_mask, src_key_padding_mask=x_padding_mask, tgt_key_padding_mask=y_padding_mask)
        y_hat = self.linear_stack(y_hat)
        # add a bookend token to the end of each sequence in y, so the loss is teaching it to end with a bookend
        y_target = torch.cat([y, torch.full((1, batch_size), self.pad_index)], dim=0)
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
    dataset = CorrectorDataset("data/corpus/srWaC", split="validation", tensors_out=True)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=collate_sequences)
    model = NeuralCorrector(dataset.alphabet_size)
    for batch_ in dataloader:
        output = model(batch_[0])
        loss_ = model.training_step(batch_, 0)
        break
