import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.optim import Optimizer

from model.positional_encoding import PositionalEncoding


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
        batch_size = x.shape[1]  # 0 = sequence, 1 = batch
        x = self.positional_encoding(self.embedding_src(x))
        x = self.transformer.encoder(x)  # TODO: padding mask
        sequence = torch.zeros((1, batch_size), dtype=torch.long)
        sequence[0, :] = self.bookend_index
        while True:
            sequence_embed = self.positional_encoding(self.embedding_tgt(sequence))
            new_thing = self.transformer.decoder(sequence_embed, x)[-1, :]  # TODO: masks
            new_thing = self.linear_stack(new_thing)
            # TODO: softmax and such
            # TODO: append to sequence
            break  # TODO: until the decoder generates a 'bookend' token (on all batches)
        return sequence

    def training_step(self, batch, batch_idx) -> Tensor:
        x, y = batch
        x = self.positional_encoding(self.embedding_src(x))
        # TODO: add a bookend token to the start of y
        y_embed = self.positional_encoding(self.embedding_tgt(y))  # TODO: put it in here, not y
        y_mask = nn.Transformer.generate_square_subsequent_mask(y.shape[0])
        y_hat = self.transformer(x, y_embed, tgt_mask=y_mask)  # TODO: padding masks?
        y_hat = self.linear_stack(y_hat)
        # TODO: add a bookend token to the end of y
        loss = self.criterion(y_hat, y)  # TODO: put it in here, not y
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=4e-3)
        return optimizer
