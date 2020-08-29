from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Ch3_DL.data import StackoverflowNGramDataset
from Ch3_DL.model_pytorch import FFLanguageModel
import logging as log

log.basicConfig(level=log.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
BATCH_SIZE = 32
LEARNING_RATE = 1e-2


def train(model: nn.Module, dataset: Dataset, n_epochs: int):
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.NLLLoss()
    log.info(f'Starting training for {n_epochs} epochs')
    for epoch in range(n_epochs):
        for context_batch, target_batch in dataloader:
            pred_batch = model(context_batch)
            loss = loss_fn(pred_batch, target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        log.info(f'Epoch: {epoch}, Loss: {loss.detach().numpy():.4f}')
    return model


if __name__ == '__main__':
    dataset = StackoverflowNGramDataset(
        stackoverflow_categories=['title'],
        min_token_frequency=3,
        min_sentence_length=4,
        max_sentence_length=100,
        to_wordindices=True,
        to_tensor=True
    )
    model = FFLanguageModel(vocab_size=len(dataset.vocab), embedding_dim=100, context_size=2)
    train(model, dataset, n_epochs=10)