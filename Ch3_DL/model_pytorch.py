import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from Ch1_PreparingDataset import data
from Ch3_DL.data import StackoverflowNGramDataset


BATCH_SIZE = 32


class FFLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(FFLanguageModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds = embeds.view((embeds.shape[0], -1))
        out = F.relu(self.linear1(embeds))
        out = F.relu(self.linear2(out))
        log_probs = F.log_softmax(out, dim=1)
        return log_probs



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
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    batch = dataset[:BATCH_SIZE]
    print(model(x))
