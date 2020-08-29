from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from Ch1_PreparingDataset.data import load_preprocessed_data
from Ch2_NGram.ngram import NGramModel
import logging as log

log.basicConfig(level=log.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


class StackoverflowNGramDataset(Dataset):
    """
    A dataset for language modeling on the Stackoverflow dataset.
    Returns tuples of the form (tuple of context words, target word).
    """
    def __init__(self, context_size: int = 2,
                 stackoverflow_categories: List[str] = None,
                 min_token_frequency: int = None,
                 min_sentence_length: int = None,
                 max_sentence_length: int = None,
                 to_wordindices: bool = True,
                 to_tensor: bool = True):
        super(StackoverflowNGramDataset, self).__init__()
        self.context_size = context_size
        self._to_wordindices = to_wordindices
        self._to_tensor = to_tensor

        log.info('Loading preprocessed data')
        data = load_preprocessed_data(load_from='pickle')
        log.info('\tDone')

        if stackoverflow_categories is not None:
            data = data.loc[data.category.isin(stackoverflow_categories)].reset_index(drop=True)
        if min_sentence_length is not None:
            data = data.loc[data.tokens.str.len() >= np.maximum(min_sentence_length, context_size + 1)]
        else:
            data = data.loc[data.tokens.str.len() >= context_size + 1]
        if max_sentence_length is not None:
            data = data.loc[data.tokens.str.len() <= np.maximum(max_sentence_length, context_size + 1)]

        log.info('Creating NGram dataset')
        ngram_model = NGramModel(n_gram=context_size + 1)
        ngram_model.init_prefix_word_counts(data=data)
        if min_token_frequency:
            dataset = [(context, target) for context, targets in ngram_model.ngram_counts.items()
                       for target, count in targets.items() if count >= min_token_frequency]
        else:
            dataset = [(context, target) for context, targets in ngram_model.ngram_counts.items()
                       for target, count in targets.items()]
        log.info('\tDone')
        self.dataset = dataset

        log.info('Creating vocabulary')
        vocab = []
        for prefix, target in self.dataset:
            vocab.extend([w for w in prefix])
            vocab.append(target)
        self.vocab = set(vocab)
        self.vocab_index = {w : i for i, w in enumerate(self.vocab)}
        log.info('\tDone')

        transforms = []
        if to_wordindices:
            transforms.append(self.to_wordindices)
        if to_tensor:
            transforms.append(self.to_tensor)
        if len(transforms) > 0:
            log.info('Applying transforms')
            for i, entry in enumerate(self.dataset):
                for transform in transforms:
                    entry = transform(entry)
                self.dataset[i] = entry
            log.info('\tDone')

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)

    def to_wordindices(self, entry):
        context, target = entry
        context_indices = tuple([self.vocab_index[w] for w in context])
        target_index = self.vocab_index[target]
        return (context_indices, target_index)

    def to_tensor(self, entry):
        return (torch.tensor(entry[0], requires_grad=False), torch.tensor(entry[-1], requires_grad=False))


if __name__ == '__main__':
    dataset = StackoverflowNGramDataset(
        stackoverflow_categories=['title'],
        min_token_frequency=3,
        min_sentence_length=4,
        max_sentence_length=100,
        to_wordindices=True,
        to_tensor=True
    )
    print(f'Dataset size: {len(dataset)}')
    print(f'Vocab size: {len(dataset.vocab)}')
    print(f'Vocabulary: {list(dataset.vocab)[:50]}...')
    # print(f'Vocab index \'what\': {dataset.vocab_index["what"]}')
    # print(f'Vocab index \'are\': {dataset.vocab_index["are"]}')
    # print(f'Vocab index \'some\': {dataset.vocab_index["some"]}')
    print(f'Dataset 0: {dataset[0]}')
    print(f'Dataset 0:3: {dataset[:3]}')