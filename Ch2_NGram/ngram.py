"""
Ch2 - N-gram language model

1) Split the dataset into a training and a testing subset.
    Use the category “title” for the testing set and the categories “comment” and “post” for the training set.
    The short length of titles will make them good candidates later as seeds for text generation [done]
2) Build the matrix of prefix—word frequencies [done]
3) Write a text generation function [done]
4) Write a function that can estimate the probability of a sentence and use it to select the
    most probable sentence out of several candidate sentences
5) Implement the perplexity scoring function for a given sentence and for the training corpus
6) Implement Additive Laplace smoothing to give a non-zero probability to missing prefix—token
    combinations when calculating perplexity
7) Calculate the perplexity of the language model on the test set composed of titles
8) Try to improve the perplexity score of your model
"""
import numpy as np
import pandas as pd
from nltk.util import ngrams
from collections import defaultdict, Counter
import Ch1_PreparingDataset.data as ch1data
from typing import List
import logging as log

log.basicConfig(level=log.DEBUG)


def split_train_test(data):
    train = data.loc[data.category.isin(["comment", "post"])].reset_index(drop=True, inplace=False)
    test = data.loc[data.category.isin(["title"])].reset_index(drop=True, inplace=False)
    assert len(train) > 0
    assert len(test) > 0
    return train, test


class NGramModel(object):
    def __init__(self, n_gram : int = 3):
        self.n_gram = n_gram
        self.left_pad_token = '<s>'
        self.right_pad_token = '</s>'
        self.ngram_counts = None
        self.num_unique_tokens = None

    def init_prefix_word_counts(self, data : pd.DataFrame):
        """
        Parameters
        ----------
        data : pandas.DataFrame
            Must contain column `tokens` which is the tokenized version of the `text` column.
            DataFrame can be provided by the `main` method in module `Ch1_PreparingDataset.data`
        """
        counts = defaultdict(Counter)
        # create dictionary of prefix -> word frequencies (which are stored in a Counter object)
        for tokens in data.tokens:
            grams = ngrams(tokens, n=self.n_gram,
                           left_pad_symbol=self.left_pad_token, right_pad_symbol=self.right_pad_token)
            for gram in grams:
                # store prefix in dict and update the count of the word
                # example: gram = ('importance', 'of', 'language') -> prefix = ('importance', 'of'), word = 'language'
                counts[gram[:-1]].update(Counter([gram[-1]]))
        self.ngram_counts = counts
        tokens = [list(c.keys()) for c in counts.values()]
        self.tokens = set([t for tt in tokens for t in tt])
        self.num_unique_tokens = len(self.tokens)
        assert self.num_unique_tokens > 0
        return counts

    def prob_word_given_prefix(self, word : str, prefix : tuple, laplace_smoothing : float = 1e-5) -> float:
        """
        Parameters
        ----------
        word : str
            Token for which conditional probability is computed
        prefix : tuple
            History of `word` we condition on
        laplace_smoothing : float
            Add-k parameter for smoothing of MLE for conditional probability of word given prefix
        """
        assert self.ngram_counts is not None, "Initialize self.ngram_counts using init_prefix_word_counts(data)"
        assert laplace_smoothing > 0, "Laplace smoothing parameter must be positive (else division by zero if prefix or word are OOV)"
        counts = self.ngram_counts
        num_unique_tokens = self.num_unique_tokens

        if prefix in counts.keys():
            count_prefix = sum(counts[prefix].values())
        else:
            count_prefix = 0
        if word in counts[prefix].keys():
            count_prefix_word = counts[prefix][word]
        else:
            count_prefix_word = 0
        return (count_prefix_word + laplace_smoothing) / (count_prefix + laplace_smoothing * num_unique_tokens)

    @staticmethod
    def apply_temperature_scaling(p, tau):
        assert abs(sum(p) - 1) < 1e-6
        assert tau > 0
        return (p ** (1 / tau)) / sum(p ** (1 / tau))

    def generate_text(self, input_ngram : tuple, length : int = 20,
                      laplace_smoothing : float = 1e-5, temperature_sampling : float = 1) -> List[str]:
        """
        Parameters
        ----------
        input_ngram : List[str]
            Starting n-gram from from which first new word is generated. Length of n-gram has to match
            `self.ngram`
        length : int
            Maximum size of generated text (token count)
        laplace_smoothing : float
            Add-k parameter for smoothing of MLE for conditional probability of word given prefix
        temperature_sampling : float
            If 1, original word-prefix probabilities are used
            If param goes to zero, more frequest words are boosted
            If param goes to infinity, all word-prefix probabilities are equal
        """
        assert self.ngram_counts is not None, "Initialize self.ngram_counts using init_prefix_word_counts(data)"
        all_tokens = list(self.tokens)
        curr_ngram = input_ngram
        generated_text = [*curr_ngram]
        stop_token = self.right_pad_token
        while (len(generated_text) < length) or (generated_text[-1] == stop_token):
            # sample next token
            next_token = np.random.choice(a=all_tokens,
                                          p=self.apply_temperature_scaling(
                                              p=np.array([
                                                  self.prob_word_given_prefix(
                                                    word=w,
                                                    prefix=curr_ngram,
                                                    laplace_smoothing=laplace_smoothing) for w in all_tokens]),
                                              tau=temperature_sampling))
            generated_text.append(next_token)
            curr_ngram = tuple(generated_text[-self.n_gram:])
        return generated_text

    def sentence_probability(self, sentence_tokens : List[str]) -> float:
        """
        Parameters
        ----------
        sentence_tokens : List[str]
            List of tokens in sentence
        """
        assert len(sentence_tokens) > self.n_gram, "Sentence must contain at least " + str(self.n_gram) + " words"
        sen_prob = 1
        for i, w in enumerate(sentence_tokens[self.n_gram:]):
            sen_prob *= self.prob_word_given_prefix(word=w, prefix=tuple(sentence_tokens[i-self.n_gram+1: i-1]))
        return sen_prob

    def log_sentence_perplexity(self, sentence_tokens : List[str]):
        """
        Parameters
        ----------
        sentence_tokens : List[str]
            List of tokens in sentence
        """
        len_sentence = len(sentence_tokens)
        assert len_sentence >= self.n_gram, f"Sentence {sentence_tokens} is too short to compute perplexity."
        perplexity = 0
        for i, w in enumerate(sentence_tokens[self.n_gram:]):
            perplexity -= np.log(self.prob_word_given_prefix(word=w, prefix=tuple(sentence_tokens[i-self.n_gram+1: i-1])))
        return perplexity / len_sentence

    def log_corpus_perplexity(self, corpus : List[List[str]]) -> float:
        """
        Parameters
        ----------
        corpus : List[List[str]]
            List of tokenized sentences in corpus
        """
        perplexity = 0
        for sentence in corpus:
            perplexity += self.log_sentence_perplexity(sentence)
        return perplexity


if __name__ == '__main__':
    model = NGramModel(n_gram=3)
    data = ch1data.main(debug=False, keep_only_english_words=True)
    train, test = split_train_test(data)
    model.init_prefix_word_counts(data=train)
    text = model.generate_text(input_ngram=("if","it"), laplace_smoothing=1e-10)
    log.info(f"Generated text {text}")
    log.info(f"Sentence probability: {model.sentence_probability(text)}")
    log.info(f"Sentence perplexity: {model.log_sentence_perplexity(text)}")
    log.info(f"Training corpus perplexity (N={len(train)}): {model.log_corpus_perplexity(train.tokens.values)}")
    log.info(f"Test corpus perplexity (N={len(test)}): {model.log_corpus_perplexity(test.tokens.values)}")