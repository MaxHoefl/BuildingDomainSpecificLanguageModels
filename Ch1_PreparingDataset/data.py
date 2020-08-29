"""
Chapter 1 - Building domain specific language models

Workflow
1) Load the dataset into a pandas dataframe.
2) Use regular expressions to remove elements that are not words such as HTML tags, LaTeX expressions, URLs, digits, line returns, and so on.
3) Remove missing values for texts
4) Remove texts that are extremely large or too short to bring any information to the model. We want to keep paragraphs that contain at least a few words and remove the paragraphs that are composed of large numerical tables.
5) Use a tokenizer to create a version of the original text that is a string of space-separated lowercase tokens
6) Note that punctuation signs (, . : !) are also represented as tokens.
7) Export the resulting dataframe into a csv file
"""

import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import words
import urllib
import logging as log
import pickle

CURRDIR = os.path.dirname(__file__)
REPODIR = os.path.dirname(CURRDIR)
DATADIR = os.path.join(REPODIR, "data")
DLLINK = "https://liveproject-resources.s3.amazonaws.com/116/other/stackexchange_812k.csv.gz"
DATAFILE = "stackexchange_812k.csv.gz"
CSVFILE_OUTPUT = "stackexchange_812k_preprocessed.csv"
PICKLEFILE_OUTPUT = "stackexchange_812k_preprocessed.pkl"
if not os.path.exists(DATADIR):
    os.makedirs(DATADIR)

log.basicConfig(level=log.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


def download_dataset():
    dl = urllib.URLopener()
    dl.retrieve(DLLINK, os.path.join(DATADIR, DATAFILE))


def clean(x):
    x = re.sub("&amp;", "&", x)
    x = re.sub("&lt;", "<", x)
    x = re.sub("&gt;", ">", x)
    x = re.sub("\$.+\$", "", x) # latex
    x = re.sub("</?(ul|li|p|a|strong|img|em)(\\s\\S+)?>", " ", x) # html
    x = re.sub("\n", "", x) # line break
    x = re.sub("\t", " ", x) # tab
    x = re.sub("@", " ", x) # @
    x = re.sub("http://\\S+\\s", " ", x) # website links
    x = re.sub("\"", " ", x)
    x = re.sub("\*+", " ", x)
    x = re.sub("<\\\\p>", " ", x)
    x = ''.join([x[i] for i in range(len(x)) if ord(x[i]) < 128]) # word_tokenize only allows ascii ordinals < 128
    return x


def load_preprocessed_data(load_from: str = 'pickle'):
    assert load_from in ['csv', 'pickle']
    if load_from == 'csv':
        data = pd.read_csv(os.path.join(DATADIR, CSVFILE_OUTPUT))
        data = data.loc[~pd.isna(data.tokens)]
        # recover tokens lists from string in csv
        data["tokens"] = data.tokens.apply(lambda x: x[2:-2].split("', '"))
    else:
        with open(os.path.join(DATADIR, PICKLEFILE_OUTPUT), 'rb') as f:
            data = pickle.load(f)
    assert isinstance(data, pd.DataFrame)
    assert "tokens" in data.columns, "Invalid datafile. Missing column 'tokens'. Re-generated file using main()"
    return data


def main(debug: bool = False, keep_only_english_words: bool = False):
    """
    Preprocesses statexchange data
    """
    if os.path.exists(os.path.join(DATADIR, CSVFILE_OUTPUT)):
        if os.path.isfile(os.path.join(DATADIR, CSVFILE_OUTPUT)):
            return load_preprocessed_data()

    if not os.path.exists(os.path.join(DATADIR, DATAFILE)):
        log.info("Downloading dataset")
        download_dataset()

    data = pd.read_csv(os.path.join(DATADIR, DATAFILE), compression="gzip")
    if debug:
        data = data.sample(frac=1).reset_index(drop=True).iloc[:1000]

    # Minimum and maximum number of chars allowed in text
    min_char_len = np.percentile(data.text.str.len(), q=1)
    max_char_len = np.percentile(data.text.str.len(), q=99)
    # Clean text
    log.info("Cleaning dataset")
    data["text"] = data.text.apply(lambda x: clean(x))
    # Remove NaN text, NaN category, empty text, text that is too short or too lengthy
    data = data.loc[(~pd.isnull(data.text)) & (~pd.isnull(data.category)) &
                    (data.text.str.len() > min_char_len) & (data.text.str.len() < max_char_len)]
    # Tokenize text
    log.info("Tokenizing dataset")
    data["tokens"] = data.text.apply(lambda x: word_tokenize(x.lower()))
    data = data.loc[~pd.isna(data.tokens)]
    # If flat is set only keep english word tokens
    if keep_only_english_words:
        nltk.download('words')
        english_words = set(words.words())
        data["tokens"] = data.tokens.apply(lambda t: [w for w in t if (w in english_words) and w.isalpha()])
    # Pickle file
    with open(os.path.join(DATADIR, PICKLEFILE_OUTPUT), 'wb') as f:
        pickle.dump(data, f)
    # Submission file
    data.to_csv(os.path.join(DATADIR, CSVFILE_OUTPUT), index=False, header=True)
    return data


if __name__ == '__main__':
    main(debug=False, keep_only_english_words=False)
