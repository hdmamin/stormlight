from nltk.tokenize import PunktSentenceTokenizer
from multiprocessing import Pool
import pandas as pd
from textblob import Blobber
from textblob.en.sentiments import PatternAnalyzer, NaiveBayesAnalyzer
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from htools import LoggerMixin, valuecheck
from stormlight.utils import *


class SentimentScorer(LoggerMixin):

    @valuecheck
    def __init__(self, chunk: ('page', 'sentence'), verbose=True):
        self.chunk = chunk
        self.verbose = verbose
        self.logger = self.get_logger()
        self.scorers = {'vader': SentimentIntensityAnalyzer(),
                        'pattern': PatternAnalyzer(),
                        'huggingface': pipeline('sentiment-analysis'),
                        'naive': Blobber(analyzer=NaiveBayesAnalyzer())}

    def _chunk_text(self, text):
        if self.chunk == 'sentence':
            return PunktSentenceTokenizer().tokenize(text.replace(BREAK, ''))
        return text.split(BREAK)

    def score(self, text,
              scorer: ('vader', 'pattern', 'huggingface', 'naive')):
        """
        Returns
        -------
        list[dict[str, float]]
        """
        if self.verbose: self.logger.info(f'Starting {scorer} scoring...')
        return sentiment_score(lines=self._chunk_text(text),
                               sent_fn=getattr(self, f'_score_{scorer}'),
                               multi=(scorer != 'huggingface'))

    def multi_score(self, text):
        """
        Returns
        -------
        dict[str, list[dict[str, float]]]: Maps scorer name to list of
            sentiment score dicts for each chunk of the input text
            (chunks are usually sentences or pages, depending on the choice
            of `sep` in the constructor).
        """
        return {name: self.score(text, name) for name in self.scorers}

    def score_many(self, texts,
                   scorer: ('vader', 'pattern', 'huggingface', 'naive')):
        """
        Returns
        -------
        dict[str, list[dict[str, float]]]: Maps each text's name to a list of
            dicts containing sentiment scores from a single scorer.
        """
        return {k: self.score(v, scorer) for k, v in texts.items()}

    def multi_score_many(self, texts):
        """
        Returns
        -------
        dict[str, pd.DataFrame]: Maps each text's name to a dataframe of
            scores. Each dataframe contains 1 or more columns for eacher
            scorer (many scorers return multiple scores per piece of text,
            e.g. POS and NEG).
        """
        res = {k: self.multi_score(v) for k, v in texts.items()}
        return {k: pd.concat([pd.DataFrame(v2) for v2 in v.values()],
                             keys=v.keys(), axis=1)
                for k, v in res.items()}

    def _score_vader(self, text):
        res = self.scorers['vader'].polarity_scores(text)
        res['score'] = res.pop('compound')
        return res

    def _score_pattern(self, text):
        res = self.scorers['pattern'].analyze(text)._asdict()
        res['score'] = res.pop('polarity')
        return res

    def _score_huggingface(self, text):
        scorer = self.scorers['huggingface']
        try:
            res = scorer(text)[0]
        except IndexError:
            tok = scorer.tokenizer
            tokens = tok.encode(text)[:tok.model_max_length - 1]
            text_trunc = tok.decode(tokens).replace(tok.cls_token + ' ', '')
            res = scorer(text_trunc)[0]
        if res['label'] == 'NEGATIVE': res['score'] *= -1
        return res

    def _score_naive(self, text):
        res = self.scorers['naive'](text).sentiment._asdict()
        res['score'] = res['p_pos'] if res['classification'] == 'pos' \
            else res['p_neg'] * -1
        return res


def sentiment_score(lines, sent_fn, multi=True):
    if not multi:
        return list(map(sent_fn, lines))
    with Pool() as p:
        return p.map(sent_fn, lines)


def sentiment_scores(books, sent_fn, preprocess_fn, multi=True):
    return {k: sentiment_score(preprocess_fn(v), sent_fn, multi)
            for k, v in books.items()}


def split_text(text, sep, attach=False):
    chunks = [chunk for chunk in text.split(sep) if chunk]
    if attach:
        chunks = [chunk + sep for chunk in chunks]
    return chunks
