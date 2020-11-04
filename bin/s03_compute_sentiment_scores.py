import fire
import os
from pathlib import Path

from stormlight.analysis import *
from stormlight.processing import *


def generate(stormlight=True, novellas=True, mistborn=False,
             out_dir='data/interim'):
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    books = STORMLIGHT*stormlight + NOVELLAS*novellas + MISTBORN*mistborn
    books = load_books(*books)

    # Compute scores (slow, though Huggingface will be faster on GPU)
    scorer = SentimentScorer('sentence')
    book2df = scorer.multi_score_many(books)
    for book, df in book2df.items():
        df.to_csv(out_dir/f'{book}_sentiment_scores.csv', index=False)


if __name__ == '__main__':
    fire.Fire(generate)

