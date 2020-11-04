import numpy as np
from pathlib import Path

from htools import DotDict


BREAK = 'xxbrxx'
DATA_DIR = Path(__file__).parents[2]/'data'
CLEAN_DIR = DATA_DIR/'clean'
PATHS = DotDict(edge=CLEAN_DIR/'edgedancer.txt',
                war=CLEAN_DIR/'warbreaker.txt',
                mist=CLEAN_DIR/'mistborn_the_final_empire.txt',
                kings=CLEAN_DIR/'the_way_of_kings.txt',
                words=CLEAN_DIR/'words_of_radiance.txt',
                oath=CLEAN_DIR/'oathbringer.txt')
BOOK_IDX = {'edge': (965, 288_719),
            'war': (5_615, 1_121_702),
            'kings': (8_506, 2_209_305),
            'words': (10_080, 2_295_302),
            'oath': (9_275, 2_610_217),
            'mist': (3_133, 1_225_987)}
STORMLIGHT = ['kings', 'words', 'oath']
NOVELLAS = ['edge', 'war']
MISTBORN = ['mist']
PRE2FULL = dict(kings='The Way of Kings',
                words='Words of Radiance',
                oath='Oathbringer',
                edge='Edgedancer',
                war='Warbreaker',
                mist='Mistborn')


def is_ascii(tok):
    """Check if all characters in a string are ascii.

    Parameters
    ----------
    tok: str
        A token with one or more characters.

    Returns
    -------
    bool: True if all characters are ascii.
    """
    return all(ord(char) < 128 for char in tok)


def pct_ascii(tokens):
    """Compute percent of tokens that are ascii.

    Parameters
    ----------
    tokens: list[str]

    Returns
    -------
    float: Value between 0 and 1.
    """
    return np.mean([is_ascii(t) for t in tokens])

