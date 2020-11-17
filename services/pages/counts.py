import plotly.graph_objs as go
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
import re
import streamlit as st
import string

from htools import DotDict
from stormlight.processing import load_books
from stormlight.utils import STORMLIGHT, NOVELLAS, PRE2FULL

from .utils import widget


@st.cache(hash_funcs={DotDict: id})
def cached_load():
    """Wrapper to load books only once.

    Returns
    -------
    DotDict[str, str]: Maps book name prefix to book full text.
    """
    return load_books(*STORMLIGHT+NOVELLAS, replace_break='')


@st.cache(hash_funcs={list: id})
def cached_count_words(books):
    """Find total number of words in each book. Cache result so streamlit
    doesn't recompute it.

    Parameters
    ----------
    books: DotDict
        Maps book name prefix to full text.

    Returns
    -------
    list[int]: List of word counts in same order as books.
    """
    tok = WordPunctTokenizer()
    return [len([t for t in tok.tokenize(v)
                 if t not in set(string.punctuation)])
            for v in books.values()]


def count(text, term, cased=False, strict=False):
    """Find the number of occurences of a search term in a piece of text.

    Parameters
    ----------
    text: str
        Text from a book.
    term: str
        User input search term.
    cased: bool
        If True, counts are case sensitive.
    strict: bool
        If True, we require that the given search term must be preceded and
        followed by a non-word character. For example, we might want to match
        the word "blush" but not "blushes" or "blushing".

    Returns
    -------
    int: Number of occurrences of `term` in `text`.

    """
    if not cased:
        text, term = text.lower(), term.lower()
    if strict:
        return len(re.findall('\W(' + re.escape(term) + ')\W', text))
    return text.count(term)


def build(locals_):
    try:
        books = cached_load()
    except FileNotFoundError:
        st.write('This functionality is unavailable on the public app.')
        st.stop()

    counts = cached_count_words(books)
    st.title('Word Counts')
    st.markdown('Enter a word or phrase in the text box below to find the '
                'number of times it appears in each book.')
    term = st.text_input('Search term:')
    cased = widget(
        'checkbox', 'Case Sensitive',
        'When unchecked, upper and lowercase characters will be treated as '
        'equivalent.',
        key='a', sidebar=False, value=True
    )
    strict = widget(
        'checkbox', 'Strict Boundaries',
        'If unchecked, all occurrences will count, even if the term is '
        'contained within another string. E.g. “storm” would match “storms”, '
        '“stormlight”, “stormed”, “storming”, etc. When checked, it would '
        'only match “storm” itself.',
        key='b', sidebar=False
    )

    if term:
        data = [count(book, term, cased, strict) for book in books.values()]
    else:
        data = [0] * len(books)
    df = pd.DataFrame([data, counts],
                      columns=[PRE2FULL[name] for name in books],
                      index=['Term Count', 'Total Word Count (Approximate)'])
    st.table(df)

    if term:
        df = df.T
        fig = go.Figure(data=go.Bar(x=df.index, y=df['Term Count']))
        fig.update_layout(title=f'"{term}" # of Occurrences',
                          xaxis_title='Book',
                          yaxis_title='Count',
                          margin=dict(l=0, r=0, b=25, t=25, pad=0)
                          )
        st.plotly_chart(fig)


