import numpy as np
from torch.utils.data import Dataset
import warnings

from htools import select, save, load
from stormlight.processing import load_books, load_endnotes
from stormlight.utils import PATHS, BREAK


class LMDataset(Dataset):

    def __init__(self, books=tuple(PATHS.keys()), tok=None, seq_len=512,
                 tokens=None, subset_frac=1, return_tuple=False):
        """
        Parameters
        ----------
        books: Iterable[str]
            Abbreviated names of the books to load.
        tok: transformers Tokenizer
            Usually GPT2TokenizerFast. Used to tokenize and encode text.
        seq_len: int
            The number of tokens to include in each sequence. Saw conflicting
            sources but it sounds like 512 is a pretty standard max for GPT2.
            This also happens to be quite close to the average page length for
            these books.
        tokens: np.array[str] or None
            User will rarely (never?) pass in their own tokens. This is
            provided to allow loading a pickled set of tokens (see
            `from_pickle`).
        subset_frac: int or float
            This is the percent of each book in `books` that will be loaded.
            Usually this is 1, but when testing on a subset we can pass in
            a smaller number (e.g. 0.1) to quickly create a dataset with pieces
            of each book without waiting a long time for tokenization.
        return_tuple: bool
            Allows compatibility with incendio. If True, indexing into the
            dataset returns a tuple of length 1 containing an array of tokens
            rather than simply returning the array.
        """
        if not tok and tokens is None:
            raise ValueError('Must pass in either tokenizer or tokens.')
        self.books = books
        self.tok = tok
        self.seq_len = seq_len
        self.subset_frac = subset_frac
        self.tokens = self._create_tokens() if tokens is None else tokens
        self.return_tuple = return_tuple

    def _create_tokens(self):
        """Tokenize and numericalize text for the books specified in the
        constructor. Books are separated by page break characters. All
        sequences are the same length and no padding is used (the last item is
        filled out with the last book's endnotes).

        Returns
        -------
        np.array[int]: Flat array of word indices.
        """
        books = load_books(*self.books)
        text = BREAK.join(book[:int(len(book) * self.subset_frac)]
                          for book in books.values())
        tokens = self.tok.encode(text)

        # Instead of padding/dropping last item, load last book's endnotes.
        n_missing = self.seq_len - len(tokens) % self.seq_len
        if n_missing > 0:
            endnotes = load_endnotes(self.books[-1])
            tokens += self.tok.encode(endnotes)[:n_missing]
        return np.array(tokens)

    def __getitem__(self, i):
        """Get an array of `self.seq_len` tokens. Note that the tokens are
        stored as a single flat array. If the DataLoader uses shuffle=True,
        it will construct batches using different combinations of indices, but
        each item in the batch will still be an ordered sequence (as we want).

        Parameters
        ----------
        i: int
            Index in [0, len(self)).

        Returns
        -------
        np.array[i]: Word indices of length `self.seq_len`.
        """
        seq = self.tokens[i * self.seq_len:(i + 1) * self.seq_len]
        return (seq, seq) if self.return_tuple else seq

    def __len__(self):
        return int(np.ceil(len(self.tokens) / self.seq_len))

    def save(self, path):
        """Save tokens and other dataset info.

        Parameters
        ----------
        path: str or Path
            File to save to (doesn't need to exist yet).

        Returns
        -------
        None
        """
        data = select(vars(self), drop=['tok'])
        data['tok_type'] = type(self.tok)
        save(data, path)

    @classmethod
    def from_pickle(cls, path, tok=None, **kwargs):
        """Load a LMDataset whose tokens were previously pickled. Huggingface
        tokenizers can't be pickled so we simply save the type and warn the
        user if the tokenizer they provide is different. The tokenizer can also
        be left as None since the corpus has already been tokenized - we just
        won't have the option of calling `ds.tok.decode(indices)`, but it may
        be best to keep that behavior separate anyway.

        Parameters
        ----------
        path: str or Path
            File containing pickled tokens.
        tok: transformers Tokenizer or None
            Intended to be used with GPT2TokenizerFast, but most others
            probably work. If you don't care about attaching a tokenizer to
            your dataset since it's already tokenized, you can leave this as
            None.
        return_tuple: bool
            Allow compatibility with Incendio. See `__init__` method.

        Returns
        -------
        LMDataset
        """
        data = load(path)
        data.update(kwargs)
        if type(tok) != data.pop('tok_type'):
            warnings.warn('Tokenizer is different than what was used to '
                          'tokenize data.')
        return cls(tok=tok, **data)
