from stormlight.utils import PATHS, BOOK_IDX, BREAK

from htools import load, DotDict, spacer


def load_book(book, replace_break=None, subset=None):
    """Load a single book's clean text, trimming off TOC and appendices.

    Parameters
    ----------
    book: str
        One of the book prefixes defined in utils (e.g. 'war', 'edge').
    replace_break: str or None (optional)
        If not None, should be a string which will be used to replace the page
        break token xxbrxx. The typical use case is deleting page breaks by
        passing in ''.
    subset: int or None
        If an int is provided, this will return the first n characters of the
        clean book. Otherwise, the whole book will be returned. (In this
        context, "whole book" refers to the cleaned text.)

    Returns
    -------
    str: The book's full text.
    """
    text = load(PATHS[book])[slice(*BOOK_IDX[book])]
    if replace_break is not None:
        text = text.replace(BREAK, replace_break)
    return text[:subset]


def load_books(*books, replace_break=None, subset=None):
    """Load clean, trimmed text from one or more books.

    Parameters
    ----------
    books: str
        Book prefixes as defined in `utils.py`. If None are provided, all books
        will be loaded.
    replace_break: str or None (optional)
        If not None, should be a string which will be used to replace the page
        break token xxbrxx. The typical use case is deleting page breaks by
        passing in ''.
    subset: int or None
        If an int is provided, this will return the first n characters of each
        clean book. Otherwise, the full clean text for each book will be
        returned.

    Returns
    -------
    DotDict[str, str]: Maps book prefix to clean text.

    Examples
    --------
    We've defined lists of subsets of books in `utils.py` which lets us do
    things like this:

    load_books(*STORMLIGHT+NOVELLAS)
    """
    return DotDict((book, load_book(book, replace_break, subset))
                   for book in books or PATHS.keys())


def check_clean(book, *args):
    """Check # characters, # pages, # feed form characters after loading book.

    Parameters
    ----------
    book: str
        Full text.
    args: str
        Additional characters/words to check for.

    Returns
    -------
    None
    """
    print('Length:', len(book))
    print('Page breaks:', book.count(BREAK))
    print('Feed form chars:', book.count('\x0c'))
    for arg in args:
        print(repr(arg) + ':', book.count(arg))


def print_pages(book, n=150, end=True, max_pages=None):
    """Print first and last few characters from each page of a loaded book.

    Parameters
    ----------
    book: str
        Full text of a book.
    n: int
        Number of chars to print from beginning and end, respectively. The
        total printed characters will therefore be 2*n.
    end: bool
        If True, print chars from both the start and end of each page.
    max_pages: int, None
        If provided, limit the number of pages to print for. For example, in
        some cases we might just want a quick glimpse so we could print only
        the first 5 pages instead of all 500.

    Returns
    -------
    None
    """
    for page in book.split(BREAK)[:max_pages]:
        print(page[:n])
        if end:
            print(spacer(n_chars=3))
            print(page[-n:])
        print(spacer())


def load_endnotes(book):
    """Load endnotes of a book. This is the text directly following the text
    loaded in `load_book()`. It's pretty repetitive for the Stormlight books
    so I chose not to include this in the main portion. Instead, I just use
    this to augment the last sequence in a LMDataset if necessary. Note:
    I don't trim anything off the end here, so there's likely some garbage text
    included. With my current use case we never actually use the end though.

    Parameters
    ----------
    book: str
        One of the book name abbreviations defined in `utils`.

    Returns
    -------
    str: The end of the book (usually preceded by a message like
        "end of book 1 of the stormlight archive").
    """
    return load(PATHS[book])[BOOK_IDX[book][-1]:]

