"""Run from the project root with the command `streamlit run services/app.py`.
You can specify model weights and/or a tokenizer to load using a command like

`streamlit run services/app.py data/models/v4 gpt2-medium`.

The current setup with Fire seems to treat all the arguments as positional.
"""
import fire
import streamlit as st

from pages import counts, sentiment, generation
from pages.utils import html


PAGES = {
    'Text Generator': generation,
    'Sentiment Analysis': sentiment,
    'Word Counts': counts,
}


@st.cache
def load_css(path='services/style.css'):
    """Load css stylesheet as string.

    Parameters
    ----------
    path: str or Path
        Location of CSS file relative to where app is run from. I.e if we're in
        the `stormlight` directory using the command
        `streamlit run services/app.py`, this would be services/style.css.

    Returns
    -------
    str: This will need to be rendered using st.markdown() with
        unsafe_allow_html=True. It's already wrapped in style tags.
    """
    with open(path, 'r') as f:
        text = f.read()
    return f'<style>{text}</style>'


def main(mod_name='gpt2', tok_name='gpt2'):
    """Run app."""
    # Load stylesheets.
    css = load_css()
    html(css)

    # Page content.
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Pages', list(PAGES.keys()))
    PAGES[page].build(locals())


if __name__ == '__main__':
    fire.Fire(main)

