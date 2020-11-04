import base64
import streamlit as st
from transformers import GPT2TokenizerFast

from stormlight.modeling import (load_pretrained_gpt2, live_st_generate,
    generate)
from stormlight.utils import BREAK

from .utils import widget, html


@st.cache(hash_funcs={GPT2TokenizerFast: id}, allow_output_mutation=True,
          suppress_st_warning=True)
def load_cached_model(**kwargs):
    """Wrapper to load model and tokenizer that allows streamlit to cache
    objects.
    """
    return load_pretrained_gpt2(**kwargs)


def download_text(text, fname):
    """Save text as .txt file.

    Parameters
    ----------
    text: str
        Text to download.
    fname: str
        Name of file to download to. Should end with '.txt'.

    Returns
    -------
    str: When passed to a streamlit markdown object with
        `unsafe_allow_html=True`, this will produce a link that downloads
        a text file with the specified name.
    """
    text = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{text}" download="{fname}">' \
           'Download generated text</a>'


def build(locals_):
    tok, model = load_cached_model(mod_name=locals_['mod_name'],
                                   tok_name=locals_['tok_name'])

    st.title('Stormlight Text Generator')
    st.write('by GPT2 ft. Brandon Sanderson')
    st.sidebar.title('Generation Options')

    # SIDEBAR PANE
    st.sidebar.markdown('Mouse over each widget for a quick description of '
                        'how it affects the generated text. This blog post '
                        'provides more in-depth explanations:'
                        '\nhttps://huggingface.co/blog/how-to-generate'
                        '\n\nWe\'ve selected reasonable defaults so feel free '
                        'to leave them as is. The main one you might want to '
                        'adjust is "Passage Length" which should be fairly '
                        'self-explanatory.')

    live_type = widget(
        'checkbox', 'Live Typing',
        'Turning off the live typing effect will only display text once the '
        'entire sequence has been generated. This will take less time but '
        'it will likely feel slower. This is a good option if you just want '
        'download your generated passage rather than reading it in real time.',
        value=True
    )

    length = widget(
        'number_input', 'Passage Length',
        'This roughly equates to the number of words to generate. '
        'Technically, it specifies the number of word tokens, which '
        'includes both words and sub-words.',
        min_value=1, max_value=750, value=50
    )

    memory = widget(
        'slider', 'Memory',
        'This controls how much of the recent context is taken into account '
        'when generating new text. Smaller values will be faster but may be '
        'lower quality.',
        min_value=10, max_value=100, value=50
    )

    # Temperature of 0 is greedy search, 1 is more random.
    temp = widget(
        'slider', 'Temperature',
        'Larger values allow slightly more randomness.',
        min_value=0.01, max_value=1.0, value=0.7
    )

    # Redistribute softmax probabilities to words within the top `p` percent.
    p = widget(
        'slider', 'Top P',
        'Smaller values eliminate the (admittedly already low) chances of '
        'sampling some of the less likely candidate words.',
        min_value=0.01, max_value=1.0, value=0.95
    )

    # Careful: value of 2 means no bi-gram can occur > once in the whole text.
    max_ngram = widget(
        'slider', 'Max N-Gram (no repeat)',
        'This specifies the largest size N-Gram that can\'t be repeated '
        'within the generated text. Smaller values will lead to less '
        'reptition. Use with caution: when generating long passages, low '
        'values can lead to poor results.',
        min_value=2, max_value=10, value=4
    )

    # Upper bound is technically unbounded but we limit it here for QC.
    rep_penalty = widget(
        'slider', 'Repetition Penalty',
        'Larger values make the model less likely to repeat itself. This can '
        'lead to worse results at times since some repetition is often '
        'reasonable (e.g. using the same name several times in a short '
        'passage because the same characters are present throughout the '
        'scene).',
        min_value=1, max_value=10, value=5
    )

    kwargs = dict(temperature=temp, top_p=p, no_repeat_ngram_size=max_ngram,
                  repetition_penalty=rep_penalty)

    # MAIN PANE
    prompt = st.text_area('Enter text here')
    st.sidebar.markdown('Created by Harrison Mamin'
                        '\nhttps://github.com/hdmamin/stormlight')
    # html style: https://gist.github.com/ines/b320cb8441b590eedf19137599ce6685
    # `html` function is not compatible with live updates.
    html_ = '<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>'
    md = st.markdown(html_.format(''), unsafe_allow_html=True)
    if prompt:
        with st.spinner('Writing draft #5...'):
            if live_type:
                all_text = ''
                for chunk in live_st_generate(model, tok, prompt, length,
                                              BREAK, memory=memory, **kwargs):
                    all_text += chunk
                    md.markdown(html_.format(all_text.replace('\n', '<br>')),
                                unsafe_allow_html=True)
                chunk = all_text
            else:
                old, new = generate(model, tok, prompt, None, BREAK,
                                    max_length=length, verbose=False, **kwargs)
                chunk = old + new
                md.markdown(html_.format(chunk.replace('\n', '<br>')),
                            unsafe_allow_html=True)

        # FILE DOWNLOAD OPTION
        html(download_text(chunk, 'stormlight.txt'))

