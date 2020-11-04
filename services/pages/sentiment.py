import plotly.graph_objs as go
import streamlit as st

from htools import load
from stormlight.utils import STORMLIGHT, NOVELLAS, PRE2FULL

from .utils import widget


SENT_DF_PATH = 'data/clean/compound_sentiment_scores.pkl'


@st.cache(hash_funcs={dict: id})
def load_sentiment_scores(path):
    """Wrapper to load and cache sentiment scores."""
    return load(path)


def build(locals_):
    st.title('Sentiment Analysis')
    st.markdown(
        'Sentiment analysis aims to identify whether a piece of text is '
        'positive or negative and often assigns a score that quantifies the '
        'strength of this assessment. For example, the sentence "It\'s a '
        'beautiful day." is generally positive while the sentence '
        '"It was a miserable evening." is generally negative. I computed '
        'sentiment scores for each sentence in the Stormlight Archive, '
        'normalized them to lie between -1 and 1 (where scores close to 1 '
        'correspond to strong positive sentiment and scores close to -1 '
        'correspond to strong negative sentiment) and display the smoothed '
        'results below.\n\n'
        '**Note:** You can use the sidebar to adjust global options. The '
        'graphs are also interactive, so you can zoom in on different '
        'portions of each book.'
    )
    df = load_sentiment_scores(SENT_DF_PATH)
    max_len = max(d.shape[0] for d in df.values())
    books = st.sidebar.multiselect('Books',
                                   STORMLIGHT+NOVELLAS,
                                   STORMLIGHT+NOVELLAS,
                                   format_func=PRE2FULL.get)
    options = ['First n sentences', 'Last n sentences', 'Whole book']
    window = st.sidebar.radio('Book Subset', options, index=0)
    if window in options[:-1]:
        rows = st.sidebar.number_input('# of Sentences', 1, max_len,
                                       value=1_000)
    else:
        rows = None
    idx = slice(-rows, None) if window == options[1] else slice(None, rows)
    span = widget(
        'slider', 'Moving Average Span',
        'Sentiment scores vary wildly from sentence to sentence and the '
        'graphs become much more readable when we display an exponentially '
        'weighted moving average. When people speak of an “n-day EWMA“, n '
        'refers to the span. Larger values will result in smoother lines '
        'while smaller values will better showcase large fluctuations. '
        'For the statistically minded, span=(2/alpha)-1 where alpha is the '
        'EWMA smoothing parameter. We try to select a reasonable value '
        'based on the number of sentences you choose to display, but you '
        'can also select your own. ',
        min_value=5, max_value=750,
        value=375 if window == options[-1] else max(5, min(rows//10, 375))
    )

    for book in books:
        data = df[book].iloc[idx]
        traces = [go.Scatter(x=data.index,
                             y=data.scaled_minmax_score.ewm(span=s,
                                 min_periods=int(s/10)).mean(),
                             opacity=alpha/3,
                             line=dict(color='royalblue', width=alpha),
                             name=f'EWMA Span={s}')
                  for s, alpha in zip([span//5, span], [1, 3])]
        fig = go.Figure(data=traces)
        fig.update_layout(title=PRE2FULL[book], xaxis_title='Sentence #',
                          yaxis_title='Score', legend_orientation='h',
                          legend=dict(x=0, y=1), width=800,
                          margin=dict(l=0, r=0, b=35, t=35, pad=0)
        )
        st.plotly_chart(fig)

