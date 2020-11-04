import streamlit as st


def html(text, sidebar=False):
    """Convenience function to create html element in streamlit.

    Parameters
    ----------
    text: str
        The html content to render, e.g. "<div>My Div</div>".
    sidebar: bool
        If True, render the element in the page's sidebar, otherwise render it
        in the main pane.

    Returns
    -------
    st.markdown: Streamlit element. Calling html(my_html_str) will render the
    resulting element on the page.
    """
    fn = st.sidebar if sidebar else st
    return fn.markdown(text, unsafe_allow_html=True)


def widget(kind, name, annotation, sidebar=True, pos='up', key=None, **kwargs):
    """Create a streamlit widget with a description that appears when you hover
    over its title.

    Parameters
    ----------
    kind: str
        Name of the streamlit element to create, e.g. "slider". Case sensitive.
    name: str
        The name that will be displayed directly above the widget.
    annotation: str
        A longer text explanation that will appear when the user hovers over
        the widget's name.
    sidebar: bool
        If True, render the widget and hover info in the page's sidebar.
        Otherwise, render it in the main pane.
    pos: str
        Determines position of the hoverable annotation. This should probably
        always be "up", meaning the annotation appears above the title, because
        streamlit seems to mess with some of the other options. The css is
        supposed to support "left", "right", and "down", however.
    key: str or None
        Usually unnecessary (only pass in if streamlit throws error). It's used
        to distinguish between multiple widgets on a page.
    kwargs: any
        Pass additional kwargs needed to instantiate the widget, e.g.
        min_value.

    Returns
    -------
    streamlit widget: Widgets usually return a value so this function will too.

    Examples
    --------
    alpha = widget('slider', 'Alpha',
                   'Larger values will lead to smoother plots',
                   min_value=.01, max_value=.99, value=.9)
    """
    tt_cls = 'tooltip-wrap-side' if sidebar else 'tooltip-wrap'
    html(f'<div class="{tt_cls}" data-balloon-pos="{pos}" '
         f'data-balloon-length="fit" aria-label="{annotation}">{name}</div>',
         sidebar=sidebar)
    widg = getattr(st.sidebar if sidebar else st, kind)('', key=key, **kwargs)
    return widg

