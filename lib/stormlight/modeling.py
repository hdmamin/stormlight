from functools import partial
import time
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from htools import save, spacer
from stormlight.utils import BREAK


def load_pretrained_gpt2(mod_name='gpt2', tok_name='gpt2',
                         pad_token='<|endoftext|>', special=(BREAK,)):
    """Quickly load a pretrained GPT2 tokenizer and model, adding special
    tokens and ensuring the model's embedding matrix matches the tokenizer's
    vocabulary.

    Parameters
    ----------
    mod_name: str
        Weights to load or configuration to use. This can be a name defined by
        the Transformers library (e.g. "gpt2") or the path to a file containing
        saved weights.
    tok_name: str
        Usually a name from the Transformers library (e.g. "gpt2") that
        specifies what pre-trained tokenizer to load. This may sometimes differ
        from `mod_name` because I've been saving fine-tuned models but using
        the default tokenizer.
    pad_token: str
        GPT2 doesn't use padding but we pass in a token anyway to avoid
        warnings.
    special: Iterable[str]
        Strings that the tokenizer should not split. We have to add the page
        break token.

    Returns
    -------
    tuple[tokenizer, model]
    """
    tok = GPT2TokenizerFast.from_pretrained(tok_name, pad_token=pad_token)
    tok.add_special_tokens({'additional_special_tokens': list(special)})

    model = GPT2LMHeadModel.from_pretrained(mod_name,
                                            pad_token_id=tok.pad_token_id)
    model.resize_token_embeddings(len(tok))
    model.tie_weights()
    return tok, model


def generate(model, tok, x, path, *skip_tokens, max_length=112, min_length=0,
             mode='a', verbose=True, return_idx=False, skip_special=True,
             with_spacer=True, **kwargs):
    """Generates text from a prompt and writes it to a file. Should be useful
    when creating a callback to monitor how language model is progressing
    during training.

    Parameters
    ----------
    model: transformers model
    tok: transformers tokenizer
    x: Iterable[int]
        Array of token indices to use as a prompt for text generation.
    path: str or path
        File to save output to. If None, output won't be written to any file.
    *skip_tokens: str
        Optional: One or more strings to avoid generating (originally used to
        avoid generating xxbrxx tokens; however, in practice we usually ended
        up eliminating those from the data first anyway).
    max_length: int
        Sequence length (sometimes the model stops early but that seems to be
        rare). This is NOT the same as the `max_length` parameter in the
        model's `generate` method: that counts the length of the sequence, i.e.
        an input sequence of length 10 with a max_length of 10 will generate
        nothing. This function makes it so we specify the number of tokens to
        generate, so the input sequence length is not a factor.
    min_length: int
        Like max_length above, this ignores the length of the input sequence.
    mode: str
        Specifies file-writing mode. Usually want 'a' when used as a callback.
    verbose: bool
        Specifies whether to print prompt and generated text.
    return_idx: bool
        If True, output will include a list of word indices
        (inputs + generated) in addition to other outputs.
    skip_special: bool
        If True, tokenizer will skip special tokens when decoding outputs.
    with_spacer: bool
        If True, print separator line (79*'-') after generated text in file.
        This is nice when using this as a callback so we can clearly see where
        one sample ends and the next begins.
    kwargs: any
        Additional kwargs to pass to model.generate().

    Returns
    -------
    tuple[str, str]: Decoded prompt and new generated text, in that order.
    If return_idx is True, a list of word indices will be included as a third
    item.
    """
    if isinstance(x, str):
        x = tok.encode(x, return_tensors='pt')
    elif not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    seq_len = x.shape[-1]

    kwargs_ = dict(max_length=seq_len+max_length,
                   min_length=seq_len+min_length,
                   repetition_penalty=10,
                   no_repeat_ngram_size=4,
                   early_stopping=False,
                   do_sample=True,
                   temperature=.7,
                   top_p=.95,
                   top_k=max_length,
                   bad_words_ids=[[tok.get_vocab()[t] for t in skip_tokens]]
                                 if skip_tokens else None)
    # Update after so defaults are overwritten if user provides value.
    kwargs_.update(kwargs)
    res = model.generate(x, **kwargs_)[0].cpu().numpy().tolist()
    old, new = map(partial(tok.decode, skip_special_tokens=skip_special),
                   (res[:seq_len], res[seq_len:]))
    if verbose:
        print(old, '\n\n', new)
    if path:
        to_save = old + new + spacer()*with_spacer
        save(to_save, path, mode_pre=mode, verbose=verbose)
    output = (old, new)
    if return_idx: output = (*output, res)
    return output


def live_generate(mod, tok, text, max_length, *skip_tokens,
                  write_fn=partial(print, end=''), min_length=0,
                  chunk_size=20, pause=.05, **kwargs):
    """Generate text and display it in real time (character by character)
    so it looks like it's being typed.

    Parameters
    ----------
    mod: transformers model
    tok: transformers tokenizer
    text: str
        Raw text to use as a prompt.
    write_fn: callable
        Function that will be used to print generated text. Originally thought
        I'd use this function to display text in streamlit, so I didn't want
        to hard code print statements.
    max_length: int
        Sequence length to generate (not counting prompt).
    skip_tokens: str
        Optionally provide one or more tokens to avoid generating. I've been
        passing in xxbrxx to be safe even though I trained without it. Since I
        added it to the tokenizer and model's vocab, I think it might still
        end up generating this if I don't ban it.
    min_length: int
        Minimum length to generate (not counting prompt). I ended up turning
        off the `early_stopping` option so I'm not sure this matters anymore.
    chunk_size: int
        Number of tokens to generate at a time. Larger values seem to be a bit
        faster (not entirely sure why, since the model is supposed to be
        caching features).
    pause: float
        Time to pause after writing each character. Set to 0 to display as fast
        as possible.
    kwargs: any
        Additional args to pass to `generate`. These will override default
        args. Ex: temperature.

    Returns
    -------
    str: The whole text sequence including both prompt and generated text.
    """
    for char in text:
        write_fn(char)
        time.sleep(pause)
    x = tok.encode(text, return_tensors='pt').squeeze()
    total_len = x.shape[-1] + max_length
    while len(x) < total_len:
        text, new, x = generate(mod, tok, text, None, *skip_tokens,
                                max_length=min(chunk_size, total_len-len(x)),
                                min_length=min_length, verbose=False,
                                return_idx=True, **kwargs)
        for char in new:
            write_fn(char)
            time.sleep(pause)
        text += new
    return text


def live_st_generate(mod, tok, text, max_length, *skip_tokens, min_length=0,
                     chunk_size=1, memory=40, pause=.03, **kwargs):
    """Generator function that displays text in real time in streamlit. This is
    a slightly hacky solution to get around the fact that streamlit seems to
    prevent us from skipping the newline character when calling st.write or
    st.markdown repeatedly.

    Parameters
    ----------
    mod: transformers model
    tok: transformers tokenizer
    text: str
        Raw text to use as a prompt.
    max_length: int
        Sequence length to generate (not counting prompt).
    skip_tokens: str
        Optionally provide one or more tokens to avoid generating. I've been
        passing in xxbrxx to be safe even though I trained without it. Since I
        added it to the tokenizer and model's vocab, I think it might still
        end up generating this if I don't ban it.
    min_length: int
        Minimum length to generate (not counting prompt). I ended up turning
        off the `early_stopping` option so I'm not sure this matters anymore.
    chunk_size: int
        Number of tokens to generate at a time. Larger values might be a little
        faster overall but it will feel slower (text will appear in spurts
        while the model generates the next big chunk).
    memory: int
        Number of previous tokens to take into account when making each
        prediction. Pass in an enormous int to take the whole passage into
        account. Larger numbers may give better results but will be much
        slower.
    pause: float
        Time to pause after writing each character. Set to 0 if you want to
        display text as fast as possible.
    kwargs: any
        Additional args to pass to `generate`. These will override default
        args. Ex: temperature.

    Yields
    ------
    str: One character at a time from the newly generated text of length
    `chunk_size`. Unlike my initial attempt at this function, this excludes the
    prompt and any previously generated text.

    Examples
    --------
    prompt = 'I began'
    for chunk in live_st_generate(mod, tok, prompt, 3):
        md.markdown(chunk)

    First round of yields: 'I', ' ', 'b', 'e', 'g', 'a', 'n'
    Second round: ' ', 't', 'o', ' '
    Third round: 'l', 'a', 'u', 'g', 'h'
    Fourth round: 'a', 't'

    When we refer to the "first round of yields" here, we're saying this
    corresponds to the first yield statement in the function. In practice,
    we only yield a single character at each step. Ultimately, the full
    sequence reads 'I began to laugh at'.
    """
    for char in text:
        yield char
        time.sleep(pause)

    x = tok.encode(text, return_tensors='pt').squeeze()
    curr_len = x.shape[-1]
    total_len = curr_len + max_length
    while curr_len < total_len:
        # Each call passes in indices to avoid re-encoding.
        _, new, x = generate(mod, tok, x[-memory:], None, *skip_tokens,
                             max_length=min(chunk_size, total_len-len(x)),
                             min_length=min_length,
                             verbose=False, return_idx=True, **kwargs)
        for char in new:
            yield char
            time.sleep(pause)
        curr_len += chunk_size

