import fire
from pathlib import Path

from htools import load
from stormlight.modeling import load_pretrained_gpt2, generate
from stormlight.utils import BREAK


def generate_to_file(out_file, max_len, model_dir='data/models/v5', temp=.7,
                     top_p=.95, max_ngram=4, rep_penalty=5, src_file='',
                     src='', mode='w', with_spacer=False):
    assert bool(src_file) + bool(src) == 1, \
        'Specify either source file or source string.'

    tok, model = load_pretrained_gpt2(model_dir, 'gpt2')
    if src_file:
        src = load(src_file)
    print('Loaded model and tokenizer.')
    _ = generate(model, tok, src, Path('data/generated')/out_file, BREAK,
                 max_length=max_len, temp=temp, top_p=top_p,
                 max_ngram=max_ngram, rep_penalty=rep_penalty, mode=mode,
                 with_spacer=with_spacer)


if __name__ == '__main__':
    fire.Fire(generate_to_file)
