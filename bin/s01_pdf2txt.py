"""DEPRECATED:
Originally used this to convert PDFs to txt files. Didn't work on Warbreaker
for some reason so I ended up using the `pdftotext` command line tool in
raw2interim.sh, which converts all 6 files faster than a single file with this
script. Keeping this here just in case I need to convert more PDFs later and
encounter issues with `pdftotext`.
"""

from glob import glob
from pathlib import Path
import slate3k

from htools import LoggerMixin, save


SRC_DIR = 'data/raw'
OUT_DIR = Path('data/interim')


def main():
    logger = LoggerMixin().get_logger('data/logs/gen_raw.log')
    for path in glob(f'{SRC_DIR}/*.pdf'):
        logger.info(f'Processing {path.split("/")[-1]}...')
        pdf2txt(path, logger)


def pdf2txt(path, logger):
    path = Path(path)
    try:
        with open(path, 'rb') as f:
            doc = slate3k.PDF(f)
    except Exception as e:
        logger.info(f'{path}: + {str(e)}')
    else:
        text = '\n'.join(doc)
        save(text, OUT_DIR/path.parts[-1].replace('pdf', 'txt'))


if __name__ == '__main__':
    main()

