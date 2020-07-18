import sys
from tqdm import tqdm
import multiprocessing as mp
from warcio.archiveiterator import ArchiveIterator
import requests
import traceback
import lm_dataformat as lmd
import cchardet as chardet
import unicodedata
import os


blocks_to_download = sys.argv[1].split(',')
num_threads = int(os.environ['NUM_CORES'])


def urls_of_block(block):
    with open('warc_blocks/urls_' + block.rjust(4, '0')) as fh:
        yield from map(lambda x: "https://commoncrawl.s3.amazonaws.com/" + x, fh)


def warcurl_to_contents(warc_url):
    response = requests.get(warc_url.strip(), stream=True)
    for record in ArchiveIterator(response.raw, arc2warc=True):
        if record.rec_type == 'response':
            content = record.content_stream().read()
            meta = {
                'warc': warc_url.strip(),
                'warc_headers': record.rec_headers.headers,
                'http_response_headers': record.http_headers.headers,
            }

            yield content, meta


def warcurls_to_contents(warc_urls):
    for url in tqdm(list(warc_urls)):
        yield from warcurl_to_contents(url)


import pycld2 as cld2
import justext
import lxml


def html_to_text(args):
    html, meta = args
    try:
        html = html.decode('utf-8')
    except UnicodeDecodeError: 
        # try to figure out encoding if not urf-8

        guess = chardet.detect(html)['encoding']

        if not guess or guess == 'UTF-8': return

        try:
            html = html.decode(guess)
        except (UnicodeDecodeError, LookupError):
            # still cant figure out encoding, give up
            return
        
    
    try:
        try:
            _,_,details = cld2.detect(html)
        except:
            # cld2 doesn't like control characters
            # https://github.com/mikemccand/chromium-compact-language-detector/issues/22#issuecomment-435904616
            html_no_ctrl_chars = ''.join([l for l in html if unicodedata.category(l)[0] not in ['C',]])
            _,_,details = cld2.detect(html_no_ctrl_chars)

        if details[0][1] == 'en':
            meta = {
                'primary_language': 'en',
                'lang_detector': 'pycld2',
                'lang_detector_extra_info': details,
                **meta
            }
            return [x.text for x in 
                        justext.justext(html, justext.get_stoplist('English')) 
                    if not x.is_boilerplate], meta
    except lxml.etree.ParserError:
        return
    except:
        traceback.print_exc()


def get_cc_text(warc_urls):
    pool = mp.Pool(num_threads)

    yield from filter(lambda x:x and x[0],
                      pool.imap(html_to_text, warcurls_to_contents(warc_urls)))


if __name__ == '__main__':
    for block in blocks_to_download:
        print('Downloading block', block)
        warcurls = urls_of_block(block)
        ar = lmd.Archive('output')
        for text, meta in get_cc_text(warcurls):
            ar.add_data(text, meta=meta)
        ar.commit(archive_name=block)