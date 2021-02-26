import sys
from tqdm import tqdm
import multiprocessing as mp
import warcio
from warcio.archiveiterator import ArchiveIterator
import requests
import traceback
import lm_dataformat as lmd
import cchardet as chardet
import unicodedata
import os
#import fasttext
#import trafilatura
import collections
#import pybloomfilter
import zstd
import math
# from textwrap import wrap
import json
import abc


#mode = 'justext'
mode = 'cc_imgs'


blocks_to_download = sys.argv[1].split(',')
num_threads = int(os.environ['NUM_CORES'])

import re
def clean_for_bloom(x):
    x = re.sub(r'\d', '#', x.lower().strip())
    return x

def mean(x):
    return sum(x) / len(x)

def stddev(x):
    mu = mean(x)
    return math.sqrt(sum(map(lambda t: (t - mu)**2, x)) / (len(x) - 1))

def compression_ratio(text):
    return len(text) / len(zstd.ZstdCompressor(level=2).compress(text))


def chunked_compression_ratio(text, chksize):
    res = []
    for i in range(0, len(text), chksize):
        if (i+1)*chksize > len(text): continue
        chunk = text[i*chksize:(i+1)*chksize]*10
        res.append(compression_ratio(chunk))
    
    return mean(res)


def urls_of_block(block):
    with open('warc_blocks/urls_' + block.rjust(4, '0')) as fh:
        yield from map(lambda x: "https://commoncrawl.s3.amazonaws.com/" + x, fh)


def wat_urls_of_block(block):
    with open('warc_blocks/urls_' + block.rjust(4, '0')) as fh:
        yield from map(lambda x: "https://commoncrawl.s3.amazonaws.com/" + x.replace('warc.gz', 'warc.wat.gz').replace('/warc/', '/wat/'), fh)


def warcurl_to_contents(warc_url, wat=False):
    try:
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
            if wat and record.rec_type == 'metadata':
                content = record.content_stream().read()
                meta = {
                    'warc': warc_url.strip(),
                    'target': record.rec_headers.get_header('WARC-Target-URI'),
                }

                yield content, meta
    except warcio.exceptions.ArchiveLoadFailed:
        print('WARNING: WARC load failed!')
        traceback.print_exc()


def warcurls_to_contents(warc_urls, wat=False):
    for url in tqdm(list(warc_urls)):
        yield from warcurl_to_contents(url, wat)


#import pycld2 as cld2
#import justext
#import lxml

try:
    from urlparse import urljoin  # Python2
except ImportError:
    from urllib.parse import urljoin  # Python3

# from best_download import download_file
# import fasttext
# download_file('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin', 'lid.176.bin', '7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e')


# todo: make HtmlExtractor class to seperate justext and trafilatura logic
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
        if mode == 'justext':
            raise AssertionError('justext not implemented!')
        #     try:
        #         _,_,details = cld2.detect(html)
        #     except:
        #         # cld2 doesn't like control characters
        #         # https://github.com/mikemccand/chromium-compact-language-detector/issues/22#issuecomment-435904616
        #         html_no_ctrl_chars = ''.join([l for l in html if unicodedata.category(l)[0] not in ['C',]])
        #         _,_,details = cld2.detect(html_no_ctrl_chars)

        #     if details[0][1] == 'en':
        #         meta = {
        #             'primary_language': 'en',
        #             'lang_detector': 'pycld2',
        #             'lang_detector_extra_info': details,
        #             'extractor': 'justext',
        #             **meta
        #         }
        #         return [x.text for x in 
        #                     justext.justext(html, justext.get_stoplist('English')) 
        #                 if not x.is_boilerplate], meta
        # elif mode == 'trafilatura':
        #     result = trafilatura.extract(html)
        #     if result is None:
        #         return
        #     details = langdet.predict(result.replace('\n', ' ')[:2000], k=5)

        #     # turn np array in snd details into a list so json can serialize it
        #     a, b = details
        #     b = b.tolist()
        #     details = a, b
        #     meta = {
        #         'primary_language': details[0][0].replace('__label__', ''),
        #         'lang_detector': 'fasttext',
        #         'lang_detector_extra_info': details,
        #         'extractor': 'trafilatura',
        #         **meta
        #     }
        #     return result, meta
        elif mode == 'cc_imgs':
            try:
                obs = json.loads(html)['Envelope']['Payload-Metadata']['HTTP-Response-Metadata']['HTML-Metadata']['Links']
                
                cc_urls = [ob for ob in obs if 'creativecommons.org' in ob['url']]
                keep = len(cc_urls) > 0
                if not keep: return None

                obs = [{'url': urljoin(meta['target'], ob['url']), 'alt': ob['alt']} for ob in obs if ob['path'] == 'IMG@/src' and 'alt' in ob and ob['alt']]

                meta = {
                    'cc_urls': cc_urls,
                    **meta
                }
                return obs, meta
            except KeyError:
                return
        else:
            raise AssertionError('unknown mode!')
    #except lxml.etree.ParserError:
    #    return
    except:
        traceback.print_exc()


def get_cc_text(warc_urls, html_to_text, wat=False):
    pool = mp.Pool(num_threads)

    yield from filter(lambda x:x and x[0],
                      pool.imap(html_to_text, warcurls_to_contents(warc_urls, wat)))


class Hook(abc.ABC):
    @abc.abstractmethod
    def write_doc(self, doc, meta):
        pass

    @abc.abstractmethod
    def commit_block(self, block):
        pass


class ArchiveHook(Hook):
    def __init__(self):
        self.ars = {}
        self.total_docs = 0
        self.ct_by_lang = collections.defaultdict(int)

    def write_doc(self, doc, meta):
        lang = meta['primary_language']
        if lang not in self.ars:
            self.ars[lang] = lmd.Archive(f'output/{lang}', compression_level=7)
        self.ars[lang].add_data(doc, meta)
        self.ct_by_lang[lang] += 1
        self.total_docs += 1

    def commit_block(self, block):
        for ar in self.ars.values(): ar.commit(archive_name=block)

        with open('output/stats_{}.txt'.format(block), 'w') as fh:
            fh.write('total docs: {}\n'.format(self.total_docs))
            fh.write('totals by lang: {}\n'.format(self.ct_by_lang))

        self.ars = {}
        self.total_docs = 0
        self.ct_by_lang = collections.defaultdict(int)


class DebugHook(Hook):
    def __init__(self):
        self.ars = {}
        self.total_docs = 0
        self.ct_by_lang = collections.defaultdict(int)

    def write_doc(self, doc, meta):
        print(doc, meta)

    def commit_block(self, block):
        pass


class SimpleArchiveHook(Hook):
    def __init__(self):
        self.ar = lmd.Archive(f'output', compression_level=7)
        self.ct_by_lang = collections.defaultdict(int)

    def write_doc(self, doc, meta):
        self.ar.add_data(doc, meta)

    def commit_block(self, block):
        self.ar.commit(archive_name=block)
            


def download(blocks, html_to_text, keep_doc, hooks, wat=False):
    for block in blocks:
        print('Downloading block', block)
        warcurls = wat_urls_of_block(block) if wat else urls_of_block(block)

        for text, meta in get_cc_text(warcurls, html_to_text, wat):
            if keep_doc(text):
                for hook in hooks: hook.write_doc(text, meta)
        
        for hook in hooks: hook.commit_block(block)


def keep_doc(doc):
    return True

if __name__ == '__main__':
    #download(blocks_to_download, html_to_text, keep_doc, [ArchiveHook()])
    download(blocks_to_download, html_to_text, keep_doc, [SimpleArchiveHook()], wat=True)
