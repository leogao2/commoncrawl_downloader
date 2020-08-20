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
import fasttext
import trafilatura
import collections
import pybloomfilter
import zstd
import math
from textwrap import wrap


mode = 'trafilatura'


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


class RollingBloomFilter:
    def __init__(self, each_capacity=1000000, each_error_rate=0.0001, num_filters=10):
        self.filters = collections.deque()
        self.each_capacity = each_capacity
        self.each_error_rate = each_error_rate
        self.num_filters = num_filters
        self.seen = 0
    
    def roll(self):
        if len(self.filters) == self.num_filters:
            self.filters.pop()
        self.filters.appendleft(pybloomfilter.BloomFilter(self.each_capacity, self.each_error_rate))

    def add(self, x):
        if self.seen % self.each_capacity == 0:
            self.roll()

        self.filters[0].add(x)

        self.seen += 1
    
    def __contains__(self, item):
        for filter in self.filters:
            if item in filter:
                return True
        return False


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


langdet = fasttext.load_model("lid.176.bin") 


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
                    'extractor': 'justext'
                    **meta
                }
                return [x.text for x in 
                            justext.justext(html, justext.get_stoplist('English')) 
                        if not x.is_boilerplate], meta
        elif mode == 'trafilatura':
            result = trafilatura.extract(html)
            if result is None:
                return
            details = langdet.predict(result.replace('\n', ' ')[:2000], k=5)

            # turn np array in snd details into a list so json can serialize it
            a, b = details
            b = b.tolist()
            details = a, b
            meta = {
                'primary_language': details[0][0].replace('__label__', ''),
                'lang_detector': 'fasttext',
                'lang_detector_extra_info': details,
                'extractor': 'trafilatura',
                **meta
            }
            return result, meta
        else:
            raise AssertionError('unknown mode!')
    except lxml.etree.ParserError:
        return
    except:
        traceback.print_exc()


def get_cc_text(warc_urls):
    pool = mp.Pool(num_threads)

    yield from filter(lambda x:x and x[0],
                      pool.imap(html_to_text, warcurls_to_contents(warc_urls)))


compress_chunk_size = 1000
upper_sigma = 1
if __name__ == '__main__':
    for block in blocks_to_download:
        print('Downloading block', block)
        warcurls = urls_of_block(block)
        ars = {}
        
        rbf = RollingBloomFilter()


        skipped_docs_bloom = 0
        skipped_docs_ccr = 0
        total_docs = 0
        total_docs_gt_compress_chunk_size = 0

        ct_by_lang = collections.defaultdict(int)
        total_ccr_by_lang = collections.defaultdict(list)


        for text, meta in get_cc_text(warcurls):
            total_docs += 1
            if len(text.encode('utf-8')) > compress_chunk_size: total_docs_gt_compress_chunk_size += 1

            normalized = clean_for_bloom(text)
            if normalized in rbf: 
                skipped_docs_bloom += 1
                continue
            rbf.add(normalized)
            
            lang = meta['primary_language']
            if lang not in ars:
                ars[lang] = lmd.Archive(f'output/{lang}', compression_level=7)
            ct_by_lang[lang] += 1


            if len(text.encode('utf-8')) >= compress_chunk_size: 
                ccr = math.log(chunked_compression_ratio(text.encode('utf-8'), compress_chunk_size))
                total_ccr_by_lang[lang].append(ccr)

                if len(total_ccr_by_lang[lang]) > 1:
                    mu = mean(total_ccr_by_lang[lang])
                    s = stddev(total_ccr_by_lang[lang])
                    if ccr > mu + upper_sigma*s:
                        skipped_docs_ccr += 1
                        continue
            
            ars[lang].add_data(text, meta=meta)
        
        for ar in ars.values(): ar.commit(archive_name=block)

        with open('output/stats_{}.txt'.format(block), 'w') as fh:
            fh.write('skipped docs (bloom): {}\n'.format(skipped_docs_bloom))
            fh.write('skipped docs (ccr, {}-sigma): {}\n'.format(upper_sigma, skipped_docs_ccr))
            fh.write('total docs: {}\n'.format(total_docs))
            fh.write('total docs, length >{}: {}\n'.format(compress_chunk_size, total_docs_gt_compress_chunk_size))
            fh.write('total ccr by lang: {}\n'.format(total_ccr_by_lang))
            fh.write('totals by lang: {}\n'.format(ct_by_lang))