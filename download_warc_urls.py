import os
from tqdm import tqdm
import requests
import zlib


if __name__ == '__main__':
    
    index_path = 'indexes_20200607105929'

    ret = []
    with open(index_path) as ind:
        for url in tqdm(ind):
            response = requests.get(url.strip(), stream=True)
            
            data = zlib.decompress(response.content, zlib.MAX_WBITS|32)
            for warc in data.decode('utf-8').split('\n'):
                ret.append(warc)
    
    with open(index_path + '_warc_urls.txt', 'w') as fh:
        fh.write('\n'.join(ret))