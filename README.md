# commoncrawl_downloader

Sample usage:

```
docker build -t ccdl .
docker run -e NUM_CORES=8 -v $PWD/output:/app/output -it ccdl 0,1,2,3,4,5,6,7,8,9,10
```

There are 3679 blocks in total (numbered 0-3678 inclusive). To specify blocks, provide a comma-seperated list of block numbers as the argument (no spaces). 

# Resources required

3.5PB of network ingress in total is required. The final dataset should be (warning: this number is very rough and extrapolated; leave some slack space to be safe!) about 200TB. About 40k core days (non-hyperthreaded) are also required (again, a very rough estimate from extrapolation). 

# Output format

Each block outputs as a ~40GB `.jsonl.zst` file (info: [jsonlines](http://jsonlines.org/), [zstd](https://facebook.github.io/zstd/)). Each json object in the file has a `text` field, containing the webpage, and a `meta` field containing metadata about the language, the WARC headers, and the HTTP response headers.
