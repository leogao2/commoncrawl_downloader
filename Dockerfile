FROM python:3.8.4

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt
COPY download_warc_urls.py make_url_blocks indexes_20200607105929 /app/
RUN ./make_url_blocks

COPY . /app

ENTRYPOINT ["python", "download_commoncrawl.py"]