import sys
import time
import requests
from tqdm import *
from paths import *
import pandas as pd


def check_integrity(_url, _path):
    with requests.get(_url, stream=True) as r:
        content_length = int(requests.head(url).headers["Content-Length"])
        size_on_disk = int(_path.stat().st_size if _path.exists() else 0)
        print(f'{_path.name} | content_length: {content_length} - disk_size: {size_on_disk}')
        if size_on_disk < content_length:
            print(f'{_path.name} corrupted, downloading...')
            try:
                download_file(_url, _path, None)
            except:
                time.sleep(20)
                check_integrity(_url, _path)

def download_file(_url, _path, _desc):
    chunk_size = 1024
    total = int(requests.head(url).headers["Content-Length"])
    with requests.get(_url, stream=True) as r, open(_path, "wb") as f, tqdm(
        unit="B",  # unit string to be displayed.
        unit_scale=True,  # let tqdm to determine the scale in kilo, mega..etc.
        unit_divisor=1024,  # is used when unit_scale is true
        total=total,  # the total iteration.
        file=sys.stdout,  # default goes to stderr, this is the display on console.
        desc=_desc  # prefix to be displayed on progress bar.
    ) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive new chunks
                datasize = f.write(chunk)
                pbar.update(datasize)

if __name__ == '__main__':
    dtm_path.mkdir(parents=True, exist_ok=True)
    gray_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_filename)
    download_list = []
    for _, row in df.iterrows():
        name = row['dtm'].split('/')[-2]
        download_list.append((row['dtm'], dtm_path / f'{name}.IMG'))
        download_list.append((row['left'], gray_path / f'{name}_L.JP2'))
        download_list.append((row['right'], gray_path / f'{name}_R.JP2'))
    
    queue = download_list
    i = 1
    for item in queue:
        url, path = item
        path = Path(path)
        if(path.exists() == False):
            desc = f'{i}/{len(queue)}'
            try:
                download_file(url, path, desc)
            except Exception as e:
                print(f'{e}\nFile added to the download queue...')
                queue.append(url, path)
        else:
            print(f'{i + 1}/{len(queue)} already downloaded')
        i += 1

    for url, path in download_list:
        check_integrity(url, path)
