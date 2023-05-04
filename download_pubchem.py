import requests
import zipfile
import io
import gzip
import json
import os
from tqdm import tqdm
import subprocess
import multiprocessing
import sys


def get_substances(f_name):
    os.chdir(sys.argv[1])
    subprocess.call(f"obabel {f_name} -O {f_name.replace('.sdf', '.smi')} -b -d -e -r", shell=True, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    os.chdir(sys.argv[1])
    ranges = []
    for link in list(set(requests.get('https://ftp.ncbi.nlm.nih.gov/pubchem/Substance/CURRENT-Full/SDF/').text.split('<a href="')[3:-1])):
        ranges.append(link.split('.')[0])
    for range in tqdm(ranges):
        subprocess.call(f'wget https://ftp.ncbi.nlm.nih.gov/pubchem/Substance/CURRENT-Full/SDF/{range}.sdf.gz', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.call(f'gunzip *.gz', shell=True)
    files = []
    for f_name in os.listdir(sys.argv[1]):
        if f_name.endswith('.sdf'):
            files.append(f_name)
    with multiprocessing.Pool(8) as p:
        p.map(get_substances, files)
    for link in tqdm(requests.get('https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/Concise/JSON/').text.split('<a href="')[2:-2]):
        range = link.split('.')[0]
        r = requests.get(f'https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/Concise/JSON/{range}.zip')
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(sys.argv[1])
        if os.path.exists(f'{sys.argv[1]}/{range}'):
            for f_name in os.listdir(f'{sys.argv[1]}/{range}'):
                j = json.load(gzip.open(f'{sys.argv[1]}/{range}/{f_name}', 'r'))
                if j['PC_AssaySubmit']['assay']['descr']['results']:
                    try:
                        val_index = [r['name'] for r in j['PC_AssaySubmit']['assay']['descr']['results']].index('PubChem Standard Value')
                    except:
                        continue
                    for row in j['PC_AssaySubmit']['data']:
                        if row and row['data']:
                            print(f_name.replace('.concise.json.gz', ''), row['sid'], row['data'][val_index]['value']['fval'])