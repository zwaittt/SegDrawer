import os
import gdown

import json

def download_pt(list):
  dest_path = 'assets/pt'
  os.makedirs(dest_path, exist_ok=True)
  for pt in list:
    name = pt['name']
    url = pt['url']
    dest_path_exec = f'{dest_path}/{name}'
    if 'drive.google.com' in url:
      gdown.download(url, dest_path_exec, quiet=False)
    else:
      os.system(f'wget {url} -O {dest_path_exec}')

with open('models.json', 'r') as json_file:
  pt_array = json.load(json_file)
  download_pt(pt_array)

