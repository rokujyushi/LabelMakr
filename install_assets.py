import os
import logging
import shutil
import zipfile
from pathlib import Path as P

import requests
from tqdm import tqdm

#
#	extract models into the proper location
#

# logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(format="| %(levelname)s | %(message)s | %(asctime)s |",
					datefmt="%H:%M:%S")
logger.setLevel(logging.INFO)

folder = P('./models')
if not folder.exists():
	folder.mkdir()


def download_file(url, filepath):
	r = requests.get(url, stream=True)
	r.raise_for_status()

	total_size = int(r.headers.get('content-length', 0))
	block_size = 1024

	with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
		with open(filepath, 'wb') as file:
			for data in r.iter_content(block_size):
				if not data:
					continue
				pbar.update(len(data))
				file.write(data)

	if total_size != 0 and pbar.n != total_size:
		raise RuntimeError('Could not download file.')


def install_sofa_asset():
	logger.info('SetUp SOFA for LabelMakr.')
	filepath = 'SOFA-main.zip'
	extract_dir = P('./_sofa_extract')

	if extract_dir.exists():
		shutil.rmtree(extract_dir)

	url = 'https://github.com/qiuqiao/SOFA/archive/refs/heads/main.zip'
	download_file(url, filepath)
	extract_dir.mkdir(exist_ok=True)

	logger.info('Unzipping SOFA...')
	with zipfile.ZipFile(filepath, 'r') as archive:
		archive.extractall(extract_dir)

	extracted = [path for path in extract_dir.iterdir() if path.is_dir()]
	if not extracted:
		raise RuntimeError('SOFA archive could not be extracted.')

	if P('./SOFA').exists():
		shutil.rmtree('./SOFA')

	shutil.move(str(extracted[0]), './SOFA')
	shutil.rmtree(extract_dir)
	os.remove(filepath)
	logger.info('Done setting up SOFA for LabelMakr.')

def spicytigermeat_asset():
	logger.info('SetUp spicytigermeat_assets for LabelMakr.')
	logger.info('Downloading SOFA models.')

	url = 'https://github.com/spicytigermeat/LabelMakr/releases/download/assets_v030/models.zip'
	filepath = 'models.zip'
	download_file(url, filepath)

	logger.info('Sucessfully donwloaded models.')
	logger.info('Unzipping...')

	with zipfile.ZipFile(filepath, 'r') as archive:
		archive.extractall('./models')

	os.remove(filepath)
	logger.info('Done setting up spicytigermeat_assets for LabelMakr.')

def add_some_assets():
	logger.info('SetUp add_some_assets for LabelMakr.')
	logger.info('Downloading JPN_Romaji_Test2_Plus models.')

	url = 'https://github.com/Greenleaf2001/SOFA_Models/releases/tag/JPN_Test2_Plus'
	files = ['hparams.yaml','step.100000.ckpt','japanese-extension-sofa.txt']
	for file in files:
		download_file(f'{url}/{file}', file)
	
	logger.info('Sucessfully donwloaded models.')
	logger.info('Moving files...')
	_files = ['hparams.yaml','model.ckpt','dict.txt']
	for i, file in enumerate(files):
		if i == 2:
			logger.info('Setup dict.txt')
			lines = []
			with open(file, 'r') as f:
				lines = f.readlines()
			
			outlines = []
			strs = {}
			for line in lines:
				str_ = line.split('\t')[1]
				strs[str_]=str_
			with open(f'{folder}/JPN_Romaji_Test2_Plus/{_files[i]}', 'w') as f:
				for str_ in strs:
					f.write(f'{str_}\t{str_}\r\n')
			logger.info('Sucessfully set up dict.txt')
		
		os.rename(files[i], f'{folder}/JPN_Romaji_Test2_Plus/{_files[i]}')
	logger.info('Done setting up add_some_assets for LabelMakr.')

def JP_g2p_asset():
	logger.info('SetUp JP_g2p_asset for LabelMakr.')
	logger.info('Downloading pyopenjtalk-plus models.')

	url = 'https://github.com/CjangCjengh/japanese_g2p/releases/download/v1.0.0/japanese_g2p.zip'
	filepath = 'japanese_g2p.zip'
	download_file(url, filepath)
	logger.info('Sucessfully donwloaded models.')
	logger.info('Unzipping...')
	with zipfile.ZipFile(filepath, 'r') as archive:
		archive.extractall('./')
	os.rename('japanese_g2p', f'g2p-jp')
	os.remove(filepath)
	logger.info('Done setting up JP_g2p_asset for LabelMakr.')
install_sofa_asset()
spicytigermeat_asset()
add_some_assets()
JP_g2p_asset()

logger.info('Successfully downloaded models. You may exit this window.')