import os
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path as P


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
	logger.info(f'Downloading {url}')
	with urllib.request.urlopen(url) as response, open(filepath, 'wb') as file:
		shutil.copyfileobj(response, file)


def replace_directory(source: P, target: P):
	if target.exists():
		shutil.rmtree(target)
	shutil.move(str(source), str(target))


def install_ffmpeg_shared_asset():
	logger.info('SetUp shared FFmpeg for LabelMakr.')
	filepath = 'ffmpeg-n8.0-latest-win64-gpl-shared-8.0.zip'
	extract_dir = P('./_ffmpeg_extract')

	if extract_dir.exists():
		shutil.rmtree(extract_dir)

	url = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.0-latest-win64-gpl-shared-8.0.zip'
	download_file(url, filepath)
	extract_dir.mkdir(exist_ok=True)

	logger.info('Extracting shared FFmpeg...')
	with zipfile.ZipFile(filepath, 'r') as archive:
		archive.extractall(extract_dir)

	bin_dirs = [candidate.parent for candidate in extract_dir.rglob('ffmpeg.exe')]
	if not bin_dirs:
		raise RuntimeError('Shared FFmpeg archive did not contain ffmpeg.exe.')

	bin_dir = bin_dirs[0]
	for source in bin_dir.iterdir():
		name = source.name.lower()
		if source.suffix.lower() == '.dll' or name in {'ffmpeg.exe', 'ffprobe.exe'}:
			shutil.copy2(source, P('./') / source.name)

	shutil.rmtree(extract_dir)
	os.remove(filepath)
	logger.info('Done setting up shared FFmpeg for LabelMakr.')


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

	url = 'https://github.com/Greenleaf2001/SOFA_Models/releases/download/JPN_Test2_Plus'
	target_dir = folder / 'JPN_Romaji_Test2_Plus'
	target_dir.mkdir(parents=True, exist_ok=True)
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
			with open(file, 'r', encoding='utf-8') as f:
				lines = f.readlines()
			
			strs = {}
			for line in lines:
				str_ = line.split('\t')[1]
				strs[str_]=str_
			with open(target_dir / _files[i], 'w', encoding='utf-8') as f:
				for str_ in strs:
					f.write(f'{str_}\t{str_}\r\n')
			logger.info('Sucessfully set up dict.txt')
			os.remove(file)
			continue
		
		target_file = target_dir / _files[i]
		if target_file.exists():
			os.remove(target_file)
		os.replace(files[i], target_file)
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
	replace_directory(P('japanese_g2p'), P('g2p-jp'))
	os.remove(filepath)
	logger.info('Done setting up JP_g2p_asset for LabelMakr.')
install_ffmpeg_shared_asset()
install_sofa_asset()
spicytigermeat_asset()
add_some_assets()
JP_g2p_asset()

logger.info('Successfully downloaded models. You may exit this window.')