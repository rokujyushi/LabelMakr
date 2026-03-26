import argparse
import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path as P


logger = logging.getLogger(__name__)
logging.basicConfig(format="| %(levelname)s | %(message)s | %(asctime)s |", datefmt="%H:%M:%S")
logger.setLevel(logging.INFO)

SHARED_ROOT = P(__file__).resolve().parent
WORKSPACE_ROOT = SHARED_ROOT.parent
RUNTIME_A_ROOT = WORKSPACE_ROOT / 'runtime_a'
RUNTIME_B_ROOT = WORKSPACE_ROOT / 'runtime_b'
SHARED_MODELS_ROOT = RUNTIME_B_ROOT / 'models'
RUNTIME_A_G2P_ROOT = RUNTIME_A_ROOT / 'g2p-jp'
RUNTIME_B_SOFA_ROOT = RUNTIME_B_ROOT / 'SOFA'
RUNTIME_B_ONNX_ROOT = RUNTIME_B_ROOT / 'onnx_model'


def parse_args():
	parser = argparse.ArgumentParser(description='Install shared/runtime assets for the 3-runtime LabelMakr layout.')
	parser.add_argument('--skip-ffmpeg', action='store_true')
	parser.add_argument('--skip-sofa', action='store_true')
	parser.add_argument('--skip-models', action='store_true')
	parser.add_argument('--skip-g2p', action='store_true')
	parser.add_argument('--skip-pydomino-onnx', action='store_true')
	return parser.parse_args()


def download_file(url, filepath: P):
	logger.info(f'Downloading {url}')
	filepath.parent.mkdir(parents=True, exist_ok=True)
	with urllib.request.urlopen(url) as response, open(filepath, 'wb') as file:
		shutil.copyfileobj(response, file)


def replace_directory(source: P, target: P):
	if target.exists():
		shutil.rmtree(target)
	target.parent.mkdir(parents=True, exist_ok=True)
	shutil.move(str(source), str(target))


def install_ffmpeg_shared_asset():
	logger.info('Set up shared FFmpeg for LabelMakr.')
	url = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.0-latest-win64-gpl-shared-8.0.zip'
	with tempfile.TemporaryDirectory(prefix='labelmakr_ffmpeg_') as temp_dir:
		temp_root = P(temp_dir)
		archive_path = temp_root / 'ffmpeg.zip'
		extract_dir = temp_root / 'extract'
		download_file(url, archive_path)
		extract_dir.mkdir(parents=True, exist_ok=True)

		logger.info('Extracting shared FFmpeg...')
		with zipfile.ZipFile(archive_path, 'r') as archive:
			archive.extractall(extract_dir)

		bin_dirs = [candidate.parent for candidate in extract_dir.rglob('ffmpeg.exe')]
		if not bin_dirs:
			raise RuntimeError('Shared FFmpeg archive did not contain ffmpeg.exe.')

		for source in bin_dirs[0].iterdir():
			name = source.name.lower()
			if source.suffix.lower() == '.dll' or name in {'ffmpeg.exe', 'ffprobe.exe'}:
				shutil.copy2(source, SHARED_ROOT / source.name)

	logger.info('Done setting up shared FFmpeg for LabelMakr.')


def install_sofa_asset():
	logger.info('Set up SOFA for runtime_b.')
	url = 'https://github.com/qiuqiao/SOFA/archive/refs/heads/main.zip'
	with tempfile.TemporaryDirectory(prefix='labelmakr_sofa_') as temp_dir:
		temp_root = P(temp_dir)
		archive_path = temp_root / 'SOFA-main.zip'
		extract_dir = temp_root / 'extract'
		download_file(url, archive_path)
		extract_dir.mkdir(parents=True, exist_ok=True)

		logger.info('Unzipping SOFA...')
		with zipfile.ZipFile(archive_path, 'r') as archive:
			archive.extractall(extract_dir)

		extracted = [path for path in extract_dir.iterdir() if path.is_dir()]
		if not extracted:
			raise RuntimeError('SOFA archive could not be extracted.')

		replace_directory(extracted[0], RUNTIME_B_SOFA_ROOT)
		
	logger.info('Done setting up SOFA for runtime_b.')


def install_shared_models_asset():
	logger.info('Set up shared SOFA models for LabelMakr.')
	url = 'https://github.com/spicytigermeat/LabelMakr/releases/download/assets_v030/models.zip'
	with tempfile.TemporaryDirectory(prefix='labelmakr_models_') as temp_dir:
		temp_root = P(temp_dir)
		archive_path = temp_root / 'models.zip'
		download_file(url, archive_path)

		SHARED_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
		with zipfile.ZipFile(archive_path, 'r') as archive:
			archive.extractall(SHARED_MODELS_ROOT)

	logger.info('Done setting up shared SOFA models for LabelMakr.')


def install_extra_jpn_model_asset():
	logger.info('Set up additional JPN_Romaji_Test2_Plus model.')
	url = 'https://github.com/Greenleaf2001/SOFA_Models/releases/download/JPN_Test2_Plus'
	target_dir = SHARED_MODELS_ROOT / 'JPN_Romaji_Test2_Plus'
	target_dir.mkdir(parents=True, exist_ok=True)
	files = ['hparams.yaml', 'step.100000.ckpt', 'japanese-extension-sofa.txt']
	output_files = ['hparams.yaml', 'model.ckpt', 'dict.txt']

	with tempfile.TemporaryDirectory(prefix='labelmakr_jpn_model_') as temp_dir:
		temp_root = P(temp_dir)
		for file in files:
			download_file(f'{url}/{file}', temp_root / file)

		for index, file in enumerate(files):
			source_file = temp_root / file
			target_file = target_dir / output_files[index]
			if index == 2:
				lines = source_file.read_text(encoding='utf-8').splitlines()
				unique_entries = {}
				for line in lines:
					parts = line.split('\t')
					if len(parts) > 1:
						unique_entries[parts[1]] = parts[1]
				target_file.write_text(''.join(f'{text}\t{text}\r\n' for text in unique_entries), encoding='utf-8')
				continue

			if target_file.exists():
				target_file.unlink()
			shutil.move(str(source_file), str(target_file))

	logger.info('Done setting up additional JPN_Romaji_Test2_Plus model.')


def install_japanese_g2p_asset():
	logger.info('Set up japanese_g2p for runtime_a.')
	url = 'https://github.com/CjangCjengh/japanese_g2p/releases/download/v1.0.0/japanese_g2p.zip'
	with tempfile.TemporaryDirectory(prefix='labelmakr_g2p_') as temp_dir:
		temp_root = P(temp_dir)
		archive_path = temp_root / 'japanese_g2p.zip'
		download_file(url, archive_path)
		with zipfile.ZipFile(archive_path, 'r') as archive:
			archive.extractall(temp_root)
		replace_directory(temp_root / 'japanese_g2p', RUNTIME_A_G2P_ROOT)

	logger.info('Done setting up japanese_g2p for runtime_a.')


def install_pydomino_onnx_asset():
	logger.info('Set up pydomino ONNX model for runtime_b.')
	url = 'https://raw.githubusercontent.com/DwangoMediaVillage/pydomino/main/onnx_model/phoneme_transition_model.onnx'
	RUNTIME_B_ONNX_ROOT.mkdir(parents=True, exist_ok=True)
	download_file(url, RUNTIME_B_ONNX_ROOT / 'phoneme_transition_model.onnx')
	logger.info('Done setting up pydomino ONNX model for runtime_b.')


def main():
	args = parse_args()
	SHARED_MODELS_ROOT.mkdir(parents=True, exist_ok=True)

	if not args.skip_ffmpeg:
		install_ffmpeg_shared_asset()
	if not args.skip_sofa:
		install_sofa_asset()
	if not args.skip_models:
		install_shared_models_asset()
		# install_extra_jpn_model_asset()
	if not args.skip_g2p:
		install_japanese_g2p_asset()
	if not args.skip_pydomino_onnx:
		install_pydomino_onnx_asset()

	logger.info('Successfully downloaded LabelMakr assets.')


if __name__ == '__main__':
	main()