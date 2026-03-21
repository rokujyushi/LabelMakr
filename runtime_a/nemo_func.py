import os
import glob
import logging
import re
import subprocess
import tempfile
from pathlib import Path as P

from ftfy import fix_text as fxy
from pypinyin import lazy_pinyin
from shared.nemo_metadata import (
	DEFAULT_NEMO_MODEL,
	NEMO_MODELS,
	supports_language,
)

MODULE_ROOT = P(__file__).resolve().parent


def require_env_path(env_name):
	value = os.environ.get(env_name)
	if not value:
		raise RuntimeError(f'{env_name} is not set. Direct execution is unsupported; launch through runtime_a/worker.py.')
	return P(value).resolve()


WORKSPACE_ROOT = require_env_path('LABELMAKR_WORKSPACE_ROOT')
SHARED_ROOT = require_env_path('LABELMAKR_SHARED_ROOT')
CORPUS_ROOT = require_env_path('LABELMAKR_CORPUS_DIR')


def resolve_ffmpeg_executable():
	ffmpeg_path = os.environ.get('LABELMAKR_FFMPEG_EXE')
	if not ffmpeg_path:
		raise RuntimeError('LABELMAKR_FFMPEG_EXE is not set. Launch transcription through runtime_a/worker.py.')
	return P(ffmpeg_path).resolve()


def resolve_g2p_jp_executable():
	g2p_path = os.environ.get('LABELMAKR_G2P_JP_EXE')
	if not g2p_path:
		raise RuntimeError('LABELMAKR_G2P_JP_EXE is not set. Launch transcription through runtime_a/worker.py.')
	return P(g2p_path).resolve()

try:
	from g2pk import G2p as G2pK
except ImportError:
	G2pK = None

try:
	import torch
except ImportError:
	torch = None

try:
	from huggingface_hub import hf_hub_download
except ImportError:
	hf_hub_download = None

try:
	import nemo.collections.asr as nemo_asr
except ImportError:
	nemo_asr = None


def get_runtime_notes(model_id):
	notes = [
		'NeMo is officially unsupported on Windows in the current support matrix; use on Windows is best-effort.',
		'NeMo ASR models require online access on first use to download weights from Hugging Face.',
	]

	if model_id in ['reazonspeech_nemo_v2', 'hiragana_parakeet_ja_beta']:
		notes.append('Japanese `.nemo` checkpoints are about 2.5 GB and are better suited to GPU or high-memory CPU environments.')
	if model_id == 'parakeet_tdt_0_6b_v2_en':
		notes.append('The English Parakeet model is optimized for NVIDIA GPU systems; Linux is the preferred OS in the model card.')
		notes.append('The English Parakeet model card mentions at least 2 GB RAM just to load the model, and more for longer audio.')

	return notes


def log(debug=False):
	logger = logging.getLogger(__name__)

	logging.basicConfig(format="| %(levelname)s | %(message)s | %(asctime)s |",
						datefmt="%H:%M:%S")

	if debug:
		logger.setLevel(logging.DEBUG)
	logger.setLevel(logging.INFO)

	return logger


class NeMoTranscriber(object):
	def __init__(self, lang, model_id=DEFAULT_NEMO_MODEL, force_cpu=False):
		super().__init__()

		self.log = log()
		self.lang = lang
		self.model_id = model_id
		self.force_cpu = force_cpu
		self.g2pk = G2pK() if G2pK is not None else None
		self.fr_contraction = ["m'", "n'", "l'", "j'", "c'", "ç'", "s'", "t'", "d'", "qu'"]
		self.model = self._load_model()

	def _get_torch_device(self):
		if self.force_cpu or torch is None or not torch.cuda.is_available():
			return 'cpu'
		return 'cuda'

	def _load_model(self):
		if self.model_id not in NEMO_MODELS:
			raise ValueError(f'Unsupported NeMo model: {self.model_id}')

		if not supports_language(self.model_id, self.lang):
			raise ValueError(f'{NEMO_MODELS[self.model_id]["label"]} does not support language {self.lang}.')

		if nemo_asr is None or hf_hub_download is None:
			raise ImportError('NeMo transcription requires `nemo_toolkit[asr]` and `huggingface_hub`. Install the optional dependencies from requirements_nemo.txt.')

		if os.name == 'nt':
			self.log.warning('NeMo is officially unsupported on Windows. Running in best-effort mode.')

		config = NEMO_MODELS[self.model_id]
		device_name = self._get_torch_device()

		if config['loader'] == 'restore_from':
			self.log.info(f'Downloading NeMo model from {config["repo_id"]} if needed...')
			model_path = hf_hub_download(repo_id=config['repo_id'], filename=config['filename'])
			map_location = torch.device(device_name) if torch is not None else device_name
			model = nemo_asr.models.ASRModel.restore_from(model_path, map_location=map_location)
		else:
			self.log.info(f'Loading pretrained NeMo model {config["model_name"]}...')
			model = nemo_asr.models.ASRModel.from_pretrained(model_name=config['model_name'])

		model.freeze()
		if hasattr(model, 'to'):
			model = model.to(device_name)
		return model

	def _extract_text(self, transcription):
		item = transcription
		if isinstance(transcription, list):
			item = transcription[0] if transcription else ''

		if hasattr(item, 'text'):
			return item.text
		if isinstance(item, dict):
			return item.get('text', '')
		if isinstance(item, (list, tuple)):
			return self._extract_text(list(item))
		return str(item).strip()

	def _prepare_nemo_input(self, file_path):
		cache_root = (WORKSPACE_ROOT / '.cache' / 'runtime_a' / 'nemo_audio').resolve()
		cache_root.mkdir(parents=True, exist_ok=True)
		with tempfile.NamedTemporaryFile(dir=cache_root, suffix='.wav', delete=False) as temp_file:
			temp_path = P(temp_file.name)

		command = [
			str(resolve_ffmpeg_executable()),
			'-y',
			'-v',
			'error',
			'-i',
			str(P(file_path).resolve()),
			'-ac',
			'1',
			'-ar',
			'16000',
			str(temp_path),
		]
		try:
			subprocess.run(command, check=True, capture_output=True)
		except subprocess.CalledProcessError as exc:
			temp_path.unlink(missing_ok=True)
			stderr = exc.stderr.decode('utf-8', errors='ignore') if exc.stderr else ''
			raise RuntimeError(f'Failed to prepare NeMo input audio for {file_path}: {stderr.strip()}') from exc

		return temp_path

	def transcribe_file(self, file_path):
		prepared_path = self._prepare_nemo_input(file_path)
		try:
			transcriptions = self.model.transcribe([str(prepared_path)], batch_size=1)
			return self._extract_text(transcriptions)
		finally:
			prepared_path.unlink(missing_ok=True)

	def jpn_g2p(self, jpn):
		phonemes = subprocess.check_output([str(resolve_g2p_jp_executable()), '-rs', jpn.replace(' ', '')], shell=False)
		g2p_op = str(phonemes)
		fixed = re.sub(r"([aeiouAIEOUN])", r" \1 ", g2p_op[2:-5])
		fixed = re.sub("cl", "cl ", fixed)
		fixed = re.sub(r"[.!?,]", "", fixed)
		fixed = re.sub(" {2,}", " ", fixed)
		fixed = re.sub("A", "a", fixed)
		fixed = re.sub("I", "i", fixed)
		fixed = re.sub("U", "u", fixed)
		fixed = re.sub("E", "e", fixed)
		fixed = re.sub("O", "o", fixed)
		return fixed

	def run_transcription(self, lang):
		self.log.info(f'Transcription output root: {CORPUS_ROOT}')
		for file in glob.glob(str(CORPUS_ROOT / '**/*.wav'), recursive=True):
			try:
				out_name = P(file).with_suffix('.txt')
				out2_name = file[:-4] + "_Fixed" + '.txt'
				out3_name = file[:-4] + "_JP" + '.txt'
				file_flag = os.path.exists(out2_name)
				file_flag2 = lang.upper() == 'JP' and os.path.exists(out3_name)
				if file_flag2:
					print("skip：", out3_name)
					continue

				if not file_flag:
					transcribed_text = self.transcribe_file(file)
				else:
					with open(out2_name, 'r+', encoding='utf-8') as out:
						transcribed_text = out.read()

				if lang.upper() == 'JP':
					trns_str_kanjis = fxy(transcribed_text).splitlines()
					if not file_flag2:
						with open(out3_name, 'w+', encoding='utf-8') as out:
							out.write(transcribed_text)
						self.log.info(f'Saved raw JP transcription to {P(out3_name).resolve()}')

					trns_str = ''
					line_number = 0
					for trns_str_kanji in trns_str_kanjis:
						line_number = line_number + 1
						print(f"line：{line_number}")
						trns_str = trns_str + self.jpn_g2p(trns_str_kanji) + "\r\n"
				elif lang.upper() == 'ZH':
					hanzi_list = lazy_pinyin(re.sub(' ', '', fxy(transcribed_text)))
					trns_str = ""
					for word in hanzi_list:
						trns_str += f"{word} "
				elif lang.upper() == 'FR':
					trns_str = re.sub(r"[-]", " ", fxy(transcribed_text).lower())
					trns_str = re.sub(r"[A-Za-z0-9]+$", "", trns_str)
					for con in self.fr_contraction:
						trns_str = re.sub(f"{con}", f"{con} ", trns_str)
				elif lang.upper() == 'KO':
					if self.g2pk is None:
						raise ImportError('Korean transcription requires g2pK/konlpy/JPype1/python-mecab-ko to be installed.')
					trns_str = self.g2pk(fxy(transcribed_text))
				else:
					trns_str = fxy(transcribed_text).lower()

				trns_str = re.sub(r"[.,!?]", "", trns_str)

				with open(out_name, 'w+', encoding='utf-8') as out:
					out.write(trns_str)

				self.log.info(f'Saved transcription result to {out_name.resolve()}')
				self.log.info(f'Wrote transcription for {file} in corpus.')

			except RuntimeError as e:
				self.log.warning(f'Error in transcribing: {e}')

		self.log.info('Completed All Transcriptions')


if __name__ == "__main__":
	raise SystemExit('Direct execution is unsupported; launch through runtime_a/worker.py.')