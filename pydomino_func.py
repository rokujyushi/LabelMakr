import logging
import site
import shutil
import subprocess
import sys
import wave
from importlib import import_module
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)
logging.basicConfig(format="| %(levelname)s | %(message)s | %(asctime)s |",
					datefmt="%H:%M:%S")
logger.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent
PYDOMINO_MODEL_FILE_NAME = 'phoneme_transition_model.onnx'
HTK_100NS_PER_SECOND = 10_000_000
PYDOMINO_TARGET_SAMPLE_RATE = 16000


def seconds_to_htk_100ns(value_seconds):
	return int(round(float(value_seconds) * HTK_100NS_PER_SECOND))


def resolve_ffprobe_executable():
	for candidate in (
		PROJECT_ROOT / 'ffprobe.exe',
		PROJECT_ROOT.parent / 'ffprobe.exe',
	):
		if candidate.exists():
			return candidate

	ffprobe_path = shutil.which('ffprobe')
	if ffprobe_path:
		return Path(ffprobe_path)

	return None


def resolve_ffmpeg_executable():
	for candidate in (
		PROJECT_ROOT / 'ffmpeg.exe',
		PROJECT_ROOT.parent / 'ffmpeg.exe',
	):
		if candidate.exists():
			return candidate

	ffmpeg_path = shutil.which('ffmpeg')
	if ffmpeg_path:
		return Path(ffmpeg_path)

	raise FileNotFoundError('ffmpeg executable could not be found.')


def decode_audio_with_ffmpeg(audio_path):
	ffmpeg_exe = resolve_ffmpeg_executable()
	result = subprocess.run(
		[
			str(ffmpeg_exe),
			'-v',
			'error',
			'-i',
			str(audio_path),
			'-f',
			'f32le',
			'-acodec',
			'pcm_f32le',
			'-ac',
			'1',
			'-ar',
			'16000',
			'-',
		],
		capture_output=True,
		check=True,
	)
	waveform = np.frombuffer(result.stdout, dtype=np.float32)
	if waveform.size == 0:
		raise ValueError(f'ffmpeg returned no decoded audio for: {audio_path}')
	return np.clip(waveform, -1.0, 1.0).astype(np.float32)


def probe_audio_duration_seconds(audio_path):
	ffprobe_exe = resolve_ffprobe_executable()
	if ffprobe_exe is None:
		return None

	try:
		result = subprocess.run(
			[
				str(ffprobe_exe),
				'-v',
				'error',
				'-show_entries',
				'format=duration',
				'-of',
				'default=nokey=1:noprint_wrappers=1',
				str(audio_path),
			],
			capture_output=True,
			text=True,
			check=True,
		)
		return float(result.stdout.strip())
	except (subprocess.CalledProcessError, ValueError):
		return None


def get_duration_scale(source_duration_seconds, waveform):
	if source_duration_seconds is None or waveform.size == 0:
		return 1.0

	converted_duration_seconds = float(waveform.size) / float(PYDOMINO_TARGET_SAMPLE_RATE)
	if converted_duration_seconds <= 0:
		return 1.0

	return float(source_duration_seconds) / converted_duration_seconds


def scale_alignment_times(alignment, duration_scale):
	if abs(duration_scale - 1.0) < 1e-6:
		return alignment

	return [
		(begin_sec * duration_scale, end_sec * duration_scale, phoneme)
		for begin_sec, end_sec, phoneme in alignment
	]


def get_aligner_class():
	try:
		return import_module('pydomino').Aligner
	except ImportError:
		return import_module('pydomino.pydomino').Aligner


def find_pydomino_model_paths():
	model_paths = {}
	candidate_dirs = []
	candidate_dirs.extend([
		PROJECT_ROOT,
		PROJECT_ROOT / 'python',
		PROJECT_ROOT / 'models',
		PROJECT_ROOT / 'assets',
	])

	for module_name in ('pydomino', 'pydomino.pydomino'):
		try:
			module = import_module(module_name)
		except ImportError:
			continue

		module_file = getattr(module, '__file__', None)
		if module_file:
			module_dir = Path(module_file).resolve().parent
			candidate_dirs.extend([module_dir, module_dir.parent])

	for site_dir in site.getsitepackages():
		candidate_dirs.append(Path(site_dir))

	user_site = site.getusersitepackages()
	if user_site:
		candidate_dirs.append(Path(user_site))

	candidate_dirs.append(Path(sys.prefix))

	seen = set()
	for base_dir in candidate_dirs:
		if base_dir in seen:
			continue
		seen.add(base_dir)

		for candidate in (
			base_dir / 'onnx_model' / PYDOMINO_MODEL_FILE_NAME,
			base_dir / 'pydomino' / 'onnx_model' / PYDOMINO_MODEL_FILE_NAME,
			base_dir / PYDOMINO_MODEL_FILE_NAME,
		):
			if candidate.exists():
				model_paths[candidate.stem] = {'onnx_path': candidate}

		for onnx_dir in (
			base_dir / 'onnx_model',
			base_dir / 'pydomino' / 'onnx_model',
		):
			if not onnx_dir.exists():
				continue
			for candidate in sorted(onnx_dir.glob('*.onnx')):
				model_paths[candidate.stem] = {'onnx_path': candidate}

	return model_paths


def load_wav_mono_16khz(wav_path):
	path = Path(wav_path)
	try:
		with wave.open(str(path), 'rb') as wav_file:
			channels = wav_file.getnchannels()
			sample_rate = wav_file.getframerate()
			sample_width = wav_file.getsampwidth()
			frame_count = wav_file.getnframes()
			raw_frames = wav_file.readframes(frame_count)
	except (wave.Error, EOFError) as exc:
		logger.info(f'Falling back to ffmpeg decoding for {path}: {exc}')
		waveform = decode_audio_with_ffmpeg(path)
		return waveform, get_duration_scale(probe_audio_duration_seconds(path), waveform)

	source_duration_seconds = frame_count / sample_rate if sample_rate > 0 else None

	if channels != 1 or sample_rate != PYDOMINO_TARGET_SAMPLE_RATE:
		logger.info(f'Converting audio to mono/16kHz with ffmpeg: {path}')
		waveform = decode_audio_with_ffmpeg(path)
		return waveform, get_duration_scale(source_duration_seconds, waveform)

	if sample_width == 1:
		waveform = (np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
	elif sample_width == 2:
		waveform = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
	elif sample_width == 4:
		waveform = np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32) / 2147483648.0
	else:
		logger.info(f'Converting unsupported WAV sample width with ffmpeg: {path} ({sample_width} bytes)')
		waveform = decode_audio_with_ffmpeg(path)
		return waveform, get_duration_scale(source_duration_seconds, waveform)

	return np.clip(waveform, -1.0, 1.0).astype(np.float32), 1.0


def load_phonemes(phoneme_source):
	path = Path(phoneme_source)
	if path.exists():
		strs = path.read_text(encoding='utf-8').split()
		# Normalize unknown phonemes to 'pau'
		support_phones = {'pau','ry','r','my','m','ny','n','j','z','by','b','dy','d','gy','g','ky','k','ch','ts','sh','s','hy','h','v','f','py','p','t','y','w','n','a','i','u','e','o','cl','I','U','N'}
		for i, phoneme in enumerate(strs):
			if phoneme not in support_phones:
				strs[i] = 'pau'  # Normalize unknown phonemes to 'pau'
		
		strs.insert(0, 'pau')  # Add 'pau' at the beginning of the phoneme list
		strs.append('pau')  # Add 'pau' at the end of the phoneme list
		return strs
	return str(phoneme_source).split()


def write_lab(alignment, output_path):
	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', encoding='utf-8', newline='\n') as handle:
		for begin_sec, end_sec, phoneme in alignment:
			begin_100ns = seconds_to_htk_100ns(begin_sec)
			end_100ns = seconds_to_htk_100ns(end_sec)
			handle.write(f'{begin_100ns} {end_100ns} {phoneme}\n')


def write_textgrid(alignment, phonemes, output_path):
	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)

	non_zero_alignment = [item for item in alignment if item[1] - item[0] > 0]
	max_time = 0.0
	if alignment:
		max_time = max(end_sec for _, end_sec, _ in alignment)

	with path.open('w', encoding='utf-8', newline='\n') as handle:
		handle.write('File type = "ooTextFile"\n')
		handle.write('Object class = "TextGrid"\n\n')
		handle.write(f'xmin = 0\n')
		handle.write(f'xmax = {max_time:.3f}\n')
		handle.write('tiers? <exists>\n')
		handle.write('size = 2\n')
		handle.write('item []:\n')
		handle.write('\titem [1]:\n')
		handle.write('\t\tclass = "IntervalTier"\n')
		handle.write('\t\tname = "phonemes"\n')
		handle.write('\t\txmin = 0\n')
		handle.write(f'\t\txmax = {max_time:.3f}\n')
		handle.write('\t\tintervals: size = 1\n')
		handle.write('\t\tintervals [1]:\n')
		handle.write('\t\t\txmin = 0\n')
		handle.write(f'\t\t\txmax = {max_time:.3f}\n')
		handle.write(f'\t\t\ttext = "{" ".join(phonemes)}"\n')
		handle.write('\titem [2]:\n')
		handle.write('\t\tclass = "IntervalTier"\n')
		handle.write('\t\tname = "alignment result"\n')
		handle.write('\t\txmin = 0\n')
		handle.write(f'\t\txmax = {max_time:.3f}\n')
		handle.write(f'\t\tintervals: size = {len(non_zero_alignment)}\n')

		for index, (begin_sec, end_sec, phoneme) in enumerate(non_zero_alignment, start=1):
			handle.write(f'\t\tintervals [{index}]:\n')
			handle.write(f'\t\t\txmin = {begin_sec:.3f}\n')
			handle.write(f'\t\t\txmax = {end_sec:.3f}\n')
			handle.write(f'\t\t\ttext = "{phoneme}"\n')


def get_output_path(wav_path, output_format):
	path = Path(wav_path)
	labels_dir = path.parent / 'labels'
	labels_dir.mkdir(parents=True, exist_ok=True)
	if output_format == 'TextGrid':
		return labels_dir / f'{path.stem}.TextGrid'
	return labels_dir / f'{path.stem}.lab'


def infer_pydomino(onnx_path, output_format='htk', min_aligned_timeframe=3, corpus_dir='corpus'):
	aligner = pydomino_func(onnx_path)
	corpus_path = Path(corpus_dir)
	try:
		for wav_path in corpus_path.rglob('*.wav'):
			phoneme_path = wav_path.with_suffix('.txt')
			if not phoneme_path.exists():
				logger.warning(f'Skipping {wav_path}: missing phoneme file {phoneme_path.name}')
				continue

			phonemes = load_phonemes(phoneme_path)
			alignment = aligner.align(wav_path, phoneme_path, min_aligned_timeframe=min_aligned_timeframe)
			output_path = get_output_path(wav_path, output_format)

			if output_format == 'TextGrid':
				write_textgrid(alignment, phonemes, output_path)
			else:
				write_lab(alignment, output_path)

			logger.info(f'Wrote alignment: {output_path}')
	finally:
		aligner.release()


class pydomino_func:
	def __init__(self, onnx_path):
		self.onnx_path = str(onnx_path)
		self.aligner = get_aligner_class()(self.onnx_path)

	def align(self, wav_path, phoneme_source, min_aligned_timeframe=3):
		waveform, duration_scale = load_wav_mono_16khz(wav_path)
		phonemes = load_phonemes(phoneme_source)
		logger.info(f'Running pydomino alignment for {wav_path}.')
		alignment = self.aligner.align(waveform, ' '.join(phonemes), min_aligned_timeframe)
		return scale_alignment_times(alignment, duration_scale)

	def align_to_lab(self, wav_path, phoneme_source, output_path, min_aligned_timeframe=3):
		alignment = self.align(wav_path, phoneme_source, min_aligned_timeframe=min_aligned_timeframe)
		write_lab(alignment, output_path)
		logger.info(f'Wrote alignment: {output_path}')
		return alignment

	def release(self):
		self.aligner.release()


if __name__ == '__main__':
	print('Import this module and call pydomino_func(onnx_path).align(...)')
