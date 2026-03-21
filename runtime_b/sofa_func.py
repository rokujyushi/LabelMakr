import sys
import pathlib
import os
from importlib import import_module
from typing import Optional
import torch
import yaml
from pathlib import Path as P
MODULE_ROOT = P(__file__).resolve().parent


def require_env_path(env_name):
	value = os.environ.get(env_name)
	if not value:
		raise RuntimeError(f'{env_name} is not set. Direct execution is unsupported; launch through runtime_b/worker.py.')
	return P(value).resolve()


WORKSPACE_ROOT = require_env_path('LABELMAKR_WORKSPACE_ROOT')
SHARED_ROOT = require_env_path('LABELMAKR_SHARED_ROOT')
CORPUS_ROOT = require_env_path('LABELMAKR_CORPUS_DIR')
SOFA_ROOT = require_env_path('LABELMAKR_SOFA_ROOT')
import lightning as pl
import textgrid
import logging

AP_detector = None
g2p = None
LitForcedAlignmentTask = None
SOFA_EXPORTER_CLASS = None
save_htk = None
save_textgrids = None
post_processing = None
fill_small_gaps = None
add_SP = None

DEBUG = True

# logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(format="| %(levelname)s | %(message)s | %(asctime)s |",
					datefmt="%H:%M:%S")
if DEBUG:
	logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)


def load_sofa_dependencies():
	global AP_detector, g2p, LitForcedAlignmentTask
	global SOFA_EXPORTER_CLASS, save_htk, save_textgrids, post_processing, fill_small_gaps, add_SP

	if AP_detector is not None and g2p is not None and LitForcedAlignmentTask is not None and post_processing is not None:
		return

	if not SOFA_ROOT.exists():
		raise ModuleNotFoundError(f'SOFA assets were not found: {SOFA_ROOT}')

	sofa_root_str = str(SOFA_ROOT)
	if sofa_root_str not in sys.path:
		sys.path.append(sofa_root_str)

	sofa_modules = import_module('SOFA.modules')
	sofa_ap_detector = getattr(sofa_modules, 'AP_detector')
	sofa_g2p = getattr(sofa_modules, 'g2p')
	try:
		sofa_task_module = import_module('SOFA.modules.task.forced_alignment')
		sofa_task_class = getattr(sofa_task_module, 'LitForcedAlignmentTask')
	except ImportError:
		sofa_train_module = import_module('SOFA.train')
		sofa_task_class = getattr(sofa_train_module, 'LitForcedAlignmentTask')

	try:
		sofa_infer_module = import_module('SOFA.infer')
		sofa_save_htk = getattr(sofa_infer_module, 'save_htk')
		sofa_save_textgrids = getattr(sofa_infer_module, 'save_textgrids')
		sofa_post_processing = getattr(sofa_infer_module, 'post_processing')
		sofa_fill_small_gaps = getattr(sofa_infer_module, 'fill_small_gaps')
		sofa_add_sp = getattr(sofa_infer_module, 'add_SP')
		sofa_exporter_class = None
	except ImportError:
		sofa_export_tool_module = import_module('SOFA.modules.utils.export_tool')
		sofa_post_module = import_module('SOFA.modules.utils.post_processing')
		sofa_exporter_class = getattr(sofa_export_tool_module, 'Exporter')
		sofa_post_processing = getattr(sofa_post_module, 'post_processing')
		sofa_fill_small_gaps = getattr(sofa_post_module, 'fill_small_gaps')
		sofa_add_sp = getattr(sofa_post_module, 'add_SP')
		sofa_save_htk = None
		sofa_save_textgrids = None

	AP_detector = sofa_ap_detector
	g2p = sofa_g2p
	LitForcedAlignmentTask = sofa_task_class
	SOFA_EXPORTER_CLASS = sofa_exporter_class
	save_htk = sofa_save_htk
	save_textgrids = sofa_save_textgrids
	post_processing = sofa_post_processing
	fill_small_gaps = sofa_fill_small_gaps
	add_SP = sofa_add_sp

def build_grapheme_to_phoneme(dictionary: str,
							  g2p_bool: bool = False,
							  g2p_model: Optional[str] = None,
							  g2p_cfg: Optional[str] = None):
	load_sofa_dependencies()

	if g2p_bool:
		g2p_class = getattr(g2p, 'OovG2P', None)
		if g2p_class is not None:
			logger.debug('Loaded OovG2P backend.')
			return g2p_class(dictionary=dictionary,
							 g2p_model=g2p_model,
							 g2p_cfg=g2p_cfg)
		logger.warning('Installed SOFA does not expose OovG2P. Falling back to DictionaryG2P; unknown words will be skipped.')

	g2p_class = getattr(g2p, 'DictionaryG2P')
	logger.debug('Loaded DictionaryG2P backend.')
	return g2p_class(dictionary=dictionary)

def infer_sofa(ckpt: str,
			   dictionary: str,
			   op_format: str = 'htk',
			   matmul_bool: bool = False,
			   lang: str = 'EN',
			   g2p_bool: bool = False,
			   g2p_model: Optional[str] = None,
			   g2p_cfg: Optional[str] = None):

	#
	# Much of this code was referenced from "infer.py"
	# in the SOFA source code! :)
	#
	load_sofa_dependencies()

	logger.info('Running SOFA inference.')

	# determine the torch device to use.
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# use tensorcores, if selected!
	if matmul_bool:
		logger.debug('Setting matmul precision to medium.')
		torch.set_float32_matmul_precision('medium')

	# set up the G2P, based on the language selected.
	grapheme_to_phoneme = build_grapheme_to_phoneme(dictionary=dictionary,
										 g2p_bool=g2p_bool,
										 g2p_model=g2p_model,
										 g2p_cfg=g2p_cfg)
	grapheme_to_phoneme.set_in_format('txt')

	# set up the AP Detector
	AP_detector_class = getattr(AP_detector, 'LoudnessSpectralcentroidAPDetector')
	get_AP = AP_detector_class()

	# load up the dataset
	dataset = grapheme_to_phoneme.get_dataset(CORPUS_ROOT.rglob('*.wav'))

	# load model
	torch.set_grad_enabled(False)
	assert LitForcedAlignmentTask is not None
	model = LitForcedAlignmentTask.load_from_checkpoint(ckpt, map_location=device)
	model.set_inference_mode('force')
	trainer = pl.Trainer(logger=False)

	# run predictions
	try:
		predictions = trainer.predict(model, dataloaders=dataset, return_predictions=True)
	except IndexError as e:
		print(f"\nOne or more of your transcriptions are causing issues, please correct them with the transcription editor! {e}")
	try:
		predictions = get_AP.process(predictions)
	except TypeError as e:
		print(f"\n Error in one or more transcriptions, please correct them with the transcription editor!\n\n Error Code: {e}\n\n")
	assert post_processing is not None
	predictions = post_processing(predictions)

	# output
	if SOFA_EXPORTER_CLASS is not None:
		processed_predictions, log = predictions
		exporter = SOFA_EXPORTER_CLASS(processed_predictions, log)
		if op_format == 'TextGrid':
			exporter.export(['textgrid'])
		elif op_format == 'htk':
			exporter.export(['htk'])
	else:
		if op_format == 'TextGrid':
			assert save_textgrids is not None
			save_textgrids(predictions)
		elif op_format == 'htk':
			assert save_htk is not None
			save_htk(predictions)

if __name__ == "__main__":
	raise SystemExit('Direct execution is unsupported; launch through runtime_b/worker.py.')