import sys
import pathlib
import torch
import yaml
from pathlib import Path as P
MODULE_ROOT = P(__file__).resolve().parent
SOFA_ROOT = MODULE_ROOT / 'SOFA'
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

	from SOFA.modules import AP_detector as sofa_ap_detector, g2p as sofa_g2p
	try:
		from SOFA.modules.task.forced_alignment import LitForcedAlignmentTask as sofa_task_class
	except ImportError:
		from SOFA.train import LitForcedAlignmentTask as sofa_task_class

	try:
		from SOFA.infer import save_htk as sofa_save_htk, save_textgrids as sofa_save_textgrids, post_processing as sofa_post_processing, fill_small_gaps as sofa_fill_small_gaps, add_SP as sofa_add_sp
		sofa_exporter_class = None
	except ImportError:
		from SOFA.modules.utils.export_tool import Exporter as sofa_exporter_class
		from SOFA.modules.utils.post_processing import post_processing as sofa_post_processing, fill_small_gaps as sofa_fill_small_gaps, add_SP as sofa_add_sp
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
							  g2p_model: str = None,
							  g2p_cfg: str = None):
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
			   g2p_model: str = None,
			   g2p_cfg: str = None):

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
	dataset = grapheme_to_phoneme.get_dataset(P('corpus').rglob('*.wav'))

	# load model
	torch.set_grad_enabled(False)
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
			save_textgrids(predictions)
		elif op_format == 'htk':
			save_htk(predictions)

if __name__ == "__main__":
	print("What u doin silly!")