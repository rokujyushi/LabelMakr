import os, sys
import ctypes
import logging
import subprocess
from pathlib import Path as P


APP_ROOT = P(__file__).resolve().parent
WORKSPACE_ROOT = APP_ROOT.parent if APP_ROOT.name == 'gui' else APP_ROOT
GUI_ROOT = APP_ROOT if (APP_ROOT / 'labbu_func.py').exists() else WORKSPACE_ROOT / 'gui'
RUNTIME_A_ROOT = WORKSPACE_ROOT / 'runtime_a'
RUNTIME_B_ROOT = WORKSPACE_ROOT / 'runtime_b'
SHARED_ROOT = WORKSPACE_ROOT / 'shared'

for path_candidate in (GUI_ROOT, RUNTIME_A_ROOT, RUNTIME_B_ROOT, WORKSPACE_ROOT):
	path_str = str(path_candidate)
	if path_candidate.exists() and path_str not in sys.path:
		sys.path.insert(0, path_str)


PORTABLE_DLL_DIR_HANDLES = []


def preload_portable_dll(*candidates):
	if os.name != 'nt' or not hasattr(ctypes, 'WinDLL'):
		return

	for candidate in candidates:
		if not candidate.exists():
			continue
		try:
			ctypes.WinDLL(str(candidate))
			return
		except OSError:
			continue


def bootstrap_portable_tk_runtime():
	if os.name != 'nt' or not hasattr(os, 'add_dll_directory'):
		return

	python_root = APP_ROOT / 'python'
	if not python_root.exists():
		return

	for candidate in (
		python_root,
		python_root / 'DLLs',
		python_root / 'tcl',
		python_root / 'tcl' / 'tcl8.6',
		python_root / 'tcl' / 'tk8.6',
	):
		if candidate.exists():
			PORTABLE_DLL_DIR_HANDLES.append(os.add_dll_directory(str(candidate)))

	os.environ.setdefault('TCL_LIBRARY', str(python_root / 'tcl' / 'tcl8.6'))
	os.environ.setdefault('TK_LIBRARY', str(python_root / 'tcl' / 'tk8.6'))

	preload_portable_dll(python_root / 'DLLs' / 'zlib1.dll', python_root / 'zlib1.dll')
	preload_portable_dll(python_root / 'DLLs' / 'tcl86t.dll', python_root / 'tcl86t.dll')
	preload_portable_dll(python_root / 'DLLs' / 'tk86t.dll', python_root / 'tk86t.dll')
	preload_portable_dll(python_root / 'DLLs' / '_tkinter.pyd', python_root / '_tkinter.pyd')


bootstrap_portable_tk_runtime()

# GUI stuff
import customtkinter as ctk
import tkinter as tk
from ftfy import fix_text as fxy # unicode text all around fix
from PIL import Image
from CTkToolTip import CTkToolTip
from ezlocalizr import ezlocalizr
import pyglet

# function stuff
import yaml
from shared.nemo_metadata import DEFAULT_NEMO_MODEL, NEMO_HIRAGANA_BACKEND, get_model_choices, supports_language

#
#	default global config stuffs
#

pyglet.options['win32_gdi_font'] = True

DEBUG = True

ASSETS = (SHARED_ROOT / 'assets').resolve()
STRINGS = (GUI_ROOT / 'strings').resolve()
CORPUS = (WORKSPACE_ROOT / 'corpus').resolve()
MODELS = (RUNTIME_B_ROOT / 'models').resolve()

NEMO_GUI_WARNING = 'NeMo is best-effort on Windows and may need a first-run online model download from Hugging Face.'

ctk.set_default_color_theme(str(ASSETS / 'ctk_tgm_theme.json'))
ctk.deactivate_automatic_dpi_awareness()

# logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(format="| %(levelname)s | %(message)s | %(asctime)s |",
					datefmt="%H:%M:%S")
if DEBUG:
	logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

assert ASSETS.exists(), logger.warning('Unable to locate assets folder.')
assert STRINGS.exists(), logger.warning('Unable to locate strings folder.')
CORPUS.mkdir(parents=True, exist_ok=True)
assert MODELS.exists(), logger.warning('No SOFA models installed in \'models\' folder.')


def find_pydomino_model_paths():
	onnx_dir = RUNTIME_B_ROOT / 'onnx_model'
	if not onnx_dir.exists():
		logger.warning(f'pydomino model directory does not exist: {onnx_dir}')
		return {}

	model_paths = {
		candidate.stem: {'onnx_path': candidate}
		for candidate in sorted(onnx_dir.glob('*.onnx'))
	}

	if model_paths:
		logger.info(f'Found {len(model_paths)} pydomino model(s): {", ".join(model_paths.keys())}')
	else:
		logger.warning(f'No pydomino ONNX models found in: {onnx_dir}')

	return model_paths

class LabelMakr(ctk.CTk):
	def __init__(self):
		super().__init__()
		
		# init global config
		self.cfg = {
			'disp_lang': 'en_US',
			'matmul': True,
			'whisper_model': 'medium',
			'transcription_backend': 'whisper',
			'nemo_model': DEFAULT_NEMO_MODEL,
			'dark_mode': True,
			'force_cpu': False
		}

		if P(ASSETS / 'cfg.yaml').exists():
			with open(P(ASSETS / 'cfg.yaml'), 'r', encoding='utf-8') as c:
				try:
					self.cfg.update(yaml.safe_load(c))
					c.close()
				except yaml.YAMLError as exc:
					logger.warning(f'Cannot load config file, using default dictionary.')

		# init variables from config
		self.clang = ctk.StringVar(value=self.cfg['disp_lang'])
		self.inf_wh_model = ctk.StringVar(value=self.cfg['whisper_model'])
		self.transcription_backend_cfg = ctk.StringVar(value=self.cfg.get('transcription_backend', 'whisper'))
		self.nemo_model_cfg = ctk.StringVar(value=self.cfg.get('nemo_model', DEFAULT_NEMO_MODEL))
		self.matmul_var = ctk.BooleanVar(value=self.cfg['matmul'])
		self.dark_mode = ctk.BooleanVar(value=self.cfg['dark_mode'])	
		self.force_cpu = ctk.BooleanVar(value=self.cfg['force_cpu'])

		# init SOFA models
		self.sofa_models = {'models':{}}
		for model_path in MODELS.iterdir():
			model = model_path.name
			# ignore the g2p model file
			if model in ['g2p_model.py', '__pycache__']:
				continue
			g2p_bool = False
			g2p_model = None
			g2p_cfg = None

			if P(MODELS / model / 'g2p').exists():
				g2p_bool = True
				g2p_model = P(MODELS / model / 'g2p/model.ptsd')
				g2p_cfg = P(MODELS / model / 'g2p/cfg.yaml')

			self.sofa_models['models'][model] = {
				'ckpt_path': P(MODELS / model / 'model.ckpt'),
				'dict_path': P(MODELS / model / 'dict.txt'),
				'g2p': g2p_bool,
				'g2p_model': g2p_model,
				'g2p_cfg': g2p_cfg
			}

		self.pydomino_models = {'models':{}}
		self.pydomino_models['models'].update(find_pydomino_model_paths())

		# init languages w/ezlocalizr
		self.L = ezlocalizr(language=self.clang.get(),
							string_path=STRINGS,
							default_lang='en_US')

		self.wh_models = ['tiny', 'base', 'small', 'medium', 'large']
		nemo_backend_value = NEMO_HIRAGANA_BACKEND
		nemo_model_choices = get_model_choices()
		self.transcription_backend_labels = {
			'Whisper': 'whisper',
			'NeMo': nemo_backend_value
		}
		self.transcription_backend_values = {value: key for key, value in self.transcription_backend_labels.items()}
		self.transcription_backend_choice = ctk.StringVar(value=self.transcription_backend_values.get(self.transcription_backend_cfg.get(), 'Whisper'))
		self.nemo_model_labels = nemo_model_choices
		self.nemo_model_values = {value: key for key, value in self.nemo_model_labels.items()}
		default_nemo_label = self.nemo_model_labels.get(DEFAULT_NEMO_MODEL, next(iter(self.nemo_model_labels.values())))
		self.nemo_model_choice = ctk.StringVar(value=self.nemo_model_labels.get(self.nemo_model_cfg.get(), default_nemo_label))
		self.transcribe_lang_op = ['EN', 'JP', 'ZH', 'FR', 'KO']
		self.transcribe_lang_op.sort()
		self.aligner_choices = ['SOFA', 'pydomino']

		# font stuff
		pyglet.font.add_file(str(P(ASSETS / 'PixelOperator.ttf')))
		pyglet.font.add_file(str(P(ASSETS / 'PixelMplus10-Regular.ttf')))
		pyglet.font.add_file(str(P(ASSETS / 'neodgm.ttf')))
		pyglet.font.add_file(str(P(ASSETS / 'WenQuanYi.Bitmap.Song.16px.ttf')))

		self.en_font = 'Pixel Operator'
		self.jp_font = 'PixelMPlus10'
		self.ko_font = 'NeoDunggeunmo'
		self.zh_font = 'WenQuanYi Bitmap Song 16px'

		if self.clang.get() in ['jp_JP']:
			self.font = ctk.CTkFont(family=self.jp_font, size=16)
			self.font_sm = ctk.CTkFont(family=self.jp_font, size=14)
		elif self.clang.get() in ['ko_KO']:
			self.font = ctk.CTkFont(family=self.ko_font, size=16)
			self.font_sm = ctk.CTkFont(family=self.ko_font, size=14)
		elif self.clang.get() in ['zh_ZH']:
			self.font = ctk.CTkFont(family=self.zh_font, size=18)
			self.font_sm = ctk.CTkFont(family=self.zh_font, size=16)
		else:
			self.font = ctk.CTkFont(family=self.en_font, size=16)
			self.font_sm = ctk.CTkFont(family=self.en_font, size=14)

		if self.dark_mode.get():
			ctk.set_appearance_mode("dark")
		else:
			ctk.set_appearance_mode("light")

		logger.info('Successfully initialized LabelMakr.')
		self.main_window()
		
	def main_window(self):

		# window config
		self.title(self.L('app_ttl'))
		self.geometry(f"{420}x{380}")
		self.resizable(height=False, width=True)
		self.minsize(width=405, height=380)
		self.maxsize(width=600, height=380)
		self.tt_delay = 1

		# apparently trying to load an icon in linux breaks shit so. lawl.
		if sys.platform == 'win32':
			if P(ASSETS / 'tgm.ico').exists():
				self.wm_iconbitmap(P(ASSETS / 'tgm.ico'))

		#
		#	GUI Image Initialization
		#

		if P(ASSETS / 'labelmakr.png').exists():
			self.labelmakr_logo = ctk.CTkImage(light_image=Image.open(P(ASSETS / 'labelmakr.png')), size=(300,30))
		if P(ASSETS / 'folder.png').exists():
			self.folder_ico = ctk.CTkImage(light_image=Image.open(P(ASSETS / 'folder.png')))
		if P(ASSETS / 'trns.png').exists():
			self.trns_ico = ctk.CTkImage(light_image=Image.open(P(ASSETS / 'trns.png')))
		if P(ASSETS / 'align.png').exists():
			self.align_ico = ctk.CTkImage(light_image=Image.open(P(ASSETS / 'align.png')))
		if P(ASSETS / 'fix.png').exists():	
			self.fix_ico = ctk.CTkImage(light_image=Image.open(P(ASSETS / 'fix.png')))

		#
		#	TITLE LABEL
		#

		self.grid_columnconfigure(0, weight=1)
		self.grid_columnconfigure(1, weight=1)
		self.grid_rowconfigure(0, weight=1)

		# logo at the top
		self.title_lbl = ctk.CTkLabel(self, image=self.labelmakr_logo, text='')
		self.title_lbl.grid(padx=5, pady=(10, 5), sticky=tk.EW, columnspan=2)

		# whisper variables
		self.trans_lang_choice = ctk.StringVar(value='EN') 

		# what lang are you transcribing?
		self.what_lang = ctk.CTkLabel(self,
									 text=self.L('lang_choice'),
									 font=self.font)
		self.what_lang.grid(row=1, column=0, padx=5, pady=(5, 0), sticky=tk.NE)

		self.lang_cmbo = ctk.CTkComboBox(self,
										 values=self.transcribe_lang_op,
										 variable=self.trans_lang_choice,
										 command=lambda x: self.change_transcription_language(),
										 font=self.font,
										 dropdown_font=self.font,
										 justify='center')
		self.lang_cmbo.set('EN')
		self.lang_cmbo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.NW)
		self.lang_cmbo_tt = CTkToolTip(self.what_lang, delay=self.tt_delay, message=self.L('lang_choice_tt'), font=self.font)

		#
		#	Unnecessarily long tab configuration
		#

		# commented lines are for future features

		self.tabs = ctk.CTkTabview(self)
		self.tabs.grid(padx=5, pady=(0, 5), sticky=tk.EW, columnspan=2)

		self.tab_ttl_1 = self.L('tab_ttl_1')
		self.tab_ttl_2 = self.L('tab_ttl_2')
		self.tab_ttl_3 = self.L('tab_ttl_3')
		self.tab_ttl_4 = self.L('tab_ttl_4')

		self.tabs.add(self.tab_ttl_1)
		self.tabs.add(self.tab_ttl_2)
		self.tabs.add(self.tab_ttl_3)
		self.tabs.add(self.tab_ttl_4)
		self.tabs.set(self.tab_ttl_1)

		self.tabs._segmented_button.configure(font=self.font)

		# copyright label at the bottom of the screen
		self.credits = ctk.CTkLabel(self, 
									text=fxy('© tigermeat 2023-2024 | rokujyushi 2026 | v051'), 
									text_color="gray50",
									font=self.font)
		self.credits.grid(padx=5, pady=(0, 5), sticky=tk.EW, columnspan=2)

		#
		#	Transcription Tab
		#

		self.tabs.tab(self.tab_ttl_1).grid_columnconfigure(0, weight=1)
		self.tabs.tab(self.tab_ttl_1).grid_columnconfigure(1, weight=1)
		self.tabs.tab(self.tab_ttl_1).grid_rowconfigure((0, 1), weight=1)
		self.tabs.tab(self.tab_ttl_1).grid_rowconfigure((2, 3), weight=3)

		self.asr_backend_lbl = ctk.CTkLabel(self.tabs.tab(self.tab_ttl_1),
									   text=self.L('asr_backend'),
									   font=self.font)
		self.asr_backend_lbl.grid(row=0, column=0, padx=5, pady=(10, 5), sticky=tk.N)
		self.asr_backend_lbl_tt = CTkToolTip(self.asr_backend_lbl, delay=self.tt_delay, message=self.L('asr_backend_tt'), font=self.font)

		self.asr_backend_cmbo = ctk.CTkComboBox(self.tabs.tab(self.tab_ttl_1),
									values=list(self.transcription_backend_labels.keys()),
									command=lambda x: self.update_transcription_backend(),
									variable=self.transcription_backend_choice,
									font=self.font,
									dropdown_font=self.font,
									justify='center')
		self.asr_backend_cmbo.grid(row=1, column=0, padx=5, pady=5, sticky=tk.EW)

		self.asr_model_lbl = ctk.CTkLabel(self.tabs.tab(self.tab_ttl_1),
								   text=self.L('wh_model'),
								   font=self.font)
		self.asr_model_lbl.grid(row=0, column=1, padx=5, pady=(10, 5), sticky=tk.N)
		self.asr_model_lbl_tt = CTkToolTip(self.asr_model_lbl, delay=self.tt_delay, message=self.L('wh_model_tt'), font=self.font)

		self.asr_whisper_cmbo = ctk.CTkComboBox(self.tabs.tab(self.tab_ttl_1),
									 values=self.wh_models,
									 command=lambda x: self.update_wh_model(),
									 variable=self.inf_wh_model,
									 font=self.font,
									 dropdown_font=self.font,
									 justify='center')
		self.asr_whisper_cmbo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

		self.asr_nemo_cmbo = ctk.CTkComboBox(self.tabs.tab(self.tab_ttl_1),
								  values=list(self.nemo_model_labels.values()),
								  command=lambda x: self.update_nemo_model(),
								  variable=self.nemo_model_choice,
								  font=self.font,
								  dropdown_font=self.font,
								  justify='center')
		self.asr_nemo_cmbo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

		# open corpus button
		self.corp_btn = ctk.CTkButton(self.tabs.tab(self.tab_ttl_1),
									  text=self.L('corpus_folder'),
									  command=lambda: self.startfolder(CORPUS),
									  image=self.folder_ico,
									  compound=tk.LEFT,
									  font=self.font)
		self.corp_btn.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.NSEW)
		self.corp_btn_tt = CTkToolTip(self.corp_btn, delay=self.tt_delay, message=self.L('corpus_folder_tt'), font=self.font)

		# transcribe button
		self.trns_btn = ctk.CTkButton(self.tabs.tab(self.tab_ttl_1),
									  text=self.L('run_trns'),
									  command=lambda: self.run_transcriber(),
									  image=self.trns_ico,
									  compound=tk.LEFT,
									  font=self.font)
		self.trns_btn.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=tk.NSEW)
		self.trns_btn_tt = CTkToolTip(self.trns_btn, delay=self.tt_delay, message=self.L('run_trns_tt'), font=self.font)

		#
		#	Alignment Tab GUI Codes
		#

		# grid configs
		self.tabs.tab(self.tab_ttl_2).grid_columnconfigure((0, 1), weight=1)
		self.tabs.tab(self.tab_ttl_2).grid_rowconfigure((0, 1), weight=1)
		self.tabs.tab(self.tab_ttl_2).grid_rowconfigure(2, weight=3)

		self.model_choice = ctk.StringVar(value='tgm_sofa_en')
		self.aligner_choice = ctk.StringVar(value='SOFA')
		self.op_mode = ctk.StringVar(value='htk')
		self.op_choices = ['htk', 'TextGrid']
		self.update_model_choices('SOFA')

		self.aligner_lbl = ctk.CTkLabel(self.tabs.tab(self.tab_ttl_2),
								  text=self.L('aligner_lbl'),
								  font=self.font)
		self.aligner_lbl.grid(row=0, column=0, padx=5, pady=(10, 5), sticky=tk.N)
		self.aligner_lbl_tt = CTkToolTip(self.aligner_lbl, delay=self.tt_delay, message=self.L('aligner_lbl_tt'), font=self.font)

		self.aligner_cmbo = ctk.CTkComboBox(self.tabs.tab(self.tab_ttl_2),
									 values=self.aligner_choices,
									 variable=self.aligner_choice,
									 command=lambda x: self.change_aligner_mode(),
									 font=self.font,
									 dropdown_font=self.font,
									 justify='center')
		self.aligner_cmbo.set(self.aligner_choice.get())
		self.aligner_cmbo.grid(row=1, column=0, padx=5, pady=5, sticky=tk.N)

		# choose sofa model
		self.model_lbl = ctk.CTkLabel(self.tabs.tab(self.tab_ttl_2),
									  text=self.L('model_lbl'),
									  font=self.font)
		self.model_lbl.grid(row=0, column=1, padx=5, pady=(10, 5), sticky=tk.N)
		self.model_lbl_tt = CTkToolTip(self.model_lbl, delay=self.tt_delay, message=self.L('model_lbl_tt'), font=self.font)

		# model choice combobox
		self.model_cmbo = ctk.CTkComboBox(self.tabs.tab(self.tab_ttl_2),
								  values=self.get_model_names(),
										  variable=self.model_choice,
										  font=self.font,
										  dropdown_font=self.font,
										  justify='center')
		if self.get_model_names():
			self.model_cmbo.set(self.model_choice.get())
		self.model_cmbo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.N)

		# choose format
		self.op_lbl = ctk.CTkLabel(self.tabs.tab(self.tab_ttl_2),
								   text=self.L('op_lbl'),
								   font=self.font)
		self.op_lbl.grid(row=2, column=0, padx=5, pady=(10, 5), sticky=tk.N)
		self.op_lbl_tt = CTkToolTip(self.op_lbl, delay=self.tt_delay, message=self.L('op_lbl_tt'), font=self.font)

		# model choice combobox
		self.op_cmbo = ctk.CTkComboBox(self.tabs.tab(self.tab_ttl_2),
									   values=self.op_choices,
									   variable=self.op_mode,
									   font=self.font,
									   dropdown_font=self.font,
									   justify='center')
		self.op_cmbo.set(self.op_mode.get())
		self.op_cmbo.grid(row=2, column=1, padx=5, pady=(10, 5), sticky=tk.N)

		# align button
		self.align_btn = ctk.CTkButton(self.tabs.tab(self.tab_ttl_2),
									   text=self.L('run_align'),
							   command=lambda: self.run_alignment(),
									   image=self.align_ico,
									   compound=tk.LEFT,
									   font=self.font)
		self.align_btn.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=tk.NSEW)
		self.align_btn_tt = CTkToolTip(self.align_btn, delay=self.tt_delay, message=self.L('run_align_tt'), font=self.font)
		self.change_aligner_mode()

		#
		#	Fix Label Tab GUI Code
		#

		self.tabs.tab(self.tab_ttl_3).grid_columnconfigure((0, 1), weight=1)
		self.tabs.tab(self.tab_ttl_3).grid_rowconfigure(0, weight=0)
		self.tabs.tab(self.tab_ttl_3).grid_rowconfigure((1, 2), weight=1)
		self.tabs.tab(self.tab_ttl_3).grid_rowconfigure(3, weight=3)

		# help label
		self.labbu_help = ctk.CTkLabel(self.tabs.tab(self.tab_ttl_3),
									   text=self.L('labbu_help'),
									   font=self.font_sm,
									   text_color='lightgray',)
		self.labbu_help.grid(row=0, column=0, columnspan=2, padx=5, pady=2.5, sticky=tk.NSEW)

		# dxer box
		self.dxer = ctk.BooleanVar(value=True)
		self.dxer_cb = ctk.CTkCheckBox(self.tabs.tab(self.tab_ttl_3),
									   variable=self.dxer,
									   onvalue=True,
									   offvalue=False,
									   text=self.L('dxer'),
									   font=self.font)
		self.dxer_cb.grid(row=1, column=0, padx=5, pady=5, sticky=tk.N)
		self.dxer_cb_tt = CTkToolTip(self.dxer_cb, delay=self.tt_delay, message=self.L('dxer_tt'), font=self.font)

		# uhr merge
		self.uhr_merge = ctk.BooleanVar(value=True)
		self.uhr_merge_cb = ctk.CTkCheckBox(self.tabs.tab(self.tab_ttl_3),
											variable=self.uhr_merge,
											onvalue=True,
											offvalue=False,
											text=self.L('uhr_merge'),
											font=self.font)
		self.uhr_merge_cb.grid(row=1, column=1, padx=5, pady=5, sticky=tk.N)
		self.uhr_merge_tt = CTkToolTip(self.uhr_merge_cb, delay=self.tt_delay, message=self.L('uhr_merge_tt'), font=self.font)

		# merge duplicates
		self.merge_dupes = ctk.BooleanVar(value=True)
		self.merge_dupes_cb = ctk.CTkCheckBox(self.tabs.tab(self.tab_ttl_3),
											  variable=self.merge_dupes,
											  onvalue=True,
											  offvalue=False,
											  text=self.L('merge_dupes'),
											  font=self.font)
		self.merge_dupes_cb.grid(row=2, column=0, padx=5, pady=5, sticky=tk.N)
		self.merge_dupes_tt = CTkToolTip(self.merge_dupes_cb, delay=self.tt_delay, message=self.L('merge_dupes_tt'), font=self.font)

		# merge h
		self.merge_h = ctk.BooleanVar(value=True)
		self.merge_h_cb = ctk.CTkCheckBox(self.tabs.tab(self.tab_ttl_3),
										  variable=self.merge_h,
										  onvalue=True,
										  offvalue=False,
										  text=self.L('short_h'),
										  font=self.font)
		self.merge_h_cb.grid(row=2, column=1, padx=5, pady=5, sticky=tk.N)
		self.merge_h_tt = CTkToolTip(self.merge_h_cb, delay=self.tt_delay, message=self.L('short_h_tt'), font=self.font)

		# run fix button
		self.run_fix_btn = ctk.CTkButton(self.tabs.tab(self.tab_ttl_3),
									     text=self.L('run_fix'),
									     command=lambda: self.run_label_fix(),
									     image=self.fix_ico,
									     compound=tk.LEFT,
									     font=self.font)
		self.run_fix_btn.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=tk.NSEW)
		self.run_fix_tt = CTkToolTip(self.run_fix_btn, delay=self.tt_delay, message=self.L('run_fix_tt'), font=self.font)

		#
		#	Settings Tab GUI Code
		#

		# grid configure
		self.tabs.tab(self.tab_ttl_4).grid_columnconfigure((0, 1), weight=1)
		self.tabs.tab(self.tab_ttl_4).grid_rowconfigure((0, 1, 2, 3), weight=1)

		# choose display language
		self.set_lang_lbl = ctk.CTkLabel(self.tabs.tab(self.tab_ttl_4),
									     text=self.L('disp_lang'),
									     font=self.font)
		self.set_lang_lbl.grid(row=0, column=0, padx=5, pady=5, sticky=tk.N)
		self.set_lang_lbl_tt = CTkToolTip(self.set_lang_lbl, delay=self.tt_delay, message=self.L('disp_lang_tt'), font=self.font)

		# model choice combobox
		self.set_lang_cmbo = ctk.CTkComboBox(self.tabs.tab(self.tab_ttl_4),
										  	 values=self.L.lang_list,
										  	 command=lambda x: self.refresh(self.clang.get()),
										  	 variable=self.clang,
										  	 font=self.font,
										  	 dropdown_font=self.font,
										  	 justify='center')
		self.set_lang_cmbo.set(self.clang.get())
		self.set_lang_cmbo.grid(row=1, column=0, padx=5, pady=5, sticky=tk.EW)

		# tensorcore checkbox
		self.matmul_ckbx = ctk.CTkCheckBox(self.tabs.tab(self.tab_ttl_4),
										   variable=self.matmul_var,
										   onvalue=True,
										   offvalue=False,
										   text=self.L('use_tensorcore'),
										   command=lambda: self.update_matmul(),
										   font=self.font)
		if self.cfg['matmul']:
			self.matmul_ckbx.select()
		elif not self.cfg['matmul']:
			self.matmul_ckbx.deselect()

		self.matmul_ckbx.grid(row=2, column=0, padx=5, pady=5, sticky=tk.NW)
		self.matmul_ckbx_tt = CTkToolTip(self.matmul_ckbx, delay=self.tt_delay, message=self.L('use_tensorcore_tt'), font=self.font)

		# force CPU rendering checkbox
		self.force_cpu_ckbx = ctk.CTkCheckBox(self.tabs.tab(self.tab_ttl_4),
											  variable=self.force_cpu,
											  onvalue=True,
											  offvalue=False,
											  text=self.L('force_cpu'),
											  command=lambda: self.update_cpu_render(),
											  font=self.font)
		self.force_cpu_ckbx.grid(row=2, column=1, padx=5, pady=5, sticky=tk.NW)
		self.force_cpu_ckbx_tt = CTkToolTip(self.force_cpu_ckbx, delay=self.tt_delay, message=self.L('force_cpu_tt'), font=self.font)

		# appearance checkbox
		self.appearance_rbtn = ctk.CTkCheckBox(self.tabs.tab(self.tab_ttl_4),
											   variable=self.dark_mode,
											   onvalue=True,
											   offvalue=False,
											   text=self.L('dark_mode'),
											   command=lambda: self.change_appearance(),
											   font=self.font)
		self.appearance_rbtn.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=tk.NW)
		self.appearance_rbtn_tt = CTkToolTip(self.appearance_rbtn, delay=self.tt_delay, message=self.L('dark_mode_tt'), font=self.font)

		self.sync_transcription_backend_ui()

	def refresh(self, choice):
		# Better option for updating the display language tbh.
		self.cfg['disp_lang'] = choice
		with open(P(ASSETS / 'cfg.yaml'), 'w', encoding='utf-8') as f:
			yaml.dump(self.cfg, f, default_flow_style=False)
			f.close()
		self.L.load_lang(choice)

		logger.info(f'Set display language to {choice}')

		self.destroy()
		app = LabelMakr()
		app.mainloop()

	def update_matmul(self):
		self.cfg['matmul'] = self.matmul_var.get()

		with open(P(ASSETS / 'cfg.yaml'), 'w', encoding='utf-8') as f:
			yaml.dump(self.cfg, f, default_flow_style=False)
			f.close()

		logger.info(f'Updated matmul setting')

	def run_transcriber(self):
		backend = self.selected_transcription_backend()

		logger.info(f'Initializing transcription backend: {backend}')

		# forces small Whisper model upon using "CPU" only mode so your PC don't explode
		inference_model = self.inf_wh_model.get()
		if self.force_cpu.get() and backend == 'whisper':
			inference_model = 'small'

		worker_args = [
			'--working-dir',
			str(WORKSPACE_ROOT),
			'--lang',
			self.trans_lang_choice.get(),
		]

		if backend == self.transcription_backend_labels['NeMo']:
			model_id = self.selected_nemo_model()
			if not supports_language(model_id, self.trans_lang_choice.get()):
				logger.warning(f'Selected NeMo model does not support language {self.trans_lang_choice.get()}.')
				return
			logger.warning(NEMO_GUI_WARNING)

			worker_args = ['--backend', 'nemo', *worker_args, '--nemo-model', model_id]
			if self.force_cpu.get():
				worker_args.append('--force-cpu')
		else:
			worker_args = ['--backend', 'whisper', *worker_args, '--whisper-model', inference_model]

		self.launch_runtime_worker('runtime_a', 'worker.py', worker_args)

	def run_sofa(self,
				 ckpt: str,
				 dictionary: str,
				 g2p_bool: bool,
				 g2p_model: str,
				 g2p_cfg: str
		):
		worker_args = [
			'sofa',
			'--working-dir',
			str(WORKSPACE_ROOT),
			'--ckpt',
			str(ckpt),
			'--dictionary',
			str(dictionary),
			'--op-format',
			self.op_cmbo.get(),
			'--lang',
			self.lang_cmbo.get(),
		]

		if self.matmul_var.get():
			worker_args.append('--matmul')
		if g2p_bool:
			worker_args.append('--g2p')
		if g2p_model is not None:
			worker_args.extend(['--g2p-model', str(g2p_model)])
		if g2p_cfg is not None:
			worker_args.extend(['--g2p-cfg', str(g2p_cfg)])

		self.launch_runtime_worker('runtime_b', 'worker.py', worker_args)

	def run_pydomino(self, onnx_path: str):
		worker_args = [
			'pydomino',
			'--working-dir',
			str(WORKSPACE_ROOT),
			'--onnx-path',
			str(onnx_path),
			'--output-format',
			self.op_cmbo.get(),
			'--min-aligned-timeframe',
			'3',
			'--corpus-dir',
			str(CORPUS),
		]

		self.launch_runtime_worker('runtime_b', 'worker.py', worker_args)

	def resolve_runtime_python(self, runtime_name):
		runtime_root = WORKSPACE_ROOT / runtime_name
		candidates = [
			runtime_root / 'python' / 'python.exe',
			runtime_root / '.venv' / 'Scripts' / 'python.exe',
			P(sys.executable),
		]

		for candidate in candidates:
			if candidate.exists():
				return candidate

		return P(sys.executable)

	def launch_runtime_worker(self, runtime_name, worker_name, worker_args):
		worker_path = WORKSPACE_ROOT / runtime_name / worker_name
		if not worker_path.exists():
			logger.warning(f'Runtime worker could not be found: {worker_path}')
			return

		python_exe = self.resolve_runtime_python(runtime_name)
		command = [str(python_exe), str(worker_path), *worker_args]
		subprocess.Popen(command, cwd=WORKSPACE_ROOT)
		logger.info(f'Launched {runtime_name} worker: {worker_name}')

	def run_alignment(self):
		model_name = self.model_cmbo.get()
		if not model_name:
			logger.warning('No alignment model selected.')
			return

		if self.aligner_choice.get() == 'pydomino':
			if model_name not in self.pydomino_models['models']:
				logger.warning('Selected pydomino model could not be found.')
				return
			self.run_pydomino(self.pydomino_models['models'][model_name]['onnx_path'])
			return

		if model_name not in self.sofa_models['models']:
			logger.warning('Selected SOFA model could not be found.')
			return

		model = self.sofa_models['models'][model_name]
		self.run_sofa(model['ckpt_path'], model['dict_path'], model['g2p'], model['g2p_model'], model['g2p_cfg'])

	def get_model_names(self):
		if self.aligner_choice.get() == 'pydomino':
			return sorted(self.pydomino_models['models'].keys())
		return sorted(self.sofa_models['models'].keys())

	def update_model_choices(self, mode=None):
		if mode is not None:
			self.aligner_choice.set(mode)

		model_names = self.get_model_names()
		if model_names:
			self.model_choice.set(model_names[0])
		else:
			self.model_choice.set('')

	def change_aligner_mode(self):
		self.update_model_choices(self.aligner_choice.get())
		model_names = self.get_model_names()
		self.model_cmbo.configure(values=model_names if model_names else [''])
		if model_names:
			self.model_cmbo.set(self.model_choice.get())
			self.align_btn.configure(state='normal')
		else:
			self.model_cmbo.set('')
			self.align_btn.configure(state='disabled')

		if self.aligner_choice.get() == 'pydomino':
			self.model_lbl.configure(text=self.L('pydomino_model_lbl'))
			self.model_lbl_tt.configure(message=self.L('pydomino_model_lbl_tt'))
			self.align_btn.configure(text=self.L('run_align_pydomino'))
			self.align_btn_tt.configure(message=self.L('run_align_pydomino_tt'))
		else:
			self.model_lbl.configure(text=self.L('model_lbl'))
			self.model_lbl_tt.configure(message=self.L('model_lbl_tt'))
			self.align_btn.configure(text=self.L('run_align'))
			self.align_btn_tt.configure(message=self.L('run_align_tt'))

	def startfile(self, filename):
		try:
			os.startfile(filename)
		except:
			subprocess.Popen(['xdg-open', filename])

	def startfolder(self, foldername):
			"""
			Open a folder in file explorer
			If the folder doesn't exist, create it.
			"""
			folder = P(foldername).resolve()
			#create the folder if it doesn't exist
			if(not folder.is_dir()):
				folder.mkdir(parents=True, exist_ok=True)
			self.startfile(folder)

	def update_wh_model(self):
		self.inf_wh_model.set(self.asr_whisper_cmbo.get())
		self.cfg['whisper_model'] = self.asr_whisper_cmbo.get()
		with open(P(ASSETS / 'cfg.yaml'), 'w', encoding='utf-8') as f:
			yaml.dump(self.cfg, f, default_flow_style=False)
			f.close()
		logger.info(f"Set Whisper Model to {self.asr_whisper_cmbo.get()}")

	def selected_transcription_backend(self):
		return self.transcription_backend_labels.get(self.transcription_backend_choice.get(), 'whisper')

	def selected_nemo_model(self):
		return self.nemo_model_values.get(self.nemo_model_choice.get(), DEFAULT_NEMO_MODEL)

	def sync_transcription_backend_ui(self):
		whisper_enabled = self.selected_transcription_backend() == 'whisper'
		nemo_enabled = self.selected_transcription_backend() != 'whisper'
		whisper_state = 'normal' if whisper_enabled else 'disabled'
		nemo_state = 'normal' if nemo_enabled else 'disabled'
		self.asr_whisper_cmbo.configure(state=whisper_state)
		self.asr_nemo_cmbo.configure(state=nemo_state)

		if whisper_enabled:
			self.asr_model_lbl.configure(text=self.L('wh_model'), text_color=('gray10', 'gray90'))
			self.asr_model_lbl_tt.configure(message=self.L('wh_model_tt'))
			self.asr_whisper_cmbo.grid()
			self.asr_nemo_cmbo.grid_remove()
			self.matmul_ckbx.grid(row=2, column=0, padx=5, pady=5, sticky=tk.NW)
			self.force_cpu_ckbx.grid(row=2, column=1, padx=5, pady=5, sticky=tk.NW)
		else:
			self.asr_model_lbl.configure(text=self.L('nemo_model'), text_color=('gray10', 'gray90'))
			self.asr_model_lbl_tt.configure(message=self.L('nemo_model_tt'))
			self.asr_whisper_cmbo.grid_remove()
			self.asr_nemo_cmbo.grid()
			self.matmul_ckbx.grid_remove()
			self.force_cpu_ckbx.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.NW)

		self.appearance_rbtn.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=tk.NW)

	def update_transcription_backend(self):
		backend = self.selected_transcription_backend()
		self.transcription_backend_cfg.set(backend)
		self.cfg['transcription_backend'] = backend
		with open(P(ASSETS / 'cfg.yaml'), 'w', encoding='utf-8') as f:
			yaml.dump(self.cfg, f, default_flow_style=False)
			f.close()
		self.sync_transcription_backend_ui()
		logger.info(f'Set transcription backend to {backend}')

	def update_nemo_model(self):
		model_id = self.selected_nemo_model()
		self.nemo_model_cfg.set(model_id)
		self.cfg['nemo_model'] = model_id
		with open(P(ASSETS / 'cfg.yaml'), 'w', encoding='utf-8') as f:
			yaml.dump(self.cfg, f, default_flow_style=False)
			f.close()
		logger.info(f'Set NeMo model to {model_id}')

	def update_cpu_render(self):
		'''
		wip
		'''
		return None

	def run_label_fix(self):
		worker_args = [
			'labbu',
			'--working-dir',
			str(WORKSPACE_ROOT),
			'--lang',
			self.lang_cmbo.get(),
			'--corpus-dir',
			str(CORPUS),
		]

		if self.dxer_cb.get():
			worker_args.append('--dxer')
		if self.uhr_merge_cb.get():
			worker_args.append('--uhr-merge')
		if self.merge_h_cb.get():
			worker_args.append('--merge-h')
		if self.merge_dupes_cb.get():
			worker_args.append('--merge-dupes')

		self.launch_runtime_worker('runtime_b', 'worker.py', worker_args)
		logger.info('Launched label fix worker.')

	def change_transcription_language(self):
		if self.lang_cmbo.get() == 'EN':
			self.dxer_cb.select()
			self.dxer_cb.configure(state="normal")
			self.uhr_merge_cb.select()
			self.uhr_merge_cb.configure(state="normal")
		else:
			self.dxer_cb.deselect()
			self.dxer_cb.configure(state="disabled")
			self.uhr_merge_cb.deselect()
			self.uhr_merge_cb.configure(state="disabled")

	def change_appearance(self):
		dark_mode = self.appearance_rbtn.get()

		self.cfg['dark_mode'] = dark_mode

		with open(P(ASSETS / 'cfg.yaml'), 'w', encoding='utf-8') as f:
			yaml.dump(self.cfg, f, default_flow_style=False)
			f.close()

		if dark_mode:
			ctk.set_appearance_mode('dark')
			logger.info('Toggled dark mode.')
		else:
			ctk.set_appearance_mode('light')
			logger.info('Toggled light mode.')

def main():
	app = LabelMakr()
	app.mainloop()

if __name__ == "__main__":
	main()