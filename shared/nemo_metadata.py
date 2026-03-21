NEMO_HIRAGANA_BACKEND = 'nemo_hiragana_parakeet'
NEMO_HIRAGANA_REPO = 'kizuna-intelligence/hiragana-parakeet-tdt-ctc-0.6b-ja-beta'
NEMO_HIRAGANA_FILENAME = 'hiragana-parakeet-tdt-ctc-0.6b-ja.nemo'
DEFAULT_NEMO_MODEL = 'reazonspeech_nemo_v2'

NEMO_MODELS = {
	'reazonspeech_nemo_v2': {
		'label': 'ReazonSpeech NeMo V2 (JP)',
		'languages': ['JP'],
		'loader': 'restore_from',
		'repo_id': 'reazon-research/reazonspeech-nemo-v2',
		'filename': 'reazonspeech-nemo-v2.nemo',
		'description': 'Japanese long-form ASR model from ReazonSpeech.',
	},
	'hiragana_parakeet_ja_beta': {
		'label': 'Hiragana Parakeet 0.6B (JP)',
		'languages': ['JP'],
		'loader': 'restore_from',
		'repo_id': NEMO_HIRAGANA_REPO,
		'filename': NEMO_HIRAGANA_FILENAME,
		'description': 'Japanese hiragana-only ASR model.',
	},
	'parakeet_tdt_0_6b_v2_en': {
		'label': 'Parakeet TDT 0.6B V2 (EN)',
		'languages': ['EN'],
		'loader': 'from_pretrained',
		'model_name': 'nvidia/parakeet-tdt-0.6b-v2',
		'description': 'English FastConformer-TDT model with punctuation and capitalization.',
	},
}


def get_model_choices():
	return {model_id: config['label'] for model_id, config in NEMO_MODELS.items()}


def supports_language(model_id, lang):
	config = NEMO_MODELS.get(model_id)
	if config is None:
		return False
	return lang.upper() in config['languages']
