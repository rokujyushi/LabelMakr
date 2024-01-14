import sys
import os
sys.path.append('.')
from ftfy import fix_text as fxy
import subprocess
import re
import glob
from pypinyin import lazy_pinyin

class Transcriber(object):
	def __init__(self, lang, wh_model):
		super().__init__()
		import whisper
		from whisper.tokenizer import get_tokenizer

		# referenced code from MLo7's MFA Notebook :)
		self.model = whisper.load_model(wh_model)
		whisper.DecodingOptions(language=lang.lower())
		self.tokenizer = get_tokenizer(multilingual=False)
		self.number_tokens = [i for i in range(self.tokenizer.eot) if all(c in "0123456789" for c in self.tokenizer.decode([i]))]

	def jpn_g2p(self, jpn):

		# uses .exe version of openjtalk G2p
		phonemes = subprocess.check_output(f"g2p-jp/japanese_g2p.exe -rs {jpn.replace(' ', '')}", shell=False)
		g2p_op = str(phonemes)
		fixed = re.sub(r"([aeiouAIEOUN])", r" \1 ", g2p_op[2:-5])
		# fix cl
		fixed = re.sub("cl", "cl ", fixed)
		# remove punctuation
		fixed = re.sub(r"[.!?,]", "", fixed)
		# remove extra spaces
		fixed = re.sub(" {2,}", " ", fixed)
		# lowercase any uppercase vowels but _NOT_ [N]
		fixed = re.sub("A", "a", fixed)
		fixed = re.sub("I", "i", fixed)
		fixed = re.sub("U", "u", fixed)
		fixed = re.sub("E", "e", fixed)
		fixed = re.sub("O", "o", fixed)
		return fixed

	def run_transcription(self, lang):

		for file in glob.glob('corpus/**/*.wav', recursive=True):
			try:
				out_name = file[:-4] + '.txt'
				out2_name = file[:-4] +"_Fixed"+ '.txt'
				out3_name = file[:-4] +"_JP"+ '.txt'
				file_flag = os.path.exists(out2_name)
				file_flag2 = os.path.exists(out3_name)
				if file_flag2:
					print("skip：",out3_name)
					continue
				
				if not file_flag:
					# get transcription from Whisper
					answer = self.model.transcribe(file, suppress_tokens=[-1] + self.number_tokens)
				else:
					with open(out2_name, 'r+', encoding='utf-8') as out:
						answer['text'] = out.read()
						out.close()
				
				# language specifics here
				if lang.upper() == "JP":
					# turn the kanji into G2p output
					trns_str_kanji = fxy(answer['text'])
					print("function：1")
					if not file_flag2:
						with open(out3_name, 'w+', encoding='utf-8') as out:
							out.write(trns_str_kanji)
							out.close()
					print("function：2")
					trns_str = self.jpn_g2p(trns_str_kanji)
				elif lang.upper() == "ZH":
					# remove any spaces just in case ig
					hanzi_list = lazy_pinyin(re.sub(' ', '', fxy(answer['text'])))	
					trns_str = ""
					for word in hanzi_list:
						trns_str += f"{word} "
				else:
					# the default, currently just being used by English.
					trns_str = fxy(answer['text']).lower()
				
				# remove any punctuation
				trns_str = re.sub(r"[.,!?]", "", trns_str)

				# write file out
				with open(out_name, 'w+', encoding='utf-8') as out:
					out.write(trns_str)
					out.close()

				print(f"Wrote transcription for {file} in corpus.")

			except RuntimeError as e:
				print(f"Error Transcribing: {e}")

		print('Completed all transcriptions!')

if __name__ == "__main__":
	#Transcriber.eng_g2p(Transcriber, 'test')
	print('What do u think ur doing silly billy!')