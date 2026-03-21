 _       _          _                 _         
| |     | |        | |               | |        
| | __ _| |__   ___| |_ __ ___   __ _| | ___ __ 
| |/ _` | '_ \ / _ \ | '_ ` _ \ / _` | |/ / '__|
| | (_| | |_) |  __/ | | | | | | (_| |   <| |   
|_|\__,_|_.__/ \___|_|_| |_| |_|\__,_|_|\_\_|   
                                                
##############
#	CREDIT	 #
##############

SOFA developed by suco/qiuqiao
Whisper created by OpenAI

tgm_sofa English SOFA Model created by tigermeat
colstone_jpn Japanese SOFA Model created by colstone
mandarin_suco Mandarin Singing SOFA Model created by SOFA Developer suco/qiuqiao

GUI Translations:
Japanese: tigermeat
Chinese: ArchiVoice
Indonesian: Koji

##########################
#	 CURRENT LAYOUT	 #
##########################

This distribution uses a 3-runtime layout:

- gui/       : GUI only
- runtime_a/ : Whisper + NeMo transcription
- runtime_b/ : SOFA + pydomino alignment
- shared/    : common assets, FFmpeg, corpus, setup scripts

Common assets live in shared/.
Runtime-specific assets live in their own runtime folders.

#############
#	USAGE	#
#############

REQUIREMENTS:
- Windows is the primary supported portable environment.
- GPU is recommended, but CPU setup is also supported.

HOW TO INSTALL:
1: Run one of these scripts:
- shared\setup_CPU.bat
- shared\setup_GPU.bat

This will:
- install dependencies for gui/, runtime_a/, and runtime_b/
- install shared assets
- download runtime-specific assets like g2p-jp, SOFA, and pydomino ONNX

2: Start the app with:
- gui\run.bat

3: Place all files in shared\corpus\, for example:

shared\corpus
|	{name_of_speaker_1}
|	|	wav1.wav
|	|	wav2.wav
|	|	...
|	{name_of_speaker_2}
|	|	wav1.wav
|	|	wav2.wav
|	|	...

4: Use the transcription tab for Whisper / NeMo transcription.

5: Then use the alignment tab for SOFA or pydomino.

##############
#	EXTRAS	 #
##############

If you get this warning on the command line:

"You are using a CUDA device ('{your_gpu}') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision"

Then please enable "Use Tensorcores" in the settings menu! This will fully utilize your GPU, speeding up forced-alignments!