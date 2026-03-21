# LabelMakr+ README

LabelMakr+ は、SVS 用の音素ラベル生成を補助する GUI ツールです。DiffSinger 向けを主目的としていますが、他のワークフローにも流用できます。現在は英語、日本語、中国語、フランス語、韓国語の歌唱を対象にしています。

## 3 ランタイム構成

- `gui/`: 画面表示、設定保存、再生、ジョブ起動
- `runtime_a/`: `Whisper + NeMo` による転写
- `runtime_b/`: `SOFA + pydomino` によるアライメント
- `shared/`: 共通アセット、`corpus/`、FFmpeg、セットアップスクリプト

## セットアップ

Windows ではまず [shared/setup_CPU.bat](shared/setup_CPU.bat) または [shared/setup_GPU.bat](shared/setup_GPU.bat) を実行してください。

- `setup_CPU.bat`: CPU 向け依存を `gui` / `runtime_a` / `runtime_b` に導入
- `setup_GPU.bat`: GPU 向け Torch 系依存を導入
- どちらも最後に [shared/install_assets.py](shared/install_assets.py) を実行して共通アセットを配置

セットアップ後は [gui/run.bat](gui/run.bat) で GUI を起動します。

詳しい導入メモは [setup_guide.txt](setup_guide.txt) を参照してください。

## フォルダ構成

- `gui/`
  - `labelmakr.py`
  - `labbu.py`
  - `labbu_func.py`
  - `requirements.txt`
  - `run.bat`
- `runtime_a/`
  - `worker.py`
  - `whisper_func.py`
  - `nemo_func.py`
  - `requirements.txt`
  - `g2p-jp/` はアセット導入時に配置
- `runtime_b/`
  - `worker.py`
  - `sofa_func.py`
  - `pydomino_func.py`
  - `requirements.txt`
  - `SOFA/` と `onnx_model/` はアセット導入時に配置
- `shared/`
  - `assets/`
  - `models/`
  - `corpus/`
  - `ffmpeg.exe` / `ffprobe.exe`
  - `setup_CPU.bat` / `setup_GPU.bat` / `set_env.bat`

## NeMo ASR の利用

NeMo は `runtime_a` 側の依存に含めています。通常は [shared/setup_CPU.bat](shared/setup_CPU.bat) または [shared/setup_GPU.bat](shared/setup_GPU.bat) を実行すれば GUI から利用できます。

設定タブで `ASR Backend` を NeMo 側にすると、`NeMo Model` で以下を選べます。

- `ReazonSpeech NeMo V2 (JP)`: 日本語長時間音声向け、`reazon-research/reazonspeech-nemo-v2`
- `Hiragana Parakeet 0.6B (JP)`: 日本語ひらがな出力、`kizuna-intelligence/hiragana-parakeet-tdt-ctc-0.6b-ja-beta`
- `Parakeet TDT 0.6B V2 (EN)`: 英語向け、`nvidia/parakeet-tdt-0.6b-v2`

制限事項:

- NeMo は公式のサポートマトリクス上で Windows が未サポートです。このプロジェクトではベストエフォート動作です。
- 初回実行時は Hugging Face から重みをダウンロードするため、オンライン接続が必要です。
- 日本語の `.nemo` モデルは約 2.5 GB あるため、GPU またはメモリに余裕のある環境を推奨します。
- 英語 Parakeet は NVIDIA GPU と Linux が推奨環境で、CPU でも動く可能性はありますが速度は保証できません。

## 将来の予定とか

- アライメントツール、[GAME](https://github.com/openvpi/GAME) の対応
- 音声認識ツール、[Fun-ASR](https://github.com/openvpi/Fun-ASR) の対応
- 日本語 g2p の改善、[pyopenjtalk-plus](https://github.com/tsukumijima/pyopenjtalk-plus) の対応

## Community Contributions 🧑‍🤝‍🧑

- Le guide d'utilisation en Français [peut-être trouvé ici](https://utaufrance.com/comment-utiliser-labelmakr/)! (Written by [Mim](https://twitter.com/mimsynth))
- 한국어 사용 가이드는 [여기](https://docs.google.com/document/d/1-EcFrkt4VDjRlFQ8Sytvov4_3GjDt4-xHYNjQDuDScU/edit)서 찾을 수 있습니다! (Written by [군곰 KUNGOM](https://twitter.com/utaukg))

## Custom SOFA Model Implementation

Please check out the guide on how to implement custom SOFA models [here!](https://github.com/rokujyushi/LabelMakr/blob/add-pydomino/DOCS/implement_custom_sofa_model.md)

## Credits

Please check out credits [here!](https://github.com/rokujyushi/LabelMakr/blob/add-pydomino/DOCS/credits.md)

## Manual Installation 🧰

Read the guide on how to manually install LabelMakr [here!](https://github.com/rokujyushi/LabelMakr/blob/add-pydomino/DOCS/manual_install_guide.md)
