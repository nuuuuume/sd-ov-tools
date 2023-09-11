# sd-ov-tools
StableDiffusion using OpenVino tools
StableDiffusionPipelineとOVStableDiffusionPipeline、StableDiffusionImg2ImgPipelineとOVStableDiffusionImg2ImgPipelineを使ってt2iやi2iを行うツール郡。
StableDiffusionのモデル（safetensors形式）を使ってopenvino ir 形式に変換しながら生成する。
一応、openvinoでないdiffusersのStableDiffusionPipelineも対応したけど、それならwebuiのほうが良い。
openvino版は、vae,lora,controlnetなどの主要な機能が使えない。。。textual inversionだけは組み込んだ。

python -m venv venv-3_10_6-sd
venv-3_10_6-sd/Scripts/activate
pip install -r requirements.txt