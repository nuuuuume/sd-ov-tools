import sys
import os
import datetime
import subprocess
from pathlib import Path
from convert_safetensors_to_ov import safetensors_to_ov
from batch_t2i_config import make_configs

def main(args):

    """ StableDiffusionを使って画像を生成します。
    """

    # venv上のpythonパスとt2iのスクリプトへのパス
    venv_python = r".\venv-3_10_6-sd\Scripts\python.exe"
    t2i_py = r".\text2img_ov.py"
    
    # 設定。一応複数持てるように。
    # スケジューラーやシード、画像サイズなんかをプロンプトごとに変えられるようにした。
    # プロンプトごとモデルごとに回る。
    # 1つのプロンプトを色々なモデルで試したいので。。 
    configs = make_configs()

    for config in configs:
        if 'enable' in config and not config['enable']:
            continue
        for prompt in config['prompt']:
            for m in config['models']:
                model = m
                modelName = os.path.splitext(os.path.basename(model))[0]
                if config['use_openvino'] and model.endswith('.safetensors'):
                    print("safetensors file detected. try convert safetensors to openvino ir format ...")
                    model = f"models/{modelName}"
                    if not safetensors_to_ov(m, model):
                        print("failed to convert openvino ir format. skip.")
                        continue

                now = datetime.datetime.now()

                output_dir = config['output_dir']
                output_dir = os.path.join(output_dir, f"{now:%Y%m%d}")
                os.makedirs(output_dir, exist_ok=True)

                # パラメータくっつけて自作のopenvino版t2iを起動。
                args = [
                    venv_python,
                    t2i_py,
                    '--model',
                    model,
                    '--prompt',
                    prompt,
                    '--negative_prompt',
                    config['negative_prompt'],
                    '--width',
                    str(config['width']),
                    '--height',
                    str(config['height']),
                    '--batch_count',
                    str(config['batch_count']),
                    '--batch_size',
                    str(config['batch_size']),
                    '--seed',
                    str(config['seed']),
                    '--guidance_scale',
                    str(config['guidance_scale']),
                    '--steps',
                    str(config['steps']),
                    '--scheduler',
                    config['scheduler'],
                    '--output_dir',
                    output_dir,
                ]
                if config['use_gpu']:
                    args.append('--gpu')

                if config['dump_setting']:
                    args.append('--dump_setting')

                if config['use_openvino']:
                    args.append('--openvino')

                if 'vae_path' in config:
                    args.append('--vae_path')
                    args.append(config['vae_path'])

                if 'lora' in config:
                    args.append('--lora')
                    for lora in config['lora']:
                        args.append(f"{lora['path']}:{lora['scale']}")

                subprocess.run(args)

    print("done")

main(sys.argv)
