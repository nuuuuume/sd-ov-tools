# diffusersのmodelをopenvino用のモデルに変換するツール。
# 一発でやる手段もありそうな気もするけど、
# ・safetensors を diffusers/scripts/convert_original_stable_diffusion_to_diffusers.pyで --fron_safetensorsを付けて変換
#    python diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --from_safetensors --checkpoint_path <path/to/safetensors> --dump_path <path/to/diffusers_model>
# ・出来上がったモデルフォルダをこのスクリプトでopenvino形式に変換
#    python convert_diffusers_to_openvino.py --diffusers_model <path/to/diffusers_model> --output_dir <path/to/openvino_model>
# とやる。
# で、これをtext2img_ov.pyの --model に指定する --model <path/to/openvino_model> みたいに。
import sys
import os
import datetime
import json
import argparse
from optimum.intel import OVStableDiffusionPipeline

def main(args):

    print(f"--- selected model: {args.srcModel}")
    print(f"--- output name: {args.outputDir}")

    pipe = OVStableDiffusionPipeline.from_pretrained(
        args.srcModel,
        export=True)
        
    pipe.save_pretrained(args.outputDir)
    
    # release memory
    del pipe

    print("done")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='convert diffusers model to openvino format')

    parser.add_argument('--diffusers_model',
                        dest='srcModel',
                        type=str,
                        action='store',
                        required=True,
                        help='diffusers model id or diffusers model dir')
    parser.add_argument('--output_dir',
                        dest='outputDir',
                        type=str,
                        action='store',
                        required=True,
                        help='openvino format model output directory')
    args = parser.parse_args()

    main(args)
