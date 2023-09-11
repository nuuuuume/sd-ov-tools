# stable diffusionのモデルをopenvino用のモデルに変換するツール。
# 一回diffusersの形式を経由する。
# SD-safetensors -> diffusers -> openvino ir という流れで変換する。

import os
import argparse
import pathlib
import shutil
from optimum.intel import OVStableDiffusionPipeline
from diffusers import StableDiffusionPipeline

def main(args):

    safetensors_to_ov(args.safetensors_file,
                      args.dump_path,
                      args.force_overwrite)

def safetensors_to_ov(safetensors_path,
                      dump_path,
                      force_overwrite=False) -> bool:

    """
    Convert original stable diffusion safetensors model to openvino ir format.

    Parameters:
        safetensors_path: str
            original stable diffusion safetensors file path.
        dump_path: str
            save directory converted openvino ir model.
        force_overwrite: bool
            if True overwrite existsing openvino ir model
    """

    if not os.path.isfile(safetensors_path):
        print(f"safetensors_path is not found. {safetensors_path}")
        return False

    if os.path.exists(dump_path) and not force_overwrite:
        print(f"this model already converted. skip.")
        return True

    print("exporting safetensors -> diffusers model ...")
    sdpipe = StableDiffusionPipeline.from_single_file(safetensors_path)

    # diffusers_model folder is temporary folder.
    temp_dir = pathlib.Path('./diffusers_model/')
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)

    temp_model_dir = temp_dir / os.path.basename(safetensors_path)
    if os.path.isdir(temp_model_dir):
        # remove old dir
        shutil.rmtree(temp_model_dir)

    sdpipe.save_pretrained(temp_model_dir)

    print("exporting diffusers model -> openvino ir")
    if os.path.isdir(dump_path):
        shutil.rmtree(dump_path)

    pipe = OVStableDiffusionPipeline.from_pretrained(
        temp_model_dir,
        export=True)

    pipe.save_pretrained(dump_path)

    print("clean up...") 
    # release memory
    print("pipelines ...") 
    del pipe
    del sdpipe

    # remove diffusers model dir
    print("temp directory ...")
    shutil.rmtree(temp_model_dir)

    print("done")

    return True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='convert diffusers model to openvino format')

    parser.add_argument('--safetensors_file',
                        dest='safetensors_file',
                        type=str,
                        action='store',
                        required=True,
                        help='safetensors file path.')
    parser.add_argument('--dump_path',
                        dest='dump_path',
                        type=str,
                        action='store',
                        required=True,
                        help='openvino format model output directory')

    args = parser.parse_args()

    main(args)
