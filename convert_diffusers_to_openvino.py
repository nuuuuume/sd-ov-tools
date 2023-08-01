# Windowsだと動かないがLinuxだと動く。。
import sys
import os
import datetime
import json
import argparse
from optimum.intel import OVStableDiffusionPipeline

def isSafetenorFile(model):
    """ input.jsonのmodelがsafetensorsファイルの場合Trueを返します。
    単にendswithしてるだけ。
    """
    return model.endswith("safetensors")

def no_safety_checker(images, **kwargs):
    return images, None

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
