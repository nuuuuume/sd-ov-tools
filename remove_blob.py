import os
import glob
import argparse

def remove_model_blobs(model_root_dir: str):

    if not os.path.isdir(model_root_dir):
        return

    for modeldir_name in os.listdir(model_root_dir):

        modeldir = os.path.join(model_root_dir, modeldir_name) 
        subdirs = ['feature_extractor', 'scheduler', 'text_encoder', 'tokenizer', 'unet', 'vae_decoder', 'vae_encoder']

        for subfolder in subdirs:
            dir = os.path.join(modeldir, subfolder)
            pattern = os.path.join(dir, '*.blob')
            for blob_file in glob.glob(pattern):
                print(f"--- {blob_file}")
                os.remove(blob_file)

if __name__ == '__main__':

    p = argparse.ArgumentParser('remove openvino inter blob files when text2img killed in processing.')

    p.add_argument('--model_root_dir',
                   dest='model_root_dir',
                   action='store',
                   type=str,
                   default='models',
                   help='model root directory')

    args = p.parse_args() 
    remove_model_blobs(args.model_root_dir)