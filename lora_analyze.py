import argparse
from safetensors.torch import load_file

def main(args):

    state_dict = load_file(args.lora_path)

    with open(args.dump_path, 'wt') as f:
        for key in state_dict.keys():
            f.write(f"{key} {state_dict[key].shape}\n") 

if __name__ == '__main__':

    p = argparse.ArgumentParser()

    p.add_argument('--lora_path',
                   dest='lora_path',
                   type=str,
                   default=r'C:\Users\webnu\source\repos\StableDiffusion\stable-diffusion-webui\models\Lora\brighter-eye2.safetensors')
    p.add_argument('--dump_path',
                   dest='dump_path',
                   type=str,
                   default='lora.txt')
    args = p.parse_args()
    main(args)