import argparse
import os
import glob
import json
import subprocess

def readJson(filePath):
    """ JSONファイルを読み込んで返します
    filePath: jsonファイルパス 
    """
    with open(filePath, 'r') as f:
        ret = json.load(f)

    return ret

def main(args):
    """ text2img_intel.pyによって生成された画像と設定ダンプファイルをもとにimg2imgを実行します。
    変換対象の画像と設定ダンプファイルを所定のフォルダに入れておくとそれら全てに対してimg2imgを実行します。
    """    

    venvPython = os.path.join("venv", "Scripts", "python.exe")
    i2iPath = "img2img_intel.py"

    # 処理の流れ
    # args.outputDirがなければ作成
    # args.inputDirが存在するか検査
    # args.inputDirの中の pngファイルをすべて取得
    # pngファイルを回す
    # pngファイルと同じ名前のtxtファイルを探す
    # なければその画像はスキップ
    # txtファイルをJSONとして開く
    # prompt, negativePrompt, を取得する
    # argsの値と合体させてimg2img_intel.pyを実行する

    if not os.path.isdir(args.inputDir):
        print(f"{args.inputDir} was not found...")
        return

    inputPattern = os.path.join(args.inputDir, "*.png")
    inputPngFiles = glob.glob(inputPattern)
    for i, inputFilePath in inputPngFiles:  
        print(f"--- {i + 1}/{len(inputPngFiles)} : {inputFilePath} ...")
        # 同名のtxtファイルが必要（t2i時の設定ファイル）
        baseFilePath, ext = os.path.splitext(inputFilePath)
        settingFilePath = f"{baseFilePath}.txt"
        if not os.path.isfile(settingFilePath):
            print(f"{settingFilePath} was not found, skip this image.")
            continue

        # jsonとして開く
        setting = readJson(settingFilePath)

        # ほしいのはmodelとpromptとnegativePromptとseed
        # あとはargsと組み合わせてi2i用のパラメータを作って実行する。

        # seedについてはこのバッチにて指定されていれば元の設定を上書きする
        seed = args.seed if args.seed != None else setting['seed'] 

        subArgs = [
            venvPython,
            i2iPath,
            '--model',
            setting['model'],
            '--src_image_path',
            inputFilePath,
            '--prompt',
            setting['prompt'],
            '--negative_prompt',
            setting['negativePrompt'],
            '--scale',
            str(args.scale),
            '--num_images_per_prompt',
            str(args.numImages),
            '--seed',
            str(seed),
            '--guidance_scale',
            str(args.guidanceScale),
            '--steps',
            str(args.steps),
            '--scheduler',
            args.scheduler,
            '--output_dir',
            args.outputDir,
        ]
        if(setting['dumpSetting']):
            subArgs.append['--dump_setting']

        print(f"{' '.join(subArgs)}")
        subprocess.run(subArgs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('img2img diffusion batch process.')

    parser.add_argument('--input_dir',
                        dest='inputDir',
                        type=str,
                        required=True,
                        help='input directory')
    parser.add_argument('--output_dir',
                        dest='outputDir',
                        type=str,
                        required=True,
                        help='output directory')
    parser.add_argument('--scale', 
                        dest='scale',
                        action='store', 
                        type=float,
                        default=1.0,
                        help='scale image.')
    parser.add_argument('--num_images_per_prompt', 
                        dest='numImages',
                        action='store', 
                        default=1,
                        help='generate image num per prompt')
    parser.add_argument('--guidance_scale', 
                        dest='guidanceScale',
                        action='store', 
                        default=7,
                        help='guidance scale')
    parser.add_argument('--steps', 
                        dest='steps',
                        action='store', 
                        default=50,
                        help='inference steps')
    parser.add_argument('--scheduler', 
                        dest='scheduler',
                        action='store',
                        choices=["DDIMScheduler", "PNDMScheduler", "LMSDiscreteScheduler"],
                        default="LMSDiscreteScheduler",
                        help='inference steps')
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        dest='seed',
                        help='seed')
    parser.add_argument('--dump_setting',
                        dest='dumpSetting',
                        default=False,
                        action='store_true',
                        help='dump input setting')

    args = parser.parse_args()

    main(args)