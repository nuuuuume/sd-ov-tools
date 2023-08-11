import sys
import os
import datetime
import time
import argparse
import json
import numpy as np
import cv2 as cv
from optimum.intel import OVStableDiffusionPipeline
from lauda import stopwatch
from diffusers.schedulers import *

def printEalased(watch, function):
    m, s = divmod(watch.elapsed_time, 60)
    h, m = divmod(m, 60)
    
    print(f"<<< {int(h)}:{int(m)}:{int(s)} spent.")

@stopwatch(callback=printEalased)
def main(args):

    print(f"--- useGpu: {args.useGpu}")
    print(f"--- model: {args.model}")
    print(f"--- output dir: {args.outputDir}")
    print(f"--- seed {args.seed}")
    print(f"--- widthxheight: {args.width}x{args.height}")
    print(f"--- prompt: {args.prompt}")
    print(f"--- negativePrompt: {args.negativePrompt}")
    print(f"--- inference steps: {args.steps}")
    print(f"--- guidance scale: {args.guidanceScale}")
    print(f"--- num images per prompt: {args.numImages}")
    print(f"--- scheduler: {args.scheduler}")

    print(f"--- model from pretranined ... ")
    
    seed = args.seed if args.seed >= 0 else int(time.time())
    print(f"--- seed: {seed}")
    args.seed = seed

    # メモ
    # ・VAEを外から当てる事はできるか
    # OpenvinoではOVModelVaeEncoder, OVModelVaeDecoderが公開されていないため、
    # 外からVAEを指定することができないっぽい。
    # モデルのサブフォルダに vae_encoder, vae_decoderがあるとそれをみにいくようなので
    # VAEを変更したければモデルのフォルダをイジる必要がありそう。 
    # ・Loraについて
    # Loraはunetに対して、 pipe.unet.load_attn_procs(lora_checkpoint_path) こんなようなコードを呼ぶとロードしてくれるらしい。
    # あとはトリガーワード（Loraをよく効かせるためのワード）があればpromptに指定するだけっぽい。
    # openvinoの場合、Unetに相当するOVUnetModelにload_attn_procsがイないので今のところはムリポなのかな。
    # Loraは学習済みモデルに上乗せするものっぽいので、Loraを上乗せした状態のモデルを保存することができればあるいは。。
    # なんかopenvinoでLoRAの利用を実装しているコードがあった。あの変ヒントにすればなにかできるかも？
    # ・Scheduler
    # ソースを見ると DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, の3つしかサポートしてないような感じだけど、
    # from_pretarainedしてから任意のスケジューラーをぶっこめば動くっぽい。diffusersのSchedulerが使えそう。本家のStableDiffusionPipelineもこれら3つ能から一つと
    # 書いてある割に別のデモいけるから、あくまで初期値としては、ってことなのかもしれないね。SchedulerMixinを実装してるやつ？Mixinといいながらベースクラスっぽいね？
    # class DDIMScheduler(SchedulerMixin, ConfigMixin): ←こんな感じの定義だし。
 
    generator = np.random.RandomState(seed)
    
    modelName = os.path.splitext(os.path.basename(args.model))[0]
    print(f"--- model: {modelName}")
    # openvinoで読み込み
    pipe = OVStableDiffusionPipeline.from_pretrained(
        args.model,
        compile=False)

    # モデルのサイズを限定するとメモリが浮く 
    pipe.reshape(batch_size=1,
        height=args.height,
        width=args.width,
        num_images_per_prompt=args.numImages)

    pipe.scheduler = eval(args.scheduler).from_config(pipe.scheduler.config)

    if args.useGpu:
        # GPUを使う場合は以下を実行。t2iは動くけどi2iは画像が真っ黒になった（CPUだとi2iもちゃんとできる）
        # 11世代CoreのIrisXEだとCPUで回したほうが速度は出るという、、。
        pipe.to('GPU')

    pipe.compile()    

    # 出力ファイルの末尾につける連番。 
    suffix = 1

    # 推論実行。絵ができるよ。
    result = pipe(prompt=args.prompt,
        negative_prompt=args.negativePrompt,
        width=args.width,
        height=args.height,
        guidance_scale=args.guidanceScale,
        num_images_per_prompt=args.numImages,
        generator=generator,
        num_inference_steps=args.steps)

    # num_images_per_promptで指定した枚数の画像ができるのでそれぞれ保存
    while len(result.images) > 0:

        image = result.images[0]
        
        now = datetime.datetime.now()

        outputDir = args.outputDir
        os.makedirs(outputDir, exist_ok=True)

        fileNameBase = f"{now.strftime('%Y%m%d%H%M%S')}_{modelName}_{suffix:03}"
        fileName = f"{fileNameBase}.png"
        outputFilePath = os.path.join(outputDir, fileName)
        print(f"--- output file path: {outputFilePath}")
        image.save(outputFilePath)
        print(f"--- saved. {outputFilePath}")
        # pngにexifとして設定を埋め込むのがWebUIとかはやってるみたいだけど、
        # テキストファイルに落としておけばビューワもいらないからとりあえずテキストにダンプ。
        # 必要が出たらpngに埋めればいいんでなかな。
        if args.dumpSetting:
            dumpFileName = f"{fileNameBase}.txt"
            dumpFilePath = os.path.join(outputDir, dumpFileName)
            with open(dumpFilePath, "w") as f:
                json.dump(args.__dict__, f, ensure_ascii=False, indent=4)
            print(f"--- setting dumped: {dumpFilePath}")

        suffix += 1
        del result.images[0]

    # release memory
    del pipe

    print("done")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="StableDiffusion text-to-image with intel gpu")

    parser.add_argument('--model', 
                        dest='model', 
                        type=str, 
                        action='store', 
                        required=True,
                        help='Openvino format model')
    parser.add_argument('--prompt', 
                        dest='prompt',
                        type=str,
                        action='store', 
                        required=True,
                        help='prompt')
    parser.add_argument('--output_dir', 
                        dest='outputDir',
                        type=str,
                        action='store', 
                        required=True,
                        help='generated image saved here')
    parser.add_argument('--negative_prompt', 
                        dest='negativePrompt',
                        type=str,
                        action='store', 
                        default="",
                        help='negative prompt')
    parser.add_argument('--width', 
                        dest='width',
                        type=int,
                        action='store', 
                        default=512,
                        help='generated image width')
    parser.add_argument('--height', 
                        dest='height',
                        type=int,
                        action='store', 
                        default=512,
                        help='generated image height')
    parser.add_argument('--num_images_per_prompt', 
                        dest='numImages',
                        type=int,
                        action='store', 
                        default=1,
                        help='generate image num per prompt')
    parser.add_argument('--seed', 
                        dest='seed',
                        type=int,
                        action='store', 
                        default=-1,
                        help='random seed')
    parser.add_argument('--guidance_scale', 
                        dest='guidanceScale',
                        type=float,
                        action='store', 
                        default=7,
                        help='guidance scale')
    parser.add_argument('--steps', 
                        dest='steps',
                        type=int,
                        action='store', 
                        default=50,
                        help='inference steps')
    parser.add_argument('--scheduler', 
                        dest='scheduler',
                        type=str,
                        action='store',
                        default="DDIMScheduler",
                        help='scheduler which one of diffusers.schedulers')
    parser.add_argument('--gpu', 
                        dest='useGpu', 
                        action='store_true', 
                        default=False,
                        help="true if use gpu. default False.")
    parser.add_argument('--dump_setting', 
                        dest='dumpSetting', 
                        action='store_true', 
                        default=False,
                        help="dump t2i setting to file if True. default is False.")


    args = parser.parse_args()    

    main(args)
