import sys
import os
import datetime
import time
import argparse
import json
import numpy as np
import torch
from optimum.intel import OVStableDiffusionImg2ImgPipeline
from lauda import stopwatch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.schedulers import *
from PIL import Image

def resizeImage(image, scale):
    """
    PILのimageをscaleだけリサイズします。アスペクト比は維持されます。
    image: PILの画像
    scale: double 1.0で等倍。2.0で2倍。
    
    return リサイズ後の画像（ndarray）, リサイズ後の横幅、リサイズ後の縦幅
    """
    sw, sh = image.size
    
    # resize
    rw = int(sw * scale)
    rh = int(sh * scale)

    return (image.resize((rw, rh), Image.Resampling.BICUBIC), rw, rh)    
    
def printElapsed(watch, function):
    m, s = divmod(watch.elapsed_time, 60)
    h, m = divmod(m, 60)
    
    print(f"<<< {int(h)}:{int(m)}:{int(s)} spent.")

def isSafetenorFile(model):
    """ input.jsonのmodelがsafetensorsファイルの場合Trueを返します。
    単にendswithしてるだけ。
    """
    return model.endswith("safetensors")

def no_safety_checker(images, **kwargs):
    return images, [False] * len(images)

def makePipeline(args):

    """
    コマンドライン引数の情報からStableDiffusionPipelineとgeneratorを作成して返します
    """

    deviceType = "cuda" if args.useGpu else "cpu"
    torchdtype = torch.float16 if args.useGpu else torch.float32

    generator = torch.Generator(device=deviceType).manual_seed(args.seed)

    if isSafetenorFile(args.model):
        print(f"--- model from file ... ")
        pipe = StableDiffusionImg2ImgPipeline.from_single_file(
            args.model,
            torch_dtype=torchdtype,
            load_safety_checker=False)
    else:
        print(f"--- model from pretranined ... ")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.model,
            torch_dtype=torchdtype)

        if pipe.safety_checker != None:
            pipe.safety_checker = no_safety_checker

    pipe = pipe.to(deviceType)

    return (pipe, generator)

def makeOpenvinoPipeline(args, w, h):

    """
    コマンドライン引数からOVStableDiffusionPipelineとgeneratorを作成して返します
    """
    generator = np.random.RandomState(args.seed)

    # openvinoで読み込み
    pipe = OVStableDiffusionImg2ImgPipeline.from_pretrained(
        args.model,
        compile=False)

    # モデルのサイズを限定するとメモリが浮く 
    pipe.reshape(batch_size=1,
        height=h,
        width=w,
        num_images_per_prompt=args.numImages)

    if args.useGpu:
        # GPUを使う場合は以下を実行。t2iは動くけどi2iは画像が真っ黒になった（CPUだとi2iもちゃんとできる）
        # 11世代CoreのIrisXEだとCPUで回したほうが速度は出るという、、。
        pipe.to('GPU')

    pipe.compile()    

    return (pipe, generator)

@stopwatch(callback=printElapsed)
def main(args):

    """ StableDiffusionを使って画像を生成します。
    """
    print(f"--- useGpu: {args.useGpu}")
    print(f"--- model: {args.model}")
    print(f"--- srcImagePath: {args.srcImagePath}")
    print(f"--- output dir: {args.outputDir}")
    print(f"--- seed {args.seed}")
    print(f"--- scale: {args.scale}")
    print(f"--- prompt: {args.prompt}")
    print(f"--- negativePrompt: {args.negativePrompt}")
    print(f"--- inference steps: {args.steps}")
    print(f"--- guidance scale: {args.guidanceScale}")
    print(f"--- num images per prompt: {args.numImages}")
    print(f"--- scheduler: {args.scheduler}")
    print(f"--- use openvino?: {args.useOpenvino}")
    
    print(f"--- model from pretranined ... ")
    
    compile = False if args.useGpu else True
    seed = args.seed if args.seed >= 0 else int(time.time())
    print(f"--- seed: {seed}")
    args.seed = seed

    generator = np.random.RandomState(seed)
    
    # 画像を開いて、Bicubicでリサイズ（ここでは劣化）、その後そのリサイズ後の画像にImg2Imgするらしい。
    srcImage = Image.open(args.srcImagePath).convert("RGB")
    sw, sh = srcImage.size
    print(f"--- source image size: {sw}x{sh}")
    (srcImage, w, h) = resizeImage(srcImage, args.scale)
    print(f"--- resized image size: {w}x{h}")
        
    modelName = os.path.splitext(os.path.basename(args.model))[0]
    print(f"--- model: {modelName}")

    pipe, genereator = makeOpenvinoPipeline(args, w, h) if args.useOpenvino else makePipeline(args)

    result = pipe(prompt=args.prompt,
        negative_prompt=args.negativePrompt,
        image=srcImage,
        guidance_scale=args.guidanceScale,
        num_images_per_prompt=args.numImages,
        generator=generator,
        num_inference_steps=args.steps)

    suffix = 1
    while len(result.images) > 0:
        print(f"--- nsfw detected? : {result.nsfw_content_detected}")
        image = result.images[0]

        now = datetime.datetime.now()

        outputDir = args.outputDir
        os.makedirs(outputDir, exist_ok=True)

        fileNameBase = f"img2img_{modelName}_{now.strftime('%Y%m%d%H%M%S')}_{suffix:03}"
        fileName = f"{fileNameBase}.png"
        outputFilePath = os.path.join(outputDir, fileName)
        print(f"--- output file path: {outputFilePath}")
        image.save(outputFilePath)
        print(f"--- saved. {outputFilePath}")
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

    parser = argparse.ArgumentParser(description="StableDiffusion image-to-image with intel gpu")

    parser.add_argument('--model', 
                        dest='model', 
                        type=str, 
                        action='store', 
                        required=True,
                        help='Openvino format model')
    parser.add_argument('--src_image_path', 
                        dest='srcImagePath',
                        action='store', 
                        type=str,
                        required=True,
                        help='input image path')
    parser.add_argument('--prompt', 
                        dest='prompt',
                        action='store', 
                        type=str,
                        required=True,
                        help='prompt')
    parser.add_argument('--negative_prompt', 
                        dest='negativePrompt',
                        action='store', 
                        type=str,
                        default="",
                        help='negative prompt')
    parser.add_argument('--output_dir', 
                        dest='outputDir',
                        action='store', 
                        type=str,
                        required=True,
                        help='generated image saved here')
    parser.add_argument('--scale', 
                        dest='scale',
                        action='store', 
                        type=float,
                        default=1.0,
                        help='scale image.')
    parser.add_argument('--num_images_per_prompt', 
                        dest='numImages',
                        type=int,
                        action='store', 
                        default=1,
                        help='generate image num per prompt')
    parser.add_argument('--seed', 
                        dest='seed',
                        action='store', 
                        type=int,
                        default=-1,
                        help='random seed')
    parser.add_argument('--guidance_scale', 
                        dest='guidanceScale',
                        type=int,
                        action='store', 
                        default=7,
                        help='guidance scale')
    parser.add_argument('--steps', 
                        dest='steps',
                        action='store', 
                        type=int,
                        default=20,
                        help='inference steps')
    parser.add_argument('--scheduler', 
                        dest='scheduler',
                        action='store',
                        type=str,
                        default="EulerAncestralDiscreteScheduler",
                        help='inference steps')
    parser.add_argument('--gpu', 
                        dest='useGpu', 
                        action='store_true', 
                        default=False)
    parser.add_argument('--openvino',
                        dest='useOpenvino',
                        action='store_true',
                        default=False,
                        help="if True use openvino")
    parser.add_argument('--dump_setting', 
                        dest='dumpSetting', 
                        action='store_true', 
                        default=False,
                        help="dump i2i setting to file if True. default is False.")


    args = parser.parse_args()    

    main(args)
