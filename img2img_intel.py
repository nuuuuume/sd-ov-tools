import sys
import os
import datetime
import time
import argparse
import numpy as np
from optimum.intel import OVStableDiffusionImg2ImgPipeline
from lauda import stopwatch
from diffusers import (
    DDIMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler
)
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
    
def no_safety_checker(images, **kwargs):
    return images, None

def printEalased(watch, function):
    m, s = divmod(watch.elapsed_time, 60)
    h, m = divmod(m, 60)
    
    print(f"<<< {h}:{m}:{s} spent.")


@stopwatch(callback=printEalased)
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

    print(f"--- model from pretranined ... ")
    
    compile = False if args.useGpu else True
    seed = args.seed if args.seed >= 0 else int(time.time())
    print(f"--- seed: {seed}")

    generator = np.random.RandomState(seed)
    
    print(f"--- model from pretranined ... ")
    
    compile = False if args.useGpu else True
    dtype = np.float16 if args.useGpu else np.float32

    # 画像を開いて、Bicubicでリサイズ（ここでは劣化）、その後そのリサイズ後の画像にImg2Imgするらしい。
    srcImage = Image.open(args.srcImagePath).convert("RGB")
    sw, sh = srcImage.size
    print(f"--- source image size: {sw}x{sh}")
    (srcImage, w, h) = resizeImage(srcImage, args.scale)
    print(f"--- resized image size: {w}x{h}")
        
    modelName = os.path.splitext(os.path.basename(args.model))[0]
    print(f"--- model: {modelName}")
    # openvinoでIntelGPUを使ってモデルの読み込み
    pipe = OVStableDiffusionImg2ImgPipeline.from_pretrained(
        args.model,
        dtype=dtype,
        compile=False)
    pipe.scheduler = eval(args.scheduler).from_config(pipe.scheduler.config)
    pipe.reshape(batch_size=1,
        height=h,
        width=w,
        num_images_per_prompt=args.numImages)
        
    if args.useGpu:
        pipe.to('GPU')
    
    pipe.compile()    
    
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

        fileName = f"img2img_{modelName}_{now.strftime('%Y%m%d%H%M%S')}_{suffix:03}.png"
        outputFilePath = os.path.join(outputDir, fileName)
        print(f"--- output file path: {outputFilePath}")
        image.save(outputFilePath)
        print(f"--- saved. {outputFilePath}")

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
                        required=True,
                        help='input image path')
    parser.add_argument('--prompt', 
                        dest='prompt',
                        action='store', 
                        required=True,
                        help='prompt')
    parser.add_argument('--negative_prompt', 
                        dest='negativePrompt',
                        action='store', 
                        default="",
                        help='negative prompt')
    parser.add_argument('--output_dir', 
                        dest='outputDir',
                        action='store', 
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
    parser.add_argument('--gpu', 
                        dest='useGpu', 
                        action='store_true', 
                        default=False)

    args = parser.parse_args()    

    main(args)
