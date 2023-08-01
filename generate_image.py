import sys
import os
import datetime
import json
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from compel import Compel
from lauda import stopwatch


def isSafetenorFile(model):
    """ input.jsonのmodelがsafetensorsファイルの場合Trueを返します。
    単にendswithしてるだけ。
    """
    return model.endswith("safetensors")

def isUseGpu(inputData):
    """ inputJsonのuseGpuが設定され、かつ"true"の場合にTrueを返します。
    """
    return inputData['useGpu'] if 'useGpu' in inputData else False

def no_safety_checker(images, **kwargs):
    return images, False

def printEalased(watch, function):
    m, s = divmod(watch.elapsed_time, 60)
    h, m = divmod(m, 60)
    
    print(f"<<< {h}:{m}:{s} spent.")


@stopwatch(callback=printEalased)
def main(args):

    """ StableDiffusionを使って画像を生成します。
    """
    inputData = {
      "useGpu": False,
      "model": "C:\\Users\\webnu\\source\\repos\\StableDiffusion\\stable-diffusion-webui\\models\\Stable-diffusion\\SakuraMix-v2.1.safetensors",
      "prompt": "masterpiece, best quality, absurdres, ultra detailed beautiful face and large red eyes, 1girl, (flat chest)++, black frilled lolita dress, (short skirt with white inner color)++, (open mouth, happy).blend(), black hair, twintails with large white ribbons, (full body)++, (black over-the-knee-socks)++, no shoes",
      "negativePrompt": "nsfw+++, (worst quality, low quality:1.4), (lip, nose, tooth, rouge, lipstick, eyeshadow:1.4), bad hands, mutated hands and fingers, bad feet, bad legs, flat color, flat shading, (jpeg artifacts:1.4), (depth of field, bokeh, blurry, film grain, chromatic aberration, lens flare:1.0), (1boy, abs, muscular, rib:1.0), greyscale, monochrome, dusty sunbeams, trembling, motion lines, motion blur, emphasis lines, text, title, logo, signature,",
      "outputDir": "G:\\マイドライブ\\StableDiffusion\\output",
      "width": 768,
      "height": 768,
      "numOfGenerates": 1,
      "seed": -1,
      "guidanceScale": 7,
    }

    now = datetime.datetime.now()

    outputDir = inputData['outputDir']
    useGpu = isUseGpu(inputData)
    seed = inputData['seed'] if 'seed' in inputData else 33
    width = inputData['width'] if 'width' in inputData else 512
    height = inputData['height'] if 'height' in inputData else 512
    deviceType = "cuda" if useGpu else "cpu"
    torchdtype = torch.float16 if useGpu else torch.float32

    print(f"--- selected model: {inputData['model']}")
    print(f"--- output dir: {inputData['outputDir']}")
    print(f"--- use gpu? {useGpu}")
    print(f"--- seed {seed}")
    print(f"--- widthxheight: {width}x{height}")
    print(f"--- prompt: {inputData['prompt']}")
    print(f"--- negativePrompt: {inputData['negativePrompt']}")
    print(f"--- guidance scale: {inputData['guidanceScale']}")

    if isSafetenorFile(inputData['model']):
        print(f"--- model from file ... ")
        pipe = StableDiffusionPipeline.from_single_file(
            inputData['model'],
            torch_dtype=torchdtype,
            load_safety_checker=False)
    else:
        print(f"--- model from pretranined ... ")
        pipe = StableDiffusionPipeline.from_pretrained(
            inputData['model'],
            torch_dtype=torchdtype)

        #pipe.safety_checker = no_safety_checker
    
    pipe = pipe.to(deviceType)
    # 
    compelProc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
    prompt = compelProc(inputData['prompt'])
    negativePrompt = compelProc(inputData['negativePrompt'])

    generator = torch.Generator(device=deviceType).manual_seed(seed)
    print(f"--- generated seed: {generator.seed()}")

    images = pipe(prompt_embeds=prompt,
        width=width,
        height=height,
        generator=generator,
        guidance_scale=inputData['guidanceScale'],
        negative_prompt_embeds=negativePrompt,
        num_images_per_prompt=inputData['numOfGenerates']).images

    i = 1
    while len(images) > 0:

        fileName = f"output_{now.strftime('%Y%m%d%H%M%S')}_{i:03}.png"
        outputFilePath = os.path.join(outputDir, fileName)
        print(f"--- output file path: {outputFilePath}")
        images[0].save(outputFilePath)
        print(f"--- saved. {outputFilePath}")

        i += 1
        del images[0]

    # release memory
    del pipe
    del generator
    del compelProc

    # release gpu cache
    if useGpu:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("done")

main(sys.argv)
