import sys
import os
import datetime
import subprocess

def main(args):

    """ StableDiffusionを使って画像を生成します。
    """

    # venv上のpythonパスとt2iのスクリプトへのパス
    venvPython = ".\\venv\\Scripts\\python.exe"
    txt2imgPy = ".\\text2img_intel.py"

    config = {
      "useGpu": True,
      "models": [
       # "models\\IrisMix-v3-ov",
        "models\\SakuraMix-v2.1-ov",
       # "models\\HimawariMix-v8-ov",
       # "models\\KaedeMix-v2A-ov",
       # "models\\AsagaoMix-v1-ov",
        "models\\AnzuMix-v1-ov",
      ],
      "prompt": [
        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, shift dress, (black over the knee socks), no shoes, (happy), blush, bed room",
        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, corset dress, (black over the knee socks), no shoes, (happy), blush, bed room",
        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, tank dress, (black over the knee socks), no shoes, (happy), blush, bed room",
        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, peasant dress, (black over the knee socks), no shoes, (happy), blush, bed room",
        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, shirt dress, (black over the knee socks), no shoes, (happy), blush, bed room",
        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, fortuny pleat, (black over the knee socks), no shoes, (happy), blush, bed room",
        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, princess line dress, (black over the knee socks), no shoes, (happy), blush, bed room",
        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, slender line dress, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, white corset dress, red pleated skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, white corset dress, red kilt skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, white corset dress, red flared skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, white corset dress, red corset skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, white corset dress, red tulle skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, white corset dress, red hakama, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, white gimp, red short skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, white blouse shirt, red short skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, white bustier, red short skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, white bustier tops, red short skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, white camisole, red short skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, white camisole blouse, red short skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, white crop top, red short skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (full body), (flat chest:1.2), big red eyes, black hair, twintails, white short sleeve shirt, red short skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (flat chest:1.2), red eyes, black hair, twintails, black corset dress with red short skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (flat chest:1.2), big red eyes, black hair, twintails, white frilled dress, red short skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (flat chest:1.2), red eyes, black hair, twintails, white frilled princess dress, red short skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (flat chest:1.2), red eyes, black hair, twintails, white frilled blouse, red short skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (flat chest:1.2), red eyes, black hair, twintails, white lolita, red short skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (flat chest:1.2), red eyes, black hair, twintails, white maid, red short skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, (dakimakura), (flat chest:1.2), red eyes, black hair, twintails, white shirt, red short skirt, (black over the knee socks), no shoes, (happy), blush, bed room",
#        "crystal, 1girl, full body, (flat chest:1.2), red eyes, black hair, twintails, black corset dress with red short skirt, (black over the knee socks), no shoes, (happy), blush, sit on a cushion, bed room",
#        "kawaii, crystal, 1girl, (flat chest:1.2), big red eyes, black hair, twintails, white blouse, red short skirt, (black over the knee socks:1.2), no shoes, (happy), blush, sit on a cushion, in the bed room",
#        "kawaii, crystal, 1girl, (flat chest:1.2), big red eyes, black hair, twintails, white blouse, red short skirt, (black over the knee socks:1.2), no shoes, (happy), blush, dakimakura, in the bed room",
      ],
      "negativePrompt": "(nsfw:1.4), (worst quality, low quality:1.4), (lip, nose, tooth, rouge, lipstick, eyeshadow:1.4), bad hands, mutated hands and fingers, bad feet, bad legs, flat color, flat shading, (jpeg artifacts:1.4), (depth of field, bokeh, blurry, film grain, chromatic aberration, lens flare:1.0), (1boy, abs, muscular, rib:1.0), greyscale, monochrome, dusty sunbeams, trembling, motion lines, motion blur, emphasis lines, text, title, logo, signature",
      # このフォルダの下に %Y%m%dフォルダが作られてそこに出力画像が格納される
      "outputDir": "G:\\マイドライブ\\StableDiffusion\\output",
      "width": "512",
      "height": "512",
      # 一回のプロンプトで何枚画像を作るか（内回り）
      "numImages": "1",
      "seed": "253912154",
      "guidanceScale": "7",
      # inferense steps
      "steps": "50",
      # scheduler は次の3つから選択らしい。本家？の方はたくさんあるんだけどなあ。 [ "DDIMScheduler", "PNDMScheduler", "LMSDiscreteScheduler"],
      "scheduler": "LMSDiscreteScheduler",
      "dumpSetting": True,
    }

    # バックアップ用 ...
    inputData_bk = {
      "_prompt": "masterpiece, best quality, absurdres, ultra detailed beautiful face and eyes, 1girl, full body, (flat chest:1.2), (large red eyes), black hair, twintails, black corset dress with short skirt, black over the knee socks, no shoes, open mouth, happy, blush, sit on a cushion, bed room",
      "__prompt": "masterpiece, best quality, absurdres, ultra detailed beautiful face and eyes, 1girl, full body, (flat chest:1.2), (large red eyes), black hair, twintails, frilled blouse, short skirt, black over the knee socks, no shoes, open mouth, happy, blush, sit on a cushion, in the bed room",
    }

    # プロンプトごとモデルごとに回る。
    # 1つのプロンプトを色々なモデルで試したいので。。 
    for prompt in config['prompt']:
        for model in config['models']:
        
            modelName = os.path.splitext(os.path.basename(model))[0]
            now = datetime.datetime.now()

            outputDir = config['outputDir']
            outputDir = os.path.join(outputDir, f"{now:%Y%m%d}")
            os.makedirs(outputDir, exist_ok=True)

            args = [
                venvPython,
                txt2imgPy,
                '--model',
                model,
                '--prompt',
                prompt,
                '--negative_prompt',
                config['negativePrompt'],
                '--width',
                str(config['width']),
                '--height',
                str(config['height']),
                '--num_images_per_prompt',
                str(config['numImages']),
                '--seed',
                str(config['seed']),
                '--guidance_scale',
                str(config['guidanceScale']),
                '--steps',
                str(config['steps']),
                '--scheduler',
                config['scheduler'],
                '--output_dir',
                outputDir,
            ]
            if config['useGpu']:
                args.append('--gpu')

            if config['dumpSetting']:
                args.append('--dump_setting')

            subprocess.run(args, shell=True)

    print("done")

main(sys.argv)
