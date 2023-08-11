import sys
import os
import datetime
import subprocess

def main(args):

    """ StableDiffusionを使って画像を生成します。
    """


    # venv上のpythonパスとt2iのスクリプトへのパス
    venvPython = ".\\venv\\Scripts\\python.exe"
    txt2imgPy = ".\\text2img_ov.py"

    # 設定。一応複数持てるように。
    # スケジューラーやシード、画像サイズなんかをプロンプトごとに変えられるようにした。
    configs = [
        {
            "useGpu": False,
            "models": [
                "models\\SakuraMix-v4-ov",
                "models\\AnzuMix-v1-ov",
            ],
            "prompt": [
                "crystal, 1girl, from below, looking down, sitting on bed, hand between legs, (flat chest:1.2), red eyes, (black long hair:1.3), pink frill thigh high socks, white and pink frill lolita, twintails, no shoes, (smile, blush), bed room, at night",
            ], 
            "negativePrompt": "(pink hair), (bad anatomy:1.3), (mutated legs:1.3), (bad feet:1.3), (bad legs:1.3), (missing legs:1.3), (extra legs:1.3), (bad hands:1.2), (mutated hands and fingers:1.2), (bad proportions), lip, nose, tooth, rouge, lipstick, eyeshadow, flat color, flat shading, (jpeg artifacts:1.4), (depth of field, bokeh, blurry, film grain, chromatic aberration, lens flare:1.0), (1boy, abs, muscular, rib:1.0), greyscale, monochrome, dusty sunbeams, trembling, motion lines, motion blur, emphasis lines, text",
            "width": "512",
            "height": "512",
            "outputDir": "G:\\マイドライブ\\StableDiffusion\\output",
            # 一回のプロンプトで何枚画像を作るか（内回り）
            "numImages": "4",
            #"seed": "253912154",
            "seed": "-1",
            "guidanceScale": "9.5",
            # inferense steps
            "steps": "15",
            # scheduler は次の3つから選択らしい。本家？の方はたくさんあるんだけどなあ。 [ "DDIMScheduler", "PNDMScheduler", "LMSDiscreteScheduler"],
            "scheduler": "DDIMScheduler",
            "dumpSetting": True,
        },
        {
            "useGpu": False,
            "models": [
                "models\\SakuraMix-v4-ov",
                "models\\AnzuMix-v1-ov",
            ],
            "prompt": [
                "crystal, 1girl, from below, looking at viewer, lie on bed on stomach, snooze, (flat chest:1.2), red eyes, (black long hair:1.3), pink frill thigh high socks, white and pink lolita, twintails, no shoes, (smile, blush), bed room, at night",
            ], 
            "negativePrompt": "(pink hair), (bad anatomy:1.3), (mutated legs:1.3), (bad feet:1.3), (bad legs:1.3), (missing legs:1.3), (extra legs:1.3), (bad hands:1.2), (mutated hands and fingers:1.2), (bad proportions), lip, nose, tooth, rouge, lipstick, eyeshadow, flat color, flat shading, (jpeg artifacts:1.4), (depth of field, bokeh, blurry, film grain, chromatic aberration, lens flare:1.0), (1boy, abs, muscular, rib:1.0), greyscale, monochrome, dusty sunbeams, trembling, motion lines, motion blur, emphasis lines, text",
            "width": "512",
            "height": "384",
            "outputDir": "G:\\マイドライブ\\StableDiffusion\\output",
            # 一回のプロンプトで何枚画像を作るか（内回り）
            "numImages": "4",
            #"seed": "253912154",
            "seed": "-1",
            "guidanceScale": "9.5",
            # inferense steps
            "steps": "15",
            # scheduler は次の3つから選択らしい。本家？の方はたくさんあるんだけどなあ。 [ "DDIMScheduler", "PNDMScheduler", "LMSDiscreteScheduler"],
            # DDIM が一番良好ぽい。
            "scheduler": "DDIMScheduler",
            "dumpSetting": True,
        },
    ]

    # プロンプトごとモデルごとに回る。
    # 1つのプロンプトを色々なモデルで試したいので。。 
    for config in configs:
        for prompt in config['prompt']:
            for model in config['models']:
            
                modelName = os.path.splitext(os.path.basename(model))[0]
                now = datetime.datetime.now()

                outputDir = config['outputDir']
                outputDir = os.path.join(outputDir, f"{now:%Y%m%d}")
                os.makedirs(outputDir, exist_ok=True)

                # パラメータくっつけて自作のopenvino版t2iを起動。
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
