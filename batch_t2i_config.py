def make_configs(): 

    """
    make batch_text2img_ov.py's config file

    config format
    [
        {
            # if False skip this config
            ? "enable": <Boolean> = True,
            "use_gpu": <Boolean>,
            # if True use openvion else diffusers
            "use_openvino": <Boolean>,
            # safetensors or openvino ir model
            "models": [
                <Text|Path>,
            ],
            # prompt
            "prompt"[
                <Text>,
            ]
            # negative prompt
            "negative_prompt": <Text>,
            "width": <Integer>,
            "height": <Integer>,
            "output_dir": <Path>,
            # inference count per prompt
            "batch_count": <Integer>,
            # num images per prompt
            "batch_size": <Integer>,
            "seed": <Integer>,
            "guidance_scale": <Float>,
            "steps": <Integer>,
            "scheduler": <Text>,
            "dump_seting": <Boolean>,
            ? "lora": [
                {
                    "path": <Text>,
                    "scale": <Float>.
                }
            ],
            ? "lora_scale": <Float>,
            ? "vae_path": <Text>,
        }
    ]
    """

    configs = [
        {
            "use_gpu": True, 
            "use_openvino": True,
            "models": [
            #    "anzu_flat",
                r"C:\Users\webnu\source\repos\StableDiffusion\stable-diffusion-webui\models\Stable-diffusion\AsagaoMix-v2.safetensors",
            #    r"models\IrisMix-v5b-ov",
            #    r"models\AnzuMix-v1-ov",
            ],
            "prompt": [
                # 下アングル(from below)でこちらを見つめる(縦長画像のほうが意図通り出やすい) warizaを入れるとあんまり下からにならない。
                # 上アングル(from above)
                # 後ろから(from behind)
                # 上目遣い(looking up)
                # 女の子ずわり(wariza)
                # 四つん這い（on all fours）
                # フリむむ(looking back)
                # 不安(uneasy)
                # よろこび(happy)
                # 恥じらい(embarrassed)
                # お祈り（own hands together)
                # りょうてを足の間に(hand between legs)
                # 衣装（corset dress, maid, camisole, negligee, baby doll, maid, shirt, short skirt, kilt skirt, pleated skirt, gothic lolita）
                # 髪型（twintails, straight hair）
                # おばにー（thigh high socks, over the knee socksよりthigh high socksのほうが出やすい）
                # 水着（swimsuit, bikini, strings bikini）
                # 巫女さん全身
                "1girl, looking at viewer,  \
sitting, own hands together, \
(flat chest:1.2), red eyes, \
(bright black long hair:1.2), twintails, \
frill white miko, red short skirt, \
(embarrassed), \
at japanese garden", 
##                # 鉄板
##                "1girl, from above, looking down,  \
##wariza, \
##(flat chest:1.2), red eyes, \
##(bright black long hair:1.2), twintails, \
##frill pink negligee, red ribbons, no shoes, \
##(embarrassed), \
##in hotel bedroom, winter, at night",

            ], 
            "negative_prompt": "<EasyNegative>, text, nsfw",
            "width": "512",
            "height": "768",
            "output_dir": r"G:\マイドライブ\StableDiffusion\output",
            "batch_count": "2",
            # 一回のプロンプトで何枚画像を作るか（内回り）
            "batch_size": "1",
            "seed": "-1",
            "guidance_scale": "9.5",
            # inferense steps
            "steps": "30",
            "scheduler": "DPM++ 2M Karras",
            "dump_setting": True,
            # diffusersであれば効くようになった。
            "lora": [
                {
                    "path": r"C:\Users\webnu\source\repos\StableDiffusion\stable-diffusion-webui\models\Lora\flat2.safetensors",
                    "scale": -0.5,
                }
            ],
            #"lora_scale": 1.5,
        },
        {
            "enable": False, 
            "use_gpu": True,
            # openvinoを使う場合、vae_path、lora, lora_scaleは無視される。
            "use_openvino": True,
            "models": [
                r"models\IrisMix-v5-ov",
                r"models\IrisMix-v5b-ov",
                r"models\AnzuMix-v1-ov",
            ],
            # promptとnegative_promptに<>でくくった文字列を記載すると、textual-inversionsフォルダの同名のファイル(.safetensorか.pt)を探して読み込みます
            "prompt": [
                # 下アングル(from below)でこちらを見つめる(縦長画像のほうが意図通り出やすい) warizaを入れるとあんまり下からにならない。
                # 上アングル(from above)
                # 後ろから(from behind)
                # 上目遣い(looking up)
                # 女の子ずわり(wariza)
                # 四つん這い（on all fours）
                # フリむむ(looking back)
                # 不安(uneasy)
                # よろこび(happy)
                # 恥じらい(embarrassed)
                # お祈り（own hands together)
                # りょうてを足の間に(hand between legs)
                # 衣装（corset dress, maid, camisole, negligee, baby doll, maid, shirt, short skirt, kilt skirt, pleated skirt, gothic lolita）
                # 髪型（twintails, straight hair）
                # おばにー（thigh high socks, over the knee socksよりthigh high socksのほうが出やすい）

                # 上アングルでこちらを見上げる
                "1girl, from above, looking up, \
wariza, hand between legs, \
(flat chest:1.2), red eyes, \
pink frill baby doll no shoes, \
bright black long hair, twintails, \
(embarrassed), \
in hotel bedroom",

                "1girl, from above, looking up, \
wariza, hand between legs, \
(flat chest:1.2), red eyes, \
pink frill negligee, no shoes, \
bright black long hair, twintails, \
(embarrassed), \
in hotel bedroom",

                # 立ち絵
                "1girl, full body, looking at viewer, \
(flat chest:1.2), red eyes, \
frill white thigh high socks, white camisole, red short pleated skirt, \
bright black long hair, twintails, \
(smile, happy), \
in the park, at morning",
            ], 
            #"negative_prompt": "(bad anatomy:1.3), (mutated legs:1.3), (bad feet:1.3), (bad legs:1.3), (missing legs:1.3), (extra legs:1.3), (bad hands:1.2), (mutated hands and fingers:1.2), (bad proportions), lip, nose, tooth, rouge, lipstick, eyeshadow, flat color, flat shading, (jpeg artifacts:1.4), (depth of field, bokeh, blurry, film grain, chromatic aberration, lens flare:1.0), (1boy, abs, muscular, rib:1.0), greyscale, monochrome, dusty sunbeams, trembling, motion lines, motion blur, emphasis lines, text",
            "negative_prompt": "<EasyNegative>, text",
            "width": "512",
            "height": "768",
            "output_dir": r"G:\マイドライブ\StableDiffusion\output",
            # 一回のプロンプトで何枚画像を作るか（内回り）
            "batch_size": "1",
            # 一回のプロンプトで作成を何回ループするか（外回り）
            "batch_count": "4",
            "seed": "-1",
            "guidance_scale": "9.5",
            # inferense steps
            "steps": "20",
            # scheduler は次の3つから選択らしい。本家？の方はたくさんあるんだけどなあ。 [ "DDIMScheduler", "PNDMScheduler", "LMSDiscreteScheduler"],
            # openvino版もdiffusers.schedulersのスケジューラが使えるぽい。単に初期値としては先の３つで、あとからぶっこむ場合は大丈夫みたい。
            # ～ a シリーズは Ancestral がつくやつっぽい。
            # Karrasはinit時にuse_karras_sigmas=True とするとものによっては有効になるらしい。
            # from_config()に**kwargsが渡せるので、use_karras_sigmasはそこで指定医して渡すとよさげ。
            # DPMSolverMultistepScheduelr: DPM++ 2M
            # EulerDiscreteScheduler: Euler
            # EulerAncestralScheduler: Euler a
            # A1111のサンプラーの名前も利用可能
            "scheduler": "DPMSolverMultistepScheduler",
            "dump_setting": True,
            # For original diffusers pipeline (not openvino)
            # vae_pathはVAE内臓のモデルだったら不要。内蔵でもvae_pathが指定されれば多分これを優先する。
            "vae_path": r"C:\Users\webnu\source\repos\StableDiffusion\stable-diffusion-webui\models\VAE\clearvae_v23.safetensors",
            # LoRAは重ねがけができるらしい。
            "lora": [
                {
                    "path": r"C:\Users\webnu\source\repos\StableDiffusion\stable-diffusion-webui\models\Lora\flat2.safetensors",
                    "scale": 1.0,
                }
            ],
            "lora_scale": -1.5,
        },
        {
            # この設定を飛ばす場合はFalseに設定。未定義、Trueの場合は使う。
            "enable": False,
            "use_gpu": False,
            "use_openvino": False,
            "models": [
                #r"C:\Users\webnu\source\repos\StableDiffusion\stable-diffusion-webui\models\Stable-diffusion\AnzuMix-v1.safetensors",
                r"C:\Users\webnu\source\repos\StableDiffusion\stable-diffusion-webui\models\Stable-diffusion\blue_pencil-v10.safetensors",
            ],
            "prompt": [
                "crystal, 1girl, \
                from below, looking at viewer, lie on bed on stomach, snooze, \
                (black long hair:1.3), twintails, \
                (flat chest:1.2), red eyes, \
                pink frill thigh high socks, white and pink negligee, no shoes, \
                (smile, blush), \
                bed room, at night",
            ], 
            "negative_prompt": "(pink hair), (bad anatomy:1.3), (mutated legs:1.3), (bad feet:1.3), (bad legs:1.3), (missing legs:1.3), (extra legs:1.3), (bad hands:1.2), (mutated hands and fingers:1.2), (bad proportions), lip, nose, tooth, rouge, lipstick, eyeshadow, flat color, flat shading, (jpeg artifacts:1.4), (depth of field, bokeh, blurry, film grain, chromatic aberration, lens flare:1.0), (1boy, abs, muscular, rib:1.0), greyscale, monochrome, dusty sunbeams, trembling, motion lines, motion blur, emphasis lines, text",
            "width": "512",
            "height": "512",
            "output_dir": r"G:\マイドライブ\StableDiffusion\output",
            "batch_count": "4",
            # 一回のプロンプトで何枚画像を作るか（内回り）
            "batch_size": "1",
            #"seed": "253912154",
            "seed": "-1",
            "guidance_scale": "9.5",
            # inferense steps
            "steps": "15",
            # scheduler は次の3つから選択らしい。本家？の方はたくさんあるんだけどなあ。 [ "DDIMScheduler", "PNDMScheduler", "LMSDiscreteScheduler"],
            # openvino版もdiffusers.schedulersのスケジューラが使えるぽい。単に初期値としては先の３つで、あとからぶっこむ場合は大丈夫みたい。
            "scheduler": "EulerAncestralDiscreteScheduler",
            "dump_setting": True,
            # For original Pipeline
            # vae_pathはVAE内臓のモデルだったら不要。内蔵でもvae_pathが指定されれば多分これを優先する。
            "vae_path": r"C:\Users\webnu\source\repos\StableDiffusion\stable-diffusion-webui\models\VAE\clearvae_v23.safetensors",
            # LoRAは重ねがけができるらしい。
            "lora": [
                {
                    "path": r"C:\Users\webnu\source\repos\StableDiffusion\stable-diffusion-webui\models\Lora\flat2.safetensors",
                    "scale": 1.0,
                }
            ],
            "lora_scale": -1.5,
        },
    ]
    
    return configs