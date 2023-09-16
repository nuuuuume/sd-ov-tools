import sys
import os
import datetime
import time
import argparse
import json
import re
import numpy as np
import torch
import xml.etree.ElementTree as ET
from pathlib import Path
from diffusers import StableDiffusionPipeline, AutoencoderKL
from openvino.runtime import Core, Model, Type
from optimum.intel import OVStableDiffusionPipeline
from openvino.runtime.passes import Manager, GraphRewrite, MatcherPass, WrapType, Matcher
from openvino.runtime import opset11 as ops
from lauda import stopwatch, stopwatchcm
from diffusers.schedulers import *
from safetensors.torch import load_file
from remove_blob import remove_model_blobs

class InsertLoRA(MatcherPass):
    
    # https://github.com/FionaZZ92/OpenVINO_sample/blob/master/SD_controlnet/run_pipe.py
    # このソースをべーすに改造した。

    def __init__(self,lora_dict_list):
        MatcherPass.__init__(self)
        self.model_changed = False

        param = WrapType("opset11.Constant")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            root_output = matcher.get_match_value()
            for y in lora_dict_list:
                # y=loraの方は 以下のような値。
                # up_blocks_2_attentions_1_transformer_blocks_0_attn1_to_v 
                # up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_k 
                # up_blocks_2_attentions_1_transformer_blocks_0_attn2_to_q 
                # up_blocks_2_attentions_1_transformer_blocks_0_ff_net_0_proj
                # modelの方は以下のような値（変換前）
                # /up_blocks.1/attentions.0/transformer_blocks.0/attn1/Cast_1
                # /up_blocks.1/attentions.0/transformer_blocks.0/attn1/Cast_2
                # /up_blocks.1/attentions.0/transformer_blocks.0/attn1/Cast_3
                # みたいな値。
                # 戦闘の /　を消す、/と.を_に変換すると同じような書式になるが、Cast_1あたりの部分が合わない。
                # なのでこのコードだとAddに入らず何も起きない.
                # to_q,k,vはMatMulにしかないので、そのあたりで披露か。
                 if root.get_friendly_name() == y["name"]:
                    consumers = root_output.get_target_inputs()
                    # openvinoはsdからみて行列が反転している？？
                    lora_weights = ops.constant(y["value"].mT, Type.f32, name=y["name"])
                    #print(f"matched! node shape: {root.shape} lora_weights shape: {lora_weights.shape}")
                    add_lora = ops.add(root, lora_weights, auto_broadcast='numpy')
                    for consumer in consumers:
                        consumer.replace_source_output(add_lora.output(0))

                    # For testing purpose
                    self.model_changed = True
                    # Use new operation for additional matching
                    self.register_new_node(add_lora)

            # Root node wasn't replaced or changed
            return False

        self.register_matcher(Matcher(param,"InsertLoRA"), callback)

def apply_ov_lora_model(pipe, state_dict_list, scale_list, ov_unet_model_xml_path, ov_text_encoder_model_path):

    """
    add lora weights
    
    parameters:
        pipe: 
            openvino stablediffusion pipeline
        state_dict_list:
            lora weight list
        scale_list:
            lora scale list 
        
    """

    # https://github.com/FionaZZ92/OpenVINO_sample/blob/master/SD_controlnet/run_pipe.py
    # このソースをべーすに改造した。

    visited = []
    lora_dict = {}
    lora_dict_list = []
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    flag = 0
    manager = Manager()
    # この辺は大丈夫ぽい。
    for iter in range(len(state_dict_list)):
        visited = []
        for key in state_dict_list[iter]:
            if ".alpha" in key or key in visited:
                continue
            if "text" in key:
                layer_infos = key.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split(".")[0]
                lora_dict = dict(name=layer_infos)
                lora_dict.update(type="text_encoder")
            else:
                layer_infos = key.split(LORA_PREFIX_UNET + "_")[1].split('.')[0]
                lora_dict = dict(name=layer_infos)
                lora_dict.update(type="unet")
            pair_keys = []
            if "lora_down" in key:
                pair_keys.append(key.replace("lora_down", "lora_up"))
                pair_keys.append(key)
            else:
                pair_keys.append(key)
                pair_keys.append(key.replace("lora_up", "lora_down"))

            # update weight
            if len(state_dict_list[iter][pair_keys[0]].shape) == 4:
                # len(shape) == 4 のは proj_in proj_outくらいらしい。
                weight_up = state_dict_list[iter][pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
                weight_down = state_dict_list[iter][pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
                # ここでloraに設定した係数（scale値、、1とか0.5とかflat2だと-1とか）がのる？
                lora_weights = scale_list[iter] * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                lora_dict.update(value=lora_weights)
            else:
                # その他、は2次元。基本はこちらの値を相手にする形になるのかねえ？
                weight_up = state_dict_list[iter][pair_keys[0]].to(torch.float32)
                weight_down = state_dict_list[iter][pair_keys[1]].to(torch.float32)
                # ここでloraに設定した係数（scale値、、1とか0.5とかflat2だと-1とか）がのる？
                lora_weights = scale_list[iter] * torch.mm(weight_up, weight_down)
                lora_dict.update(value=lora_weights)

            # check if this layer has been appended in lora_dict_list
            for ll in lora_dict_list:
                if ll["name"] == lora_dict["name"]:
                    ll["value"] += lora_dict["value"] # all lora weights added together
                    flag = 1
            if flag == 0:
                lora_dict_list.append(lora_dict)
            # update visited list
            for item in pair_keys:
                visited.append(item)
            flag = 0

    # sdのname -> ov name
    ov_lora_dict_list = map_ov_and_lora(lora_dict_list, ov_unet_model_xml_path, ov_text_encoder_model_path)

    # 上で作った加工済みのlora情報をregister_pass -> run_passesと流し、その中でConvertの値にlora_の値をAddする、
    # という流れになっているポイ。
    manager.register_pass(InsertLoRA(ov_lora_dict_list))
    if (True in [('type', 'text_encoder') in l.items() for l in lora_dict_list]):
        print("--- text encoder run_passes")
        rp_ret = manager.run_passes(pipe.text_encoder.model)
        print(f"--- run_passes result: {rp_ret}")

    print("--- unet run_passes")
    rp_ret = manager.run_passes(pipe.unet.model)
    print(f"--- run_passes result: {rp_ret}")

def map_ov_and_lora(lora_dict_list: list, 
                    ov_unet_xml_path: str, 
                    ov_text_encoder_xml_path: str):

    """
    loraのキーをopenvinoのキーにマップします

    Parameters:
        lora_dict_list: list
            loraの読込み結果リスト

        ov_unet_xml_path: str

        ov_text_encoder_xml_path:
    """

    # text_encoderはあったりなかったりだけど、とりあえず全部読んでまとめて処理してしまおう。。。
    for xml in [ov_unet_xml_path, ov_text_encoder_xml_path]:

        with open(xml, 'rt') as f:
            buf = f.read()
        root = ET.fromstring(buf)      
        layers = root.find('layers')
        edges = root.find('edges')

        # constを探してその中でname=onnx:: で始まるlayerを見つける
        # 見つけたlayerのidをedgesのfrom-layerで見つけ、to-layerを取得
        # to-layerをidにもつlayerのnameをsdの形式に変換してlora_dict_listのnameとマッチさせる
        # マッチしたlora_dict_listのnameをfrom-layerのnameに置換する。
        # で、この関数は終わり。
        # その後はrun_passesに流してConstantをMatchさせ、nameが一致する値にweight（value）を足し込む
        # ようにしてみる。
        ret = []
        const_layer_list = layers.findall("./layer[@type='Const']")
        for cl in const_layer_list:

            # lnameはこんな感じのものが入っている
            # onnx::MatMul_9138
            lname = cl.attrib['name']
            # onnx::MatMul始まりと .weight 終わりのみを対象にする
            if not lname.startswith('onnx::MatMul') and not lname.endswith('.weight'):
                continue

            lid = cl.attrib['id']
            # edgeを探す
            edge = edges.find(f"./edge[@from-layer='{lid}']")
            if edge == None:
                continue

            # 見つけたedgeからto-layerを取得、そのidをもつ
            # to-layer が layer の id となる。
            toid = edge.attrib['to-layer']
            to_layer = layers.find(f"./layer[@id='{toid}']")
            if to_layer == None:
                continue

            target_name = lname if lname.endswith('.weight')  else to_layer.attrib['name']

            # 【unet】
            # ・to_q, to_k, to_v
            # onnx::MatMul始まりのlayerを探す。見つかったidでedge.from-layerを検索し、ヒットしたedgeのto-layerをidにもつlayerを取得
            # そのlayer.nameを見ると以下のようなものがある。
            # /down_blocks.0/attentions.1/transformer_blocks.0/attn2/to_q/MatMul
            # ・proj_in, proj_out
            # こちらはto_qなどのような形でなく、const値としてダイレクトに下記のようなnameを持つlayerがあるので、それを使う
            # down_blocks.0.attentions.0.proj_in.weight
            # loraのキーはこんな感じ
            # ・to_q, to_k, to_v
            # lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight
            # ・proj_in, proj_out
            # lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight
            # 実際は lora_unet_ は削除されていて、.以降は削除されているので、以下のようになる。
            # ・to_q, to_k, to_v
            # down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k
            # ・proj_in, proj_out
            # down_blocks_0_attentions_0_proj_in
            # down_blocks_0_attentions_0_proj_out
            # down_blocks_0_attentions_0.proj_in
            # なので、to_layerのnameをこれに合わせる必要がある。
            # 【text_encoder】
            # 基本的にはonnx::MatMul始まりのconstを見つけて行き先のnameを取得するというunetのto_qとかのやり方と同じっぽい。
            # proj_inみたいな方式の直接参照できる形のものはなさそう。
            # nameの種類としては大きくこの２パターン。両方ともloraにも似たような定義があったので置換ルールはunetと同じで行けそう
            # 45:onnx::MatMul_2260:0[Const] -> 46:/text_model/encoder/layers.0/self_attn/q_proj/MatMul:1[MatMul]
            # 253:onnx::MatMul_2281:0[Const] -> 254:/text_model/encoder/layers.0/mlp/fc1/MatMul:1[MatMul]
            # lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight torch.Size([8, 768])
            # lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_up.weight torch.Size([768, 8])
            # lora_te_text_model_encoder_layers_10_mlp_fc1.lora_down.weight torch.Size([8, 768])
            # lora_te_text_model_encoder_layers_10_mlp_fc1.lora_up.weight torch.Size([3072, 8])
            # 具体的には
            # ・/を_に変換
            # ・.を_に変換
            # ・先頭が_だったら取り除く
            # ・行末の _MatMulを取り除く
            # ・行末の_weightを取り除く
            target_name = target_name.replace('/', '_').replace('.', '_').replace('_MatMul', '').replace('_weight', '')
            if target_name[0] == '_':
                target_name = target_name[1:]

            #print(f"{to_layer.attrib['name']} -> {target_name}")
            # lora_state_dictで同じnameを持つものを探す
            # 見つかったら、そのvalueをfrom-layerのnameをキーとした辞書に保存
            for ll in lora_dict_list:
                if ll["name"] == target_name:
                    ll["name"] = lname

    return lora_dict_list

def print_elapsed(watch, function):
    m, s = divmod(watch.elapsed_time, 60)
    h, m = divmod(m, 60)
    
    print(f"<<< {int(h)}:{int(m)}:{int(s)} spent.")

def print_elapsed_with(watch):
    m, s = divmod(watch.elapsed_time, 60)
    h, m = divmod(m, 60)
    
    print(f"<<< {int(h)}:{int(m)}:{int(s)} spent.")


def create_scheduler(pipe, a1111name):

    """
    A1111で使われるサンプラーの文字列や、diffusersのSchedulerのクラス名からスケジューラーを作って返します。
    HuggingFaceに乗っているマッピングをべーすに一部変更している。
    DPM++ 2M	DPMSolverMultistepScheduler	
    DPM++ 2M Karras	DPMSolverMultistepScheduler	init with use_karras_sigmas=True
    DPM++ 2M SDE	DPMSolverMultistepScheduler	init with algorithm_type="sde-dpmsolver++"
    DPM++ 2M SDE Karras	DPMSolverMultistepScheduler	init with use_karras_sigmas=True and algorithm_type="sde-dpmsolver++"
    DPM++ 2S a	N/A	very similar to DPMSolverSinglestepScheduler
    DPM++ 2S a Karras	N/A	very similar to DPMSolverSinglestepScheduler(use_karras_sigmas=True, ...)
    DPM++ SDE	DPMSolverSinglestepScheduler	
    DPM++ SDE Karras	DPMSolverSinglestepScheduler	init with use_karras_sigmas=True
    DPM2	KDPM2DiscreteScheduler	
    DPM2 Karras	KDPM2DiscreteScheduler	init with use_karras_sigmas=True
    DPM2 a	KDPM2AncestralDiscreteScheduler	
    DPM2 a Karras	KDPM2AncestralDiscreteScheduler	init with use_karras_sigmas=True
    DPM adaptive	N/A	
    DPM fast	N/A	
    Euler	EulerDiscreteScheduler	
    Euler a	EulerAncestralDiscreteScheduler	
    Heun	HeunDiscreteScheduler	
    LMS	LMSDiscreteScheduler	
    LMS Karras	LMSDiscreteScheduler	init with use_karras_sigmas=True
    N/A	DEISMultistepScheduler	
    UniPC   UniPCMultistepScheduler
    """
    name_config_dict = {
        "DPM++ 2M": {
            "class": "PMSolverMultistepScheduler",
            "config": {}
        },
        "DPM++ 2M KARRAS": {
            "class": "DPMSolverMultistepScheduler",
            "config": {
                "use_karras_sigmas": True,
            }
        },
        "DPM++ 2M SDE": {
            "class": "DPMSolverMultistepScheduler",
            "config": {
                "algorithm_type": "sde-dpmsolver++",
            },
        },
        "DPM++ 2M SDE Karras": {
            "class": "DPMSolverMultistepScheduler",
            "config": {
                "use_karras_sigmas": True,
                "algorithm_type": "sde-dpmsolver++",
            },
        },
        "DPM++ 2S A": {
            "class": "DPMSolverSinglestepScheduler",
            "config": {},
        },
        "DPM++ 2S A KARRAS": {
            "class": "DPMSolverSinglestepScheduler",
            "config": {
                "karras_sigmas": True,
            },
        },
        "DPM++ SDE": {
            "class": "DPMSolverSinglestepScheduler",
            "config": {},
        },
        "DPM++ SDE KARRAS": {
            "class": "DPMSolverSinglestepScheduler",
            "config": {
                "use_karras_sigmas": True,
            },
        },
        "DPM2": {
            "class": "KDPM2DiscreteScheduler",
            "config": {},
        },
        "DPM2 KARRAS": {
            "class": "KDPM2DiscreteScheduler",
            "config": {
                "use_karras_sigmas": True,
            },
        },
        "DPM2 A": {
            "class": "KDPM2AncestralDiscreteScheduler",
            "config": {},
        },
        "DPM2 A KARRAS": {
            "class": "KDPM2AncestralDiscreteScheduler",
            "config": {
                "use_karras_sigmas": True,
            },
        },
        #DPM adaptive	N/A	
        #DPM fast	N/A	
        "EULER": {
            "class": "EulerDiscreteScheduler",
            "config": {},
        },
        "EULER A": {
            "class": "EulerAncestralDiscreteScheduler",
            "config": {},
        },
        "HEUN": {
            "class": "HeunDiscreteScheduler",
            "config": {},
        },
        "LMS": {
            "class": "LMSDiscreteScheduler",
            "config": {},
        },	
        "LMS Karras": {
            "class": "LMSDiscreteScheduler",
            "config": {
               "use_karras_sigmas": True,
            },
        },
        #N/A	DEISMultistepScheduler	
        "UniPC": {
            "class": "UniPCMultistepScheduler",
            "config": {},
        },
        "UniPC Karras": {
            "class": "UniPCMultistepScheduler",
            "config": {
                "use_karras_sigmas": True,
            },
        },
     } 

    class_name = a1111name
    a1111nameupper = a1111name.upper()
    config = {}
    if a1111nameupper in name_config_dict:
        class_name = name_config_dict[a1111nameupper]['class']
        config = name_config_dict[a1111nameupper]['config']

    print(f"--- scheduler {a1111name} -> {class_name}")
    print(f"--- config: {config}")

    return eval(class_name).from_config(pipe.scheduler.config, **config)

def make_diffusers_pipeline(args):

    """
    コマンドライン引数の情報からdiffusersのStableDiffusionPipelineとgeneratorを作成して返します
    """

    device_type = "cuda" if args.use_gpu else "cpu"
    torch_dtype = torch.float16 if args.use_gpu else torch.float32

    generator = torch.Generator(device=device_type).manual_seed(args.seed)

    vae = None    
    if args.vae_path:
        print(f"--- load vae from: {args.vae_path}")
        vae = AutoencoderKL.from_single_file(args.vae_path)

    if is_safetensor_file(args.model):
        print(f"--- model from file ... ")
        pipe = StableDiffusionPipeline.from_single_file(
            args.model,
            torch_dtype=torch_dtype,
            load_safety_checker=False,
            vae=vae)
    else:
        print(f"--- model from pretranined ... ")
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model,
            torch_dtype=torch_dtype)

        if pipe.safety_checker != None:
            pipe.safety_checker = no_safety_checker

    load_textual_inversion(pipe, args.prompt)
    load_textual_inversion(pipe, args.negative_prompt)

    if args.lora != None and len(args.lora) > 0:
        for lora in args.lora:
            print(f"--- load lora weight from: {lora}")
            path_and_scale = lora.split(':')
            scale = path_and_scale[-1] if len(path_and_scale) > 1 else 1.0
            path = ':'.join(path_and_scale[:-1])
            print(f"--- path:{path}")
            print(f"--- scale:{scale}")

            # LoRAの適用はdiffuers/scripts/convert_lora_safetensors_to_diffusers.pyの中身を参考に行う
            apply_diffusers_lora(pipe, path, float(scale))

    pipe = pipe.to(device_type)

    return (pipe, generator)

def apply_diffusers_lora(pipe, lora_path, scale):

    """
    diffusers モデルに対するLoRAの適用。
    diffusers.scripts.convert_lora_safetensors_to_diffusers.pyから拝借。
    これでscale（元のソースだとalpha）をのせてLoRAが効くことは確認した。
    """ 
    state_dict = load_file(lora_path)

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split("lora_te_")[-1].split("_")
            curr_layer = pipe.text_encoder
        else:
            layer_infos = key.split(".")[0].split("lora_unet_")[-1].split("_")
            curr_layer = pipe.unet

        # find the target layer
        # down blocks 0 attentions 0 transformer blocks 0 attn2 to v みたいに改装になっているので
        # その最後のレイヤーを探したいらしい？（上の例でいうとv のれいやー？）
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                # _ でsplitすると to_k to_v to_q も分断されてしまう。
                # __getattr__で例外に飛んでくるので、飛んできたら次の部品をくっつけると
                # to_v とかが復元されるぽい。
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        # ↑で探したlayerに対してLoRAのウェイトを足し込む。とLoRAがささる。
        # proj_in, proj_out, proj は LoRACompatibleConv レイヤー
        # to_q, to_k, to_v, out_0 は Linear レイヤー
        # にそれぞれ打ち込んでいるよう。
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += scale * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += scale * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)

def make_openvino_pipeline(args):

    """
    コマンドライン引数からOVStableDiffusionPipelineとgeneratorを作成して返します

    Arguments:
        args (`dict`):
            commandline arguments.
    """
    generator = np.random.RandomState(args.seed)

    unet_xml_path = Path(args.model) / "unet" / "openvino_model.xml"
    text_encoder_xml_path = Path(args.model) / "text_encoder" / "openvino_model.xml"

    # openvinoで読み込み
    pipe = OVStableDiffusionPipeline.from_pretrained(
        args.model,
        compile=False)

    # apply textual_inversion
    load_textual_inversion(pipe, args.prompt)
    load_textual_inversion(pipe, args.negative_prompt)

    # apply lora
    if args.lora != None and len(args.lora) > 0:
        state_dict = []
        scale_list = []
        for lora in args.lora:
            print(f"--- load lora weight from: {lora}")
            path_and_scale = lora.split(':')
            scale = path_and_scale[-1] if len(path_and_scale) > 1 else 1.0
            path = ':'.join(path_and_scale[:-1])
            print(f"--- path:{path}")
            print(f"--- scale:{scale}")
            state_dict.append(load_file(path)) #state_dict is list of lora list
            scale_list.append(float(scale))

        apply_ov_lora_model(pipe, state_dict, scale_list, unet_xml_path, text_encoder_xml_path)
   
    if args.use_gpu:
        # GPUを使う場合は以下を実行。t2iは動くけどi2iは画像が真っ黒になった（CPUだとi2iもちゃんとできる）
        # 11世代CoreのIrisXEだとCPUで回したほうが速度は出るという、、。
        pipe.to('GPU')

    # モデルのサイズを限定するとメモリが浮く 
    pipe.reshape(batch_size=1,
        height=args.height,
        width=args.width,
        num_images_per_prompt=args.batch_size)
 
    pipe.half()

    pipe.compile()
    return (pipe, generator)

def load_textual_inversion(pipe, prompt):
    """
    load textual inversion from safetensors

    Parameters:
        pipe (`TextualInversionLoaderMixin` or `OVTextualInversionLoaderMixin` ):
            pipeline.
        
        prompt (`str`):
            prompt or negative prompt.
        
    """

    # textual-inversionは プロンプトの中に<aaa>みたいに埋まるっぽいので、その値を取り出して
    # load_textual_inversionする
    tilist = re.findall(r'<([^<>]*)>', prompt)

    for ti in tilist:
        name = f'<{ti}>'
        # safetensorsがなかったらptで読んで見る
        file_name = f'{ti}.safetensors'
        file_path = os.path.join('textual-inversions', file_name)
        if not os.path.exists(file_path):
            file_path = os.path.join('textual-inversions', f'{ti}.pt')
        print(file_path) 
        pipe.load_textual_inversion(file_path, name)
        print(f"--- load text-inversion from {file_path} -> {name}")

def is_safetensor_file(model) -> bool:
    """ 
    input.jsonのmodelがsafetensorsファイルの場合Trueを返します。
    単にendswithしてるだけ。

    model (`str`):
        model file path.

    """
    return model.endswith("safetensors")

def no_safety_checker(images, **kwargs):

    """
    nsfwのチェッカーを無効にする関数
    """

    return images, [False] * len(images)

@stopwatch(callback=print_elapsed)
def main(args):

    print(f"--- use_gpu: {args.use_gpu}")
    print(f"--- model: {args.model}")
    print(f"--- output dir: {args.outputDir}")
    print(f"--- seed {args.seed}")
    print(f"--- widthxheight: {args.width}x{args.height}")
    print(f"--- prompt: {args.prompt}")
    print(f"--- negative_prompt: {args.negative_prompt}")
    print(f"--- inference steps: {args.steps}")
    print(f"--- guidance scale: {args.guidance_scale}")
    print(f"--- batch count: {args.batch_count}")
    print(f"--- num images per prompt(batch_size): {args.batch_size}")
    print(f"--- scheduler: {args.scheduler}")
    print(f"--- use openvino?: {args.use_openvino}")
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
    # Loraはunetに対して、 diffusers.loaders.LoraLoaderMixin.load_lora_weights(path or dict) を呼ぶとロードしてくれるらしい。
    # あとはトリガーワード（Loraをよく効かせるためのワード）があればpromptに指定するだけっぽい。
    # webuiの場合はこのあたりを <lora:flat2:-1> みたいな独自のキーワードでで実装しているみたい。
    # この場合だと、flat2というloraを-1の倍率を乗っけて適用、みたいな。倍率は1が通常、0が無効、みたいな感じ。マイナス値は普通使わないらしいけど、
    # flat loraみたいなものは正の値にするとどんどん画像がベタ塗りになるが逆にマイナスになると凹凸が強調されるようになる。
    # openvinoの場合、Unetに相当するOVUnetModelにload_attn_procsがイないので今のところはムリポなのかな。
    # Loraは学習済みモデルに上乗せするものっぽいので、Loraを上乗せした状態のモデルを保存することができればあるいは。。
    # なんかopenvinoでLoRAの利用を実装しているコードがあった。あの変ヒントにすればなにかできるかも？
    # https://github.com/FionaZZ92/OpenVINO_sample/blob/master/SD_controlnet/run_pipe.py これ。
    # Controlnetとかの実装ぽいから解読して頭が足りてればーというところか、、。まずはこれ頂いて組み込んでみようかなあ。
    # ・Scheduler
    # ソースを見ると DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, の3つしかサポートしてないような感じだけど、
    # from_pretarainedしてから任意のスケジューラーをぶっこめば動くっぽい。diffusersのSchedulerが使えそう。本家のStableDiffusionPipelineもこれら3つ能から一つと
    # 書いてある割に別のデモいけるから、あくまで初期値としては、ってことなのかもしれないね。SchedulerMixinを実装してるやつ？Mixinといいながらベースクラスっぽいね？
    # class DDIMScheduler(SchedulerMixin, ConfigMixin): ←こんな感じの定義だし。
    
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    print(f"--- model: {model_name}")

    # remove blob files
    remove_model_blobs(args.model)

    # openvino版と普通のを合体させてみたけど、なんか微妙。。
    pipe, generator = make_openvino_pipeline(args) if args.use_openvino else make_diffusers_pipeline(args)
    pipe.scheduler = create_scheduler(pipe, args.scheduler)

    # 出力ファイルの末尾につける連番。 
    suffix = 1

    for _ in range(args.batch_count):
        with stopwatchcm(print_elapsed_with):
            # 推論実行。絵ができるよ。
            # LoRaの効き具合はcross_attention_kwargs.scale で指定可能、
            # 0だと効かない。
            # これだと全部のLoRaに同じ倍率が乗って舞う。
            # 一方で、loraをロードするときにもscaleがのせられるので個別の場合はそっちも使うのかな？
            # 一つしかLoraを使わないとロード時のscaleは全然効かない。
            if args.use_openvino:
                result = pipe(prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    width=args.width,
                    height=args.height,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=args.batch_size,
                    generator=generator,
                    num_inference_steps=args.steps)
            else:
                result = pipe(prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    width=args.width,
                    height=args.height,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=args.batch_size,
                    generator=generator,
                    num_inference_steps=args.steps,
                )

            # num_images_per_promptで指定した枚数の画像ができるのでそれぞれ保存
            while len(result.images) > 0:

                now = datetime.datetime.now()

                outputDir = args.outputDir
                os.makedirs(outputDir, exist_ok=True)

                file_nameBase = f"{now.strftime('%Y%m%d%H%M%S')}_{model_name}_{suffix:03}"
                file_name = f"{file_nameBase}.png"
                outputFilePath = os.path.join(outputDir, file_name)
                print(f"--- output file path: {outputFilePath}")
                result.images[0].save(outputFilePath)
                print(f"--- saved. {outputFilePath}")
                # pngにexifとして設定を埋め込むのがWebUIとかはやってるみたいだけど、
                # テキストファイルに落としておけばビューワもいらないからとりあえずテキストにダンプ。
                # 必要が出たらpngに埋めればいいんでなかな。
                if args.dump_setting:
                    dumpfile_name = f"{file_nameBase}.txt"
                    dumpFilePath = os.path.join(outputDir, dumpfile_name)
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
                        dest='negative_prompt',
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
    parser.add_argument('--batch_count',
                        dest='batch_count',
                        type=int,
                        action='store',
                        default=1,
                        help='generate count per prompt. no impact vram.')
    parser.add_argument('--batch_size', 
                        dest='batch_size',
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
                        dest='guidance_scale',
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
                        dest='use_gpu', 
                        action='store_true', 
                        default=False,
                        help='true if use gpu. default False.')
    parser.add_argument('--openvino',
                        dest='use_openvino',
                        action='store_true',
                        default=False,
                        help="if True use openvino")
    parser.add_argument('--dump_setting', 
                        dest='dump_setting', 
                        action='store_true', 
                        default=False,
                        help="dump t2i setting to file if True. default is False.")
    parser.add_argument('--vae_path',
                        dest='vae_path',
                        action='store',
                        type=str,
                        help='vae file path.')
    parser.add_argument('--lora',
                        dest='lora',
                        action='store',
                        type=str,
                        nargs='*',
                        help='<lora file path>:<scale> list')

    args = parser.parse_args()    

    main(args)
