import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from safetensors.torch import load_file
from optimum.intel import OVStableDiffusionPipeline
from openvino.runtime import Core, Model, Type
from optimum.intel import OVStableDiffusionPipeline
from openvino.runtime.passes import Manager, GraphRewrite, MatcherPass, WrapType, Matcher
from openvino.runtime import opset11 as ops
from diffusers.schedulers import *
from diffusers import StableDiffusionPipeline
import torch


class InsertLoRA(MatcherPass):
    
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
                # Castは1～8くらいまで確認.ff/net はmodelにもあった。
                # とりあえず順番とか見て適当にやってみて、まずは何か画像に影響があるか確認しよう。。
                # Openvino IRはこの順番
                # Cast, Cast_1, Cast_2, Cast_3, Cast_6, Cast_4, Cast_5, Cast_7, Cast_8,
                # lora
                # to_k, to_out_0, to_q, to_v, 
                # なので、Cast=to_k, Cast_1=to_out_0, Cast_2=to_q, Cast_3=to_v としてみよう。
                # マッチさせt何らか計算されたようで動きは変わったがshapeがどうのこうのとunetのreshapeでエラーが出てしまった。

                if root.get_friendly_name() == y["name"]:
                    # ここは基本二次元同士の足し算になる。行列の数もおなじかな？
                    consumers = root_output.get_target_inputs()
                    # 何やらopenvinoはconst値が行列反転して入っているぽい？
                    lora_weights = ops.constant(y["value"].mT, Type.f32, name=y["name"])
                    print(f"matched! node shape: {root.shape} lora_weights shape: {lora_weights.shape}")
                    add_lora = ops.add(root, lora_weights, auto_broadcast='numpy')
                    for consumer in consumers:
                        # consumerはInput型
                        consumer.replace_source_output(add_lora.output(0))

                    #print(f"lora:{lora_weights.get_output_shape(0)} add_lora:{add_lora.get_output_shape(0)}")
                    # For testing purpose
                    self.model_changed = True
                    # Use new operation for additional matching
                    self.register_new_node(add_lora)

            # Root node wasn't replaced or changed
            return False

        self.register_matcher(Matcher(param,"InsertLoRA"), callback)

def ov_add_lora_model(pipe, state_dict_list, scale_list, ov_unet_model_xml_path, ov_text_encoder_model_path):

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
                    print("appended!")
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

    # 一旦unetだけ対応してみる。
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
                    print(f"replace! {target_name} -> {lname}")
                    ll["name"] = lname

    for  ld in lora_dict_list:
        print(ld['name'])

    return lora_dict_list

def diffusers_add_lora_model(pipe, lora_state_dict):

    """
    diffusers モデルに対するLoRAの適用。
    diffusers.scripts.convert_lora_safetensors_to_diffusers.pyから拝借。
    これでscale（元のソースだとalpha）をのせてLoRAが効くことは確認した。
    """ 
    scale = 0.5
    visited = []

    # directly update weight in diffusers model
    for key in lora_state_dict:
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
        if len(lora_state_dict[pair_keys[0]].shape) == 4:
            weight_up = lora_state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = lora_state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)

            lora = scale * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            print(f"layer shape: {curr_layer.weight.data.shape} lora shape: {lora.shape}")
            curr_layer.weight.data += lora
        else:
            weight_up = lora_state_dict[pair_keys[0]].to(torch.float32)
            weight_down = lora_state_dict[pair_keys[1]].to(torch.float32)
            lora = scale * torch.mm(weight_up, weight_down)
            print(f"layer shape: {curr_layer.weight.data.shape} lora shape: {lora.shape}")
            curr_layer.weight.data += lora
        # update visited list
        for item in pair_keys:
            visited.append(item)


def main(args):

    lora_state_dict_list = []
    scale_list = []
    for file in args.lora_safetensors:
        print(f"load lora from {file}")
        lora_state_dict = load_file(file)
        lora_state_dict_list.append(lora_state_dict)
        scale_list.append(1.0)

    if args.use_diffusers:
        pipe = StableDiffusionPipeline.from_single_file(args.model)
        diffusers_add_lora_model(pipe, lora_state_dict)
    else:
        pipe = OVStableDiffusionPipeline.from_pretrained(args.model, compile=False)

        # openvino-IRはxmlのedgesにデータの連結定義がある
        # loraのファイルはstable diffusion 用でIRの形式と互換性がない。
        # IRはキーの名前が変わっているため、直接loraのキーとの連結ができない
        # IRのedgesをたどると、どのキーに相当するlayerなのかが特定できるので、
        # 特定したlayerのキーとloraのキーを引き当てて加算しようという考え方。
        unet_model_xml_path = Path(args.model) / "unet" / "openvino_model.xml"
        text_encoder_xml_path = Path(args.model) / "text_encoder" / "openvino_model.xml"

        ov_add_lora_model(pipe, lora_state_dict_list, scale_list, unet_model_xml_path, text_encoder_xml_path)

        pipe.compile()

if __name__ == '__main__':

    p = argparse.ArgumentParser()

    p.add_argument('--model',
                   type=str,
                   action='store',
                   dest='model',
                   default=r'C:\Users\webnu\source\repos\StableDiffusion\sd-ov-tools\models\AsagaoMix-v2')
    p.add_argument('--lora_safetensors',
                   type=str,
                   action='store',
                   dest='lora_safetensors',
                   nargs='*',
                   default=[r'C:\Users\webnu\source\repos\StableDiffusion\stable-diffusion-webui\models\Lora\flat2.safetensors'])
    p.add_argument('--gpu',
                   action='store_true',
                   default=False,
                   dest='gpu',
                   help='if specified use gpu else use cpu(default)') 
    p.add_argument('--diffusers',
                   action='store_true',
                   default=False,
                   dest='use_diffusers')
    
    args = p.parse_args()

    main(args)