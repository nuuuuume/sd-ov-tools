import argparse
import openvino
from safetensors.torch import load_file
from optimum.intel import OVStableDiffusionPipeline
from openvino.runtime import Core, Model, Type
from optimum.intel import OVStableDiffusionPipeline
from openvino.runtime.passes import Manager, GraphRewrite, MatcherPass, WrapType, Matcher
from openvino.runtime import opset11 as ops
from diffusers.schedulers import *
import torch


class InsertLoRA(MatcherPass):
    
    def __init__(self,lora_dict_list):
        MatcherPass.__init__(self)
        self.model_changed = False

        param = WrapType("opset11.Convert")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            root_output = matcher.get_match_value()

            print(root.get_friendly_name())
            for c in root_output.get_target_inputs():
                print(c.get_partial_shape())

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

                fname_replaced = root.get_friendly_name().replace('.', '_').replace('/', '_').replace('Cast_3', 'to_v').replace('Cast_2', 'to_q').replace('Cast_1', 'to_out_0').replace('Cast', 'to_k')[1:]
                #print(f"y      :{y['name']}")
 
                #if root.get_friendly_name().replace('.','_').replace('_weight','') == y["name"]:
                if fname_replaced == y["name"]:
                    print(f"replace:{fname_replaced}")
                    # ここは基本二次元同士の足し算になる。行列の数もおなじかな？
                    #print(f"matched!: {y['name']}")
                    consumers = root_output.get_target_inputs()
                    # lora_weightsとadd_loraはともに2次元のShapeができている。行列の数も一致している。
                    lora_weights = ops.constant(y["value"], Type.i64, name=y["name"])
                    add_lora = ops.add(root, lora_weights, auto_broadcast='numpy')
                    for consumer in consumers:
                        # consumerはInput型
                        # replaceすることで1次元の空行列が2次元の行列になる。
                        # エラーになるのはこれが原因かね？
                       # print(f"replace before:{consumer.get_shape()}")
                       # print(f"replace before:{consumer.get_tensor().size}")
                        consumer.replace_source_output(add_lora.output(0))
                       # print(f"replace after:{consumer.get_shape()}")
                       # print(f"replace after:{consumer.get_tensor().size}")

                    #print(f"lora:{lora_weights.get_output_shape(0)} add_lora:{add_lora.get_output_shape(0)}")
                    # For testing purpose
                    self.model_changed = True
                    # Use new operation for additional matching
                    self.register_new_node(add_lora)

            # Root node wasn't replaced or changed
            return False

        self.register_matcher(Matcher(param,"InsertLoRA"), callback)

def add_lora_model(pipe, state_dict, scale_list):

    """
    add lora weights
    
    parameters:
        pipe: 
            openvino stablediffusion pipeline
        state_dict:
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
    for iter in range(len(state_dict)):
        visited = []
        for key in state_dict[iter]:
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
            if len(state_dict[iter][pair_keys[0]].shape) == 4:
                # len(shape) == 4 のは proj_in proj_outくらいらしい。
                weight_up = state_dict[iter][pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
                weight_down = state_dict[iter][pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
                # ここでloraに設定した係数（scale値、、1とか0.5とかflat2だと-1とか）がのる？
                lora_weights = scale_list[iter] * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)

                print(f"{key}:{lora_weights.shape}")
                lora_dict.update(value=lora_weights)
            else:
                # その他、は2次元。基本はこちらの値を相手にする形になるのかねえ？
                weight_up = state_dict[iter][pair_keys[0]].to(torch.float32)
                weight_down = state_dict[iter][pair_keys[1]].to(torch.float32)
                # ここでloraに設定した係数（scale値、、1とか0.5とかflat2だと-1とか）がのる？
                lora_weights = scale_list[iter] * torch.mm(weight_up, weight_down)
                lora_dict.update(value=lora_weights)

            # lora_weightsは2次元だったり4次元だったり。

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

    # 上で作った加工済みのlora情報をregister_pass -> run_passesと流し、その中でConvertの値にlora_の値をAddする、
    # という流れになっているポイ。
    manager.register_pass(InsertLoRA(lora_dict_list))
    if (True in [('type', 'text_encoder') in l.items() for l in lora_dict_list]):
        print("--- text encoder run_passes")
        manager.run_passes(pipe.text_encoder.model)
        print(f"--- run_passes result: {rp_ret}")

    print("--- unet run_passes")
    rp_ret = manager.run_passes(pipe.unet.model)
    print(f"--- run_passes result: {rp_ret}")


def main(args):

    lora = load_file(args.lora_safetensors)
    #pipe = OVStableDiffusionPipeline.from_pretrained(args.openvino_model, compile=False)

    dummy_input =torch.randn(2, 3, 64, 64, requires_grad=True)
    torch.onnx.export(lora, 
                      dummy_input,
                      "lora.onnx",
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input_ids'],
                      output_names=['output_ids']
                      ) 

if __name__ == '__main__':

    p = argparse.ArgumentParser()

    p.add_argument('--openvino_model',
                   type=str,
                   action='store',
                   dest='openvino_model',
                   default=r'C:\Users\webnu\source\repos\StableDiffusion\sd-ov-tools\models\AsagaoMix-v2')
    p.add_argument('--lora_safetensors',
                   type=str,
                   action='store',
                   dest='lora_safetensors',
                   default=r'C:\Users\webnu\source\repos\StableDiffusion\stable-diffusion-webui\models\Lora\flat2.safetensors')
    p.add_argument('--gpu',
                   action='store_true',
                   default=False,
                   dest='gpu',
                   help='if specified use gpu else use cpu(default)') 

    args = p.parse_args()

    main(args)