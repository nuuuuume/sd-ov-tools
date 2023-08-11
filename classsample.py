from openvino.runtime import Model

class classA:
    def __init__(self,
                 arg1: Model):
        # __init__ はコンストラクタとほぼ同義と捉えて問題ない。
        print(self.text1)

    def __call__(self,
                 arg1: str):
        # __call__ はインスタンスを関数のように呼び出すときに呼ばれる
        # ins = classA()
        # ins() 
        # ↑これで__call__ が呼ばれる
        self.text1 = 'fromCall'

    @classmethod
    def classMethod(cls):
        # このclsはインスタンスではなくてクラスそのものを指しているようなので、
        # 以下のコードの場合は __call__ が呼ばれるわけではなくて、新しいインスタンスが生成される（__init__が呼ばれる）
        return cls(arg1="C:\\Users\\webnu\\source\\repos\\StableDiffusion\\sd-ov-tools\\models\\kl-f8-anime2_vae_ov\\vae_encoder\\openvino_model.xml") 

def main():
    ins = classA.classMethod()
    print(ins.text1)

    ins('A')
    print(ins.text1)

if __name__ == '__main__':
    main()
