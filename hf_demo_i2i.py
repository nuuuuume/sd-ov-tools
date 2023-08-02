import requests
import torch
from PIL import Image
from io import BytesIO
from optimum.intel import OVStableDiffusionImg2ImgPipeline

model_id = "runwayml/stable-diffusion-v1-5"
#model_id = "models\\SakuraMix-v2.1-ov"
pipeline = OVStableDiffusionImg2ImgPipeline.from_pretrained(model_id, compile=False, export=True)
pipeline.reshape(batch_size=1,
                 height=512,
                 width=768,
                 num_images_per_prompt=1)
#pipeline.to('GPU')
pipeline.compile()

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))
prompt = "A fantasy landscape, trending on artstation"
image = pipeline(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
image.save("fantasy_landscape.png")