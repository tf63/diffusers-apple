import src.patches.scheduling_euler_ancestral_discrete  # noqa

import os

import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import (
    StableDiffusionXLImg2ImgPipeline,
)
from diffusers.utils.loading_utils import load_image
from diffusers.utils.pil_utils import make_image_grid

from src.utils import ExperimentalContext, options


def inference(pipeline_img2img, context: ExperimentalContext, prompt: str, guidance_scale=0.0, num_inference_steps=2):
    # ソース画像の読み込み
    init_image = load_image(
        'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png'
    )
    init_image = init_image.resize((512, 512))

    negative_prompt = 'low quality, bad quality'

    # 推論
    image = pipeline_img2img(
        prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=0.5,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=context.generator,
    ).images[0]

    # 画像の保存
    image_compare = make_image_grid([init_image, image], rows=1, cols=2)
    context.save_image(image, prompt.replace(' ', '_'), f'n{num_inference_steps}_s{guidance_scale}')
    context.save_image(image_compare, prompt.replace(' ', '_'), f'n{num_inference_steps}_s{guidance_scale}_comp')


@options
def main(seed, device):
    prompt = 'cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k'

    # text2imgモデルの読み込み
    pipeline_text2img = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/sdxl-turbo', torch_dtype=torch.float16, variant='fp16'
    ).to(device)

    # img2imgモデルの読み込み
    pipeline_img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(pipeline_text2img).to(device)

    context = ExperimentalContext(seed=seed, device=device, root_dir=os.path.join('out', 'sdxl_turbo_img2img_cat'))
    inference(pipeline_img2img, context=context, prompt=prompt, num_inference_steps=2)


if __name__ == '__main__':
    main()
