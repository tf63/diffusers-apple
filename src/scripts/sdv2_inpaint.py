import src.patches.scheduling_euler_ancestral_discrete  # noqa

import os

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from diffusers.utils.loading_utils import load_image
from diffusers.utils.pil_utils import make_image_grid

from src.utils import ExperimentalContext, options


def inference(pipeline_inpaint, context: ExperimentalContext, prompt: str, guidance_scale=0.0, num_inference_steps=2):
    # ソース画像･マスク画像の読み込み
    # https://huggingface.co/docs/diffusers/ja/tutorials/autopipeline
    img_url = 'https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png'
    mask_url = 'https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png'

    init_image = load_image(img_url).convert('RGB')
    mask_image = load_image(mask_url).convert('RGB')

    negative_prompt = 'low quality, bad quality'

    # 推論
    image = pipeline_inpaint(
        prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=context.generator,
    ).images[0]

    # 画像の保存
    filename = f'n{num_inference_steps}_s{guidance_scale}'
    image_compare = make_image_grid([init_image, mask_image, image], rows=1, cols=3)
    context.save_image(image, prompt.replace(' ', '_'), filename)
    context.save_image(image_compare, prompt.replace(' ', '_'), f'{filename}_comp')


@options
def main(seed, device):
    prompt = 'a bench'
    # prompt = 'a cat, sitting on a bench'

    # img2imgモデルの読み込み
    # https://huggingface.co/stabilityai/stable-diffusion-2-inpainting
    pipeline_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-inpainting', torch_dtype=torch.float16, variant='fp16'
    ).to(device)

    context = ExperimentalContext(seed=seed, device=device, root_dir=os.path.join('out', 'sdv2_inpaint_sample'))
    inference(
        pipeline_inpaint=pipeline_inpaint, context=context, prompt=prompt, num_inference_steps=50, guidance_scale=3.0
    )


if __name__ == '__main__':
    main()
