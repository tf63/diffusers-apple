# https://huggingface.co/kandinsky-community/kandinsky-2-1-inpaint
import os

import torch
from diffusers.pipelines.auto_pipeline import AutoPipelineForInpainting
from diffusers.utils.loading_utils import load_image
from diffusers.utils.pil_utils import make_image_grid

from src.utils import ExperimentalContext, options


def inference(pipeline_inpaint, context: ExperimentalContext, prompt: str, guidance_scale=0.0, num_inference_steps=2):
    negative_prompt = 'low quality, bad quality'

    # ソース画像･マスク画像の読み込み
    # https://huggingface.co/docs/diffusers/ja/tutorials/autopipeline
    img_url = 'https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png'
    mask_url = 'https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png'

    init_image = load_image(img_url).convert('RGB')
    mask_image = load_image(mask_url).convert('RGB')

    # 推論
    image = pipeline_inpaint(
        prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        # strength=0.8,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=context.generator,
    ).images[0]

    # 画像の保存
    image_compare = make_image_grid([init_image, mask_image, image], rows=1, cols=3)
    context.save_image(image, prompt.replace(' ', '_'), f'n{num_inference_steps}_s{guidance_scale}')
    context.save_image(image_compare, prompt.replace(' ', '_'), f'n{num_inference_steps}_s{guidance_scale}_comp')


@options
def main(seed, device):
    prompt = 'a bench'
    # prompt = 'cat wizard, sitting on a bench'

    pipeline_inpaint = AutoPipelineForInpainting.from_pretrained(
        'kandinsky-community/kandinsky-2-1-inpaint', torch_dtype=torch.float16
    ).to(device)

    context = ExperimentalContext(seed=seed, device=device, root_dir=os.path.join('out', 'kandinsky_inpaint_sample'))
    inference(pipeline_inpaint=pipeline_inpaint, context=context, prompt=prompt, num_inference_steps=25)


if __name__ == '__main__':
    main()
