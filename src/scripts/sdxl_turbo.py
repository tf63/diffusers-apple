import src.patches.scheduling_euler_ancestral_discrete  # noqa

import os
from typing import List

import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

from src.utils import ExperimentalContext, options


def inference(pipeline, context: ExperimentalContext, prompts: List[str], guidance_scale=0.0, num_inference_steps=1):
    negative_prompt = 'low quality, bad quality'

    # 推論
    images = pipeline(
        prompts,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=context.generator,
    ).images

    # 画像の保存
    for i, image in enumerate(images):
        context.save_image(image, prompts[i].replace(' ', '_'), f'i{i}_n{num_inference_steps}_s{guidance_scale}')


def inference_loop(pipeline, context: ExperimentalContext, guidance_scale=0.0, num_inference_steps=1):
    batch_size = 1

    while True:
        prompt = input('Enter a prompt (type "q" to quit): ')
        if prompt == 'q':
            break

        prompts = [prompt for i in range(batch_size)]
        negative_prompt = 'low quality, bad quality'

        # 推論
        images = pipeline(
            prompts,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=context.generator,
        ).images

        # 画像の保存
        for i, image in enumerate(images):
            context.save_image(image, prompts[i].replace(' ', '_'), f'i{i}_n{num_inference_steps}_s{guidance_scale}')


@options
def main(seed, device):
    prompts = [
        'a cat, fat, with brown fur, with short legs',
        'a cat, fat, with white fur, with short legs',
    ]

    # モデルの読み込み
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/sdxl-turbo', torch_dtype=torch.float16, variant='fp16'
    ).to(device)

    context = ExperimentalContext(seed=seed, device=device, root_dir=os.path.join('out', 'sdxl_turbo'))
    inference(pipeline=pipeline, context=context, prompts=prompts, num_inference_steps=1, guidance_scale=0.0)
    context = ExperimentalContext(seed=seed, device=device, root_dir=os.path.join('out', 'sdxl_turbo'))
    inference(pipeline=pipeline, context=context, prompts=prompts, num_inference_steps=2, guidance_scale=0.0)
    context = ExperimentalContext(seed=seed, device=device, root_dir=os.path.join('out', 'sdxl_turbo'))
    inference(pipeline=pipeline, context=context, prompts=prompts, num_inference_steps=4, guidance_scale=0.0)
    context = ExperimentalContext(seed=seed, device=device, root_dir=os.path.join('out', 'sdxl_turbo'))
    inference(pipeline=pipeline, context=context, prompts=prompts, num_inference_steps=8, guidance_scale=0.0)
    context = ExperimentalContext(seed=seed, device=device, root_dir=os.path.join('out', 'sdxl_turbo'))
    inference(pipeline=pipeline, context=context, prompts=prompts, num_inference_steps=16, guidance_scale=0.0)

    # inference_loop(pipeline=pipeline, context=context, num_inference_steps=4, guidance_scale=0.0)


if __name__ == '__main__':
    main()
