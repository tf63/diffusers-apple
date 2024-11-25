import src.patches.scheduling_dpmsolver_multistep  # noqa

import os
from typing import List

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler

from src.utils import ExperimentalContext, options


def inference(pipeline, context: ExperimentalContext, prompts: List[str], guidance_scale=0.0, num_inference_steps=50):
    negative_prompts = ['low quality, bad quality' for _ in range(len(prompts))]

    # 推論
    images = pipeline(
        prompts,
        negative_prompt=negative_prompts,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=context.generator,
    ).images

    # 画像の保存
    for i, image in enumerate(images):
        context.save_image(image, prompts[i].replace(' ', '_'), f'n{num_inference_steps}_s{guidance_scale}')


@options
def main(seed, device):
    prompts = [
        'a cat, fat, with brown fur, with short legs',
        'a cat, fat, with white fur, with short legs',
    ]

    # モデルの読み込み
    pipeline = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        torch_dtype=torch.float16,
    ).to(device)

    # スケジューラの読み込み
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    context = ExperimentalContext(seed=seed, device=device, root_dir=os.path.join('out', 'sdv1_5_dpmsolver'))
    inference(pipeline=pipeline, context=context, prompts=prompts, num_inference_steps=25, guidance_scale=0.0)
    context = ExperimentalContext(seed=seed, device=device, root_dir=os.path.join('out', 'sdv1_5_dpmsolver'))
    inference(pipeline=pipeline, context=context, prompts=prompts, num_inference_steps=25, guidance_scale=1.0)
    context = ExperimentalContext(seed=seed, device=device, root_dir=os.path.join('out', 'sdv1_5_dpmsolver'))
    inference(pipeline=pipeline, context=context, prompts=prompts, num_inference_steps=25, guidance_scale=2.0)
    context = ExperimentalContext(seed=seed, device=device, root_dir=os.path.join('out', 'sdv1_5_dpmsolver'))
    inference(pipeline=pipeline, context=context, prompts=prompts, num_inference_steps=25, guidance_scale=4.0)
    context = ExperimentalContext(seed=seed, device=device, root_dir=os.path.join('out', 'sdv1_5_dpmsolver'))
    inference(pipeline=pipeline, context=context, prompts=prompts, num_inference_steps=25, guidance_scale=8.0)


if __name__ == '__main__':
    main()
