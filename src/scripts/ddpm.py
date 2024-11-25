import os

import torch
from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline

from src.utils import ExperimentalContext, options


def inference(pipeline, context: ExperimentalContext, batch_size, num_inference_steps=1000):
    # 推論
    images = pipeline(
        batch_size=batch_size,
        generator=context.generator,
        num_inference_steps=num_inference_steps,
    ).images

    # 画像の保存
    for i, image in enumerate(images):
        context.save_image(image, 'uncond', f'i{i}_n{num_inference_steps}')


@options
def main(seed, device):
    context = ExperimentalContext(seed=seed, device=device, root_dir=os.path.join('out', 'ddpm_cat'))
    batch_size = 1

    # モデルの読み込み
    pipeline = DDPMPipeline.from_pretrained('google/ddpm-cat-256', torch_dtype=torch.float16).to(context.device)
    inference(pipeline=pipeline, context=context, batch_size=batch_size, num_inference_steps=1000)


if __name__ == '__main__':
    main()
