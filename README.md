# MLKit Apple Exp

## セットアップ

### Installation

```shell
uv sync
```

### huggingface CLI のログイン

https://huggingface.co/settings/tokens にアクセスし，トークンを発行する

-   参考 https://huggingface.co/docs/huggingface_hub/guides/cli

```shell
huggingface-cli login
```

`~/.cache/huggingface/token`にトークンが保存される

### Model

`~/.cache/huggingface/hub`にモデルが保存される

### scheduler

```python
pipeline = DiffusionPipeline.from_pretrained(model_id)
pipeline.scheduler.compatibles
```

### Performance

https://huggingface.co/docs/diffusers/ja/stable_diffusion
