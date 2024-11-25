# Diffusers Apple

**uvのインストール**

本記事では [uv](https://docs.astral.sh/uv/) を使用して環境構築を行います．未インストールの場合は次のコマンドでインストールしてください

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**依存関係のインストール**

```shell
uv sync
```

**huggingface CLIのセットアップ**

[HuggingFace](https://huggingface.co/) のアカウントを作成して https://huggingface.co/settings/tokens からアクセストークンを発行します．その後，トークンを使って`huggingface-cli`のログインを済ませておきます (`huggingface-cli`は`uv sync`でインストール済み)

https://huggingface.co/docs/huggingface_hub/guides/cli

```shell
huggingface-cli login

To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Enter your token (input will not be visible):
```

入力したトークンは `~/.cache/huggingface/token`に保存されています
