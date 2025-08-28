# Transformer + LLM (Colab Starter)

This starter is designed for Google Colab (free tier) and covers:
- **Paper core**: a compact Transformer (encoder–decoder) training on a toy task.
- **Train a tiny causal LM from scratch** on a small text dataset.
- **Fine-tune a pretrained LM with LoRA/QLoRA** (distilgpt2 to keep VRAM low).
- **Basic MLOps**: config files, experiment tracking (Weights & Biases optional), simple eval.

> Note: Free Colab often gives a 16GB T4 GPU; keep batch sizes small. Do not expect to train large LLMs from scratch.

---

## Quick Start (Colab)

1. **New Colab notebook** → `Runtime > Change runtime type > GPU`.
2. In the first cell, run:
   ```bash
   !pip -q install torch==2.3.1 torchvision --index-url https://download.pytorch.org/whl/cu121
   !pip -q install transformers datasets accelerate peft bitsandbytes wandb sentencepiece tokenizers==0.15.2
   ```

3. **Upload** this zip or mount Drive, then unzip:
   ```bash
   from google.colab import drive; drive.mount('/content/drive', force_remount=True)
   !unzip -o "/content/drive/MyDrive/transformer_colab_starter.zip" -d /content
   %cd /content/transformer_colab_starter
   ```
   Or upload directly via the Colab file pane and:
   ```bash
   !unzip -o transformer_colab_starter.zip -d /content
   %cd /content/transformer_colab_starter
   ```

4. **(Optional) W&B login** for experiment tracking:
   ```python
   import wandb, os
   os.environ["WANDB_PROJECT"]="transformer-colab"
   wandb.login()  # paste your API key
   ```

5. **Run a toy Transformer (paper-style) training** (copy task):
   ```bash
   python -m src.train_transformer_scratch --max_steps 800 --d_model 128 --nhead 4 --num_layers 2
   ```

6. **Train a tiny causal LM from scratch** on WikiText-2-raw (few steps):
   ```bash
   python -m src.train_causal_lm --max_steps 1000 --eval_steps 200 --save_steps 500
   ```

7. **Fine-tune with LoRA (QLoRA if possible)** on a tiny dialogue dataset:
   ```bash
   python -m src.finetune_lora --base_model distilgpt2 --bits 4 --max_steps 600 --eval_steps 200
   ```

8. **Evaluate perplexity**:
   ```bash
   python -m src.evaluate --model_path outputs/causal_lm --split validation
   ```

## Project Layout

```
transformer_colab_starter/
├─ README.md
├─ requirements.txt
├─ configs/
│  └─ default.yaml
└─ src/
   ├─ train_transformer_scratch.py
   ├─ train_causal_lm.py
   ├─ finetune_lora.py
   ├─ evaluate.py
   └─ utils.py
```

## Tips

- Start with **`train_transformer_scratch.py`** to learn the architecture (positional encodings, attention weights).
- Use **`train_causal_lm.py`** to train a tiny GPT-like model.
- Use **`finetune_lora.py`** to adapt a pretrained LM quickly on small VRAM.
- Keep steps tiny (100–200) to verify the loop, then scale up gradually.
- If you hit CUDA OOM, lower `--batch_size`, `--seq_len`, or `--d_model` / `--n_layer`.

## Debugging

- Run with `CUDA_LAUNCH_BLOCKING=1` for clearer stack traces:
  ```bash
  CUDA_LAUNCH_BLOCKING=1 python -m src.train_causal_lm --max_steps 50
  ```
- Use `--gradient_checkpointing` to reduce memory when fine-tuning.
- Inspect gradients / norms (see logs printed every `--log_steps`).

## References
- Vaswani et al., 2017: *Attention Is All You Need* (arXiv:1706.03762)
- Hugging Face `transformers`, `datasets`, `peft`, `accelerate`
