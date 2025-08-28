
import argparse, os, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
try:
    import bitsandbytes as bnb  # noqa
except Exception:
    bnb = None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="distilgpt2")
    ap.add_argument("--bits", type=int, default=4, help="Set 0 to disable quantization.")
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--output_dir", type=str, default="outputs/lora_ft")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Small dialogue dataset for demonstration
    ds = load_dataset("daily_dialog")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def format_dialog(example):
        # join utterances into one text block
        text = "\n".join(example["dialog"][:4])  # keep it short
        return {"text": text}

    ds_small = ds["train"].select(range(500)).map(format_dialog)
    ds_val = ds["validation"].select(range(200)).map(format_dialog)

    def tok_fn(ex):
        out = tok(ex["text"], truncation=True, max_length=args.seq_len)
        return out

    train_tok = ds_small.map(tok_fn, batched=True, remove_columns=ds_small.column_names)
    val_tok = ds_val.map(tok_fn, batched=True, remove_columns=ds_val.column_names)

    quantization_config = None
    load_kwargs = {}
    if args.bits in (4, 8) and bnb is not None:
        load_kwargs = {
            "load_in_4bit": args.bits == 4,
            "load_in_8bit": args.bits == 8,
            "device_map": "auto"
        }

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)

    if args.bits in (4, 8) and bnb is not None:
        model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["c_attn","q_proj","v_proj","k_proj","o_proj"]  # covers GPT2 & many others; ignore invalids
    )
    model = get_peft_model(model, peft_cfg)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=2,
        max_steps=args.max_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=torch.cuda.is_available(),
        report_to=["none"],  # set to ["wandb"] after wandb.login()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Saved to", args.output_dir)

if __name__ == "__main__":
    main()
