
import argparse, os, math
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoConfig, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default=None, help="Use None to train from scratch.")
    ap.add_argument("--vocab", type=str, default="gpt2")  # tokenizer family
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--output_dir", type=str, default="outputs/causal_lm")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset: wikitext-2-raw
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    tok = AutoTokenizer.from_pretrained(args.vocab)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tok_fn(ex):
        return tok(ex["text"])

    tokenized = ds.map(tok_fn, batched=True, remove_columns=["text"])

    # Group texts
    def group_texts(examples):
        concatenated = sum(examples["input_ids"], [])
        total_length = (len(concatenated) // args.seq_len) * args.seq_len
        input_ids = [concatenated[i:i+args.seq_len] for i in range(0, total_length, args.seq_len)]
        return {"input_ids": input_ids}
    lm_ds = tokenized.map(group_texts, batched=True)

    cfg = None
    if args.model_name is None:
        cfg = AutoConfig.from_pretrained("gpt2")
        cfg.n_layer = 4
        cfg.n_head = 4
        cfg.n_embd = 256
        cfg.n_positions = args.seq_len
        cfg.vocab_size = tok.vocab_size
        model = AutoModelForCausalLM.from_config(cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)

    model.to(device)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        max_steps=args.max_steps,
        bf16=torch.cuda.is_available(),
        report_to=["none"],  # set to ["wandb"] after wandb.login()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_ds["train"],
        eval_dataset=lm_ds["validation"],
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Saved to", args.output_dir)

if __name__ == "__main__":
    main()
