
import argparse, math, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--seq_len", type=int, default=128)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    ds = load_dataset("wikitext", "wikitext-2-raw-v1")[args.split]

    def batch(it, n=8):
        buf = []
        for x in it:
            buf.append(x)
            if len(buf) == n:
                yield buf
                buf = []
        if buf:
            yield buf

    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for chunk in batch(ds, 8):
            enc = tok([x["text"] for x in chunk], return_tensors="pt", padding=True, truncation=True, max_length=args.seq_len)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = out.loss
            n_tokens = attention_mask.sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    ppl = math.exp(total_loss / max(total_tokens, 1))
    print(f"Perplexity: {ppl:.2f}")

if __name__ == "__main__":
    main()
