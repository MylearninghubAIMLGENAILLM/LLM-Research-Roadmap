
import argparse, math, os, time
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from .utils import PositionalEncoding, set_seed

# --- Toy dataset: copy task ---
# Input: random integer sequence; Output: same sequence.
class CopyDataset(Dataset):
    def __init__(self, n_samples=20000, vocab_size=50, min_len=5, max_len=30, pad_id=0):
        self.samples = []
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        for _ in range(n_samples):
            L = torch.randint(min_len, max_len+1, (1,)).item()
            seq = torch.randint(2, vocab_size, (L,))  # reserve 0 for PAD, 1 for BOS
            self.samples.append(seq)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        bos = torch.tensor([1], dtype=torch.long)
        src = torch.cat([bos, seq], dim=0)
        tgt_in = torch.cat([bos, seq], dim=0)    # teacher forcing input
        tgt_out = torch.cat([seq, torch.tensor([2])], dim=0)  # 2 as EOS
        return src, tgt_in, tgt_out

def collate_fn(batch, pad_id=0):
    srcs, tgts_in, tgts_out = zip(*batch)
    def pad(seqs):
        maxL = max(x.size(0) for x in seqs)
        out = torch.full((len(seqs), maxL), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :s.size(0)] = s
        return out
    return pad(srcs), pad(tgts_in), pad(tgts_out)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size=50, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.src_tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt_in):
        # src, tgt_in: (B, T)
        src_key_padding_mask = (src == self.pad_id)
        tgt_key_padding_mask = (tgt_in == self.pad_id)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(tgt_in.device)

        src_emb = self.pos(self.src_tok(src))
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        tgt_emb = self.pos(self.tgt_tok(tgt_in))
        out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=src_key_padding_mask)
        logits = self.proj(out)
        return logits

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    ds = CopyDataset(n_samples=args.n_samples, vocab_size=args.vocab_size, min_len=5, max_len=args.max_len, pad_id=args.pad_id)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, args.pad_id))

    model = TransformerSeq2Seq(vocab_size=args.vocab_size, d_model=args.d_model, nhead=args.nhead,
                               num_layers=args.num_layers, dim_feedforward=args.ffn, dropout=args.dropout, pad_id=args.pad_id).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=args.pad_id)

    global_step=0
    model.train()
    for epoch in range(1000):  # loop controlled by max_steps
        for src, tgt_in, tgt_out in dl:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            logits = model(src, tgt_in)  # (B, T, V)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            global_step += 1
            if global_step % args.log_steps == 0:
                print(f"step {global_step} | loss {loss.item():.4f}")
            if global_step >= args.max_steps:
                torch.save(model.state_dict(), os.path.join(args.output_dir, "toy_transformer.pt"))
                print("Saved model to", os.path.join(args.output_dir, "toy_transformer.pt"))
                return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--vocab_size", type=int, default=100)
    parser.add_argument("--max_len", type=int, default=40)
    parser.add_argument("--pad_id", type=int, default=0)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--ffn", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--output_dir", type=str, default="outputs/toy")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
