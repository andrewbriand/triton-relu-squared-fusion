import torch
import torch.nn.functional as F

batch = 24 * 2048
N = 50304
dim = 768
n_predict = 3

mtp_weights = torch.tensor([1.0, 0.5, 0.25], device="cuda")

dtype = torch.bfloat16

logits = torch.randn((1, batch, N), dtype=dtype, device="cuda", requires_grad=True)
target_seq = torch.randint(0, N, (batch,), dtype=torch.int64, device="cuda")

@torch.compile(dynamic=False, fullgraph=True)
def reference(logits):
    logits = 23 * torch.sigmoid((logits + 5) / 7.5)
    logits_flat = logits.view(-1, logits.size(-1))
    idx = F.pad(target_seq, (0, n_predict - 1)).unfold(0, n_predict, 1)
    target_logits = logits_flat.gather(1, idx)
    cross_entropy = torch.logsumexp(logits_flat, dim=-1).unsqueeze(1) - target_logits
    for k in range(1, n_predict):
        cross_entropy[-k:, k] = 0
    loss = (cross_entropy * mtp_weights).sum()
    return loss

loss = reference(logits)

print("loss ref: ", loss)
