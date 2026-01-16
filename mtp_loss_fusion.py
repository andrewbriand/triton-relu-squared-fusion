import torch
import torch.nn.functional as F

import triton
import triton.language as tl

batch = 8 * 2048
N = 50304
dim = 768
n_predict = 3

mtp_weights = torch.tensor([1.0, 0.5, 0.25], device="cuda")

dtype = torch.bfloat16

logits = torch.randn((1, batch, N), dtype=dtype, device="cuda", requires_grad=True)
target_seq = torch.randint(0, N, (batch,), dtype=torch.int64, device="cuda")

@triton.jit
def fused_mtp_loss_kernel(logits_ptr,
                          output_ptr,
                          batch,
                          dim,
                          BLOCK_BATCH: tl.constexpr,
                          BLOCK_DIM: tl.constexpr):

    batch_offset = tl.program_id(0).to(tl.int64) * BLOCK_BATCH
    batch_offsets = tl.arange(0, BLOCK_BATCH) + batch_offset
    batch_mask = batch_offsets < batch

    m = tl.full([BLOCK_BATCH], float("-inf"), tl.float32)

    sumexp = tl.zeros([BLOCK_BATCH], tl.float32)

    for dim_offset in range(0, dim, BLOCK_DIM):
        dim_offsets = tl.arange(0, BLOCK_DIM) + dim_offset
        dim_mask = dim_offsets < dim

        logit_offsets = batch_offsets[:, None] * dim + dim_offsets[None, :]
        logit_mask = batch_mask[:, None] & dim_mask[None, :]
        logits_block = tl.load(logits_ptr + logit_offsets, logit_mask)

        logits_block = 23 * tl.sigmoid((logits_block + 5) / 7.5)
        
        m_block = tl.max(logits_block, axis=-1)

        scaled_sum = tl.where(m_block > m, sumexp * tl.exp(m - m_block), sumexp)

        m = tl.maximum(m, m_block)

        block_sum = tl.sum(tl.exp(logits_block - m), axis=-1)

        sumexp = scaled_sum + block_sum

    output_offsets = tl.arange(0, BLOCK_BATCH) + batch_offset
    tl.store(output_ptr + output_offsets, m + tl.log(sumexp))

@torch.compile
def mtp_loss(logits, target_seq, n_predict, mtp_weights):
    BLOCK_BATCH = 64
    BLOCK_DIM = 64

    logits = logits.view(-1, logits.size(-1))

    batch = logits.shape[0]
    dim = logits.shape[1]

    output = torch.empty((batch,), dtype=torch.float32, device=logits.device)

    grid = (triton.cdiv(batch,BLOCK_BATCH),)

    fused_mtp_loss_kernel[grid](logits, output, batch, dim, BLOCK_BATCH, BLOCK_DIM)

    idx = F.pad(target_seq, (0, n_predict - 1)).unfold(0, n_predict, 1)
    target_logits = 23 * torch.sigmoid((logits.gather(1, idx) + 5) / 7.5)
    cross_entropy = output.unsqueeze(1) - target_logits
    for k in range(1, n_predict):
        cross_entropy[-k:, k] = 0
    loss = (cross_entropy * mtp_weights).sum()

    return loss

@torch.compile(dynamic=False, fullgraph=True)
def reference(logits, target_seq, n_predict, mtp_weights):
    logits = 23 * torch.sigmoid((logits + 5) / 7.5)
    logits_flat = logits.view(-1, logits.size(-1))
    idx = F.pad(target_seq, (0, n_predict - 1)).unfold(0, n_predict, 1)
    target_logits = logits_flat.gather(1, idx)
    cross_entropy = torch.logsumexp(logits_flat, dim=-1).unsqueeze(1) - target_logits
    for k in range(1, n_predict):
        cross_entropy[-k:, k] = 0
    loss = (cross_entropy * mtp_weights).sum()
    return loss

logits_ref = logits.clone()
logits_ref.retain_grad()
logits_kernel = logits.clone()

loss_ref = reference(logits_ref, target_seq, n_predict, mtp_weights).to(torch.bfloat16)

loss_ref.backward()

print("logits_ref.grad:", logits_ref.grad)

loss_kernel = mtp_loss(logits_kernel, target_seq, n_predict, mtp_weights).to(torch.bfloat16)

print("loss ref: ", loss_ref)
print("loss kernel: ", loss_kernel)

torch.testing.assert_close(loss_ref, loss_kernel)

print("PASS")
