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
def fused_softcapped_entropy_fwd_kernel(
    logits_ptr, losses_ptr, lse_ptr, targets_ptr, mtp_weights_ptr,
    stride_logits_n, stride_logits_v,
    n_rows, n_cols, n_predict,
    A, B, C,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0).to(tl.int64)
    logits_row_ptr = logits_ptr + row_idx * stride_logits_n
    
    max_val = -float('inf')
    sum_exp = 0.0
    
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        val = tl.load(logits_row_ptr + cols, mask=mask, other=-float('inf')).to(tl.float32)
        z = A * tl.sigmoid((val + B) / C)
        z = tl.where(mask, z, -float('inf'))
        curr_max = tl.max(z, axis=0)
        new_max = tl.maximum(max_val, curr_max)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.sum(tl.exp(z - new_max), axis=0)
        max_val = new_max
    
    lse = max_val + tl.log(sum_exp)
    tl.store(lse_ptr + row_idx, lse)
    
    total_loss = 0.0
    for k in range(n_predict):
        target_idx = row_idx + k
        if target_idx < n_rows:
            weight = tl.load(mtp_weights_ptr + k)
            if weight > 0:
                target = tl.load(targets_ptr + target_idx).to(tl.int32)
                if target >= 0 and target < n_cols:
                    val_target = tl.load(logits_row_ptr + target).to(tl.float32)
                    z_target = A * tl.sigmoid((val_target + B) / C)
                    total_loss += weight * (lse - z_target)
    
    tl.store(losses_ptr + row_idx, total_loss)

@triton.jit
def fused_softcapped_entropy_bwd_kernel(
    grad_input_ptr, grad_output_ptr, lse_ptr, logits_ptr, targets_ptr, mtp_weights_ptr,
    stride_logits_n, stride_logits_v, stride_grad_n, stride_grad_v,
    n_rows, n_cols, n_predict,
    A, B, C,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0).to(tl.int64)

    logits_row_ptr = logits_ptr + row_idx * stride_logits_n
    grad_row_ptr = grad_input_ptr + row_idx * stride_grad_n
    
    lse = tl.load(lse_ptr + row_idx)
    grad_loss = tl.load(grad_output_ptr + row_idx)
    
    S_w = 0.0
    for k in range(n_predict):
        if row_idx + k < n_rows:
            S_w += tl.load(mtp_weights_ptr + k)
            
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        val = tl.load(logits_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        u = (val + B) / C
        sigmoid_u = tl.sigmoid(u)
        z = A * sigmoid_u
        p = tl.exp(z - lse)
        
        term1 = S_w * p
        term2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for k in range(n_predict):
            if row_idx + k < n_rows:
                target = tl.load(targets_ptr + row_idx + k).to(tl.int32)
                weight = tl.load(mtp_weights_ptr + k)
                term2 += tl.where(cols == target, weight, 0.0)
        
        grad_z = grad_loss * (term1 - term2)
        dz_dx = (1.0 / C) * z * (1.0 - sigmoid_u)
        grad_x = grad_z * dz_dx
        tl.store(grad_row_ptr + cols, grad_x.to(tl.bfloat16), mask=mask)

class FusedSoftcappedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, mtp_weights, A=23.0, B=5.0, C=7.5):
        n_rows, n_cols = logits.shape
        if mtp_weights is None:
             mtp_weights = torch.tensor([1.0], device=logits.device, dtype=torch.float32)
        n_predict = mtp_weights.shape[0]

        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        
        logits = logits.contiguous()
        targets = targets.contiguous()
        mtp_weights = mtp_weights.contiguous()

        grid = (n_rows,)
        fused_softcapped_entropy_fwd_kernel[grid](
            logits, losses, lse, targets, mtp_weights,
            logits.stride(0), logits.stride(1),
            n_rows, n_cols, n_predict,
            A, B, C,
            BLOCK_SIZE=1024,
            num_warps=2
        )
        
        ctx.save_for_backward(logits, targets, mtp_weights, lse)
        ctx.params = (A, B, C)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, mtp_weights, lse = ctx.saved_tensors
        A, B, C = ctx.params
        n_rows, n_cols = logits.shape
        n_predict = mtp_weights.shape[0]
        
        grad_input = torch.empty((n_rows, n_cols), dtype=torch.bfloat16, device=logits.device)
        grad_output = grad_output.contiguous()
        
        grid = (n_rows,)
        fused_softcapped_entropy_bwd_kernel[grid](
            grad_input, grad_output, lse, logits, targets, mtp_weights,
            logits.stride(0), logits.stride(1), grad_input.stride(0), grad_input.stride(1),
            n_rows, n_cols, n_predict,
            A, B, C,
            BLOCK_SIZE=1024,
            num_warps=2
        )
        return grad_input, None, None, None, None, None

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
logits_kernel.retain_grad()

loss_ref = reference(logits_ref, target_seq, n_predict, mtp_weights).to(torch.bfloat16)

loss_ref.backward()

loss_kernel = FusedSoftcappedCrossEntropy.apply(logits_kernel.view(-1, logits_kernel.shape[-1]), target_seq, mtp_weights).sum().to(torch.bfloat16)

loss_kernel.backward()

print("loss ref: ", loss_ref)
print("loss kernel: ", loss_kernel)

print("logits_ref.grad:", logits_ref.grad)
print("logits_kernel.grad:", logits_kernel.grad)

torch.testing.assert_close(loss_ref, loss_kernel)
torch.testing.assert_close(logits_ref.grad, logits_kernel.grad)

print("PASS")
