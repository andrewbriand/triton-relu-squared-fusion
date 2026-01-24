import torch
import torch.nn.functional as F
from torch import Tensor

import triton
import triton.language as tl

import time

batch = 8 * 2048
N = 50304
dim = 768
n_predict = 3

mtp_weights = torch.tensor([1.0, 0.5, 0.25], device="cuda")

dtype = torch.bfloat16

target_seq = torch.randint(0, N, (batch,), dtype=torch.int64, device="cuda")

lm_head_weight = torch.randn((dim, N), dtype=torch.bfloat16, device="cuda", requires_grad=True)
x = torch.randn((1, batch, dim), dtype=dtype, device="cuda", requires_grad=True)
x_s = 100/448
w_s = 1.6/448
grad_s = 0.75/448

A = 23.0
B = 5.0
C = 7.5

USE_SOFTCAPPING = True

@torch.library.custom_op("nanogpt::mm_t", mutates_args=())
def mm_t_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    """Computes y = x @ w with F8 weights stored as (in_features, out_features)."""
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        assert x.shape[1] == w.shape[0]  # x: (batch, in), w: (in, out)

        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)

        # _scaled_mm requires column-major B. w_f8 is row-major (in, out).
        # .T.contiguous().T creates a column-major view without changing logical shape.
        w_f8_col_major = w_f8.T.contiguous().T

        out = torch._scaled_mm(
            x_f8,
            w_f8_col_major,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_t_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[0]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_t_backward", mutates_args=())
def mm_t_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        
        x_scale = grad.new_tensor(x_s, dtype=torch.float32)
        w_scale = grad.new_tensor(w_s, dtype=torch.float32)
        grad_scale = grad.new_tensor(grad_s, dtype=torch.float32)

        if grad.dtype != torch.float8_e5m2:
            grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        else:
            grad_f8 = grad
        
        # grad_x = grad @ w.T
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T, 
            out_dtype=torch.bfloat16,
            scale_a=grad_scale,
            scale_b=w_scale,
            use_fast_accum=False,
        )
        
        # grad_w = x.T @ grad
        # Result is (in, out), naturally matching weight storage. No final .T needed.
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_scale,
            scale_b=grad_scale,
            use_fast_accum=False,
        )
        
        return grad_x, grad_w

    grad_x, grad_w = impl(g, x_f8, w_f8)

    return grad_x, grad_w

@mm_t_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.to(torch.float32)

def backward_t(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_t_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context_t(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_t_op.register_autograd(backward_t, setup_context=setup_context_t)

@triton.jit
def fused_softcapped_entropy_fwd_kernel(
    logits_ptr, losses_ptr, lse_ptr, targets_ptr, mtp_weights_ptr,
    stride_logits_n, stride_logits_v,
    n_rows, n_cols, n_predict,
    A, B, C,
    BLOCK_SIZE: tl.constexpr,
    USE_SOFTCAPPING: tl.constexpr
):
    row_idx = tl.program_id(0).to(tl.int64)
    logits_row_ptr = logits_ptr + row_idx * stride_logits_n
    
    max_val = -float('inf')
    sum_exp = 0.0
    
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        val = tl.load(logits_row_ptr + cols, mask=mask, other=-float('inf')).to(tl.float32)
        if USE_SOFTCAPPING:
            z = A * tl.sigmoid((val + B) / C)
        else:
            z = val
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
                    if USE_SOFTCAPPING:
                        z_target = A * tl.sigmoid((val_target + B) / C)
                    else:
                        z_target = val_target
                    total_loss += weight * (lse - z_target)
    
    tl.store(losses_ptr + row_idx, total_loss)

@triton.jit
def fused_softcapped_entropy_bwd_kernel(
    grad_input_ptr, grad_output_ptr, lse_ptr, logits_ptr, targets_ptr, mtp_weights_ptr,
    stride_logits_n, stride_logits_v, stride_grad_n, stride_grad_v,
    n_rows, n_cols, n_predict,
    A, B, C,
    BLOCK_SIZE: tl.constexpr,
    USE_SOFTCAPPING: tl.constexpr
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
        if USE_SOFTCAPPING:
            u = (val + B) / C
            sigmoid_u = tl.sigmoid(u)
            z = A * sigmoid_u
        else:
            z = val
        p = tl.exp(z - lse)
        
        term1 = S_w * p
        term2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for k in range(n_predict):
            if row_idx + k < n_rows:
                target = tl.load(targets_ptr + row_idx + k).to(tl.int32)
                weight = tl.load(mtp_weights_ptr + k)
                term2 += tl.where(cols == target, weight, 0.0)
        
        grad_z = grad_loss * (term1 - term2)
        if USE_SOFTCAPPING:
            dz_dx = (1.0 / C) * z * (1.0 - sigmoid_u)
        else:
            dz_dx = 1
        grad_x = grad_z * dz_dx
        tl.store(grad_row_ptr + cols, grad_x.to(tl.bfloat16), mask=mask)

class FusedSoftcappedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, mtp_weights, USE_SOFTCAPPING, A=23.0, B=5.0, C=7.5):
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
            USE_SOFTCAPPING=USE_SOFTCAPPING,
            num_warps=2
        )
        
        ctx.save_for_backward(logits, targets, mtp_weights, lse)
        ctx.params = (A, B, C, USE_SOFTCAPPING)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, mtp_weights, lse = ctx.saved_tensors
        A, B, C, USE_SOFTCAPPING = ctx.params
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
            USE_SOFTCAPPING=USE_SOFTCAPPING,
            num_warps=2
        )

        return grad_input, None, None, None, None, None

@torch.compile(dynamic=False, fullgraph=True)
def kernel(x, lm_head_weight, target_seq, n_predict, mtp_weights):
    x = x.view(-1, x.shape[-1])
    logits = torch.ops.nanogpt.mm_t(x, lm_head_weight, x_s=x_s, w_s=w_s, grad_s=grad_s)[0]
    loss = FusedSoftcappedCrossEntropy.apply(logits, target_seq, mtp_weights, USE_SOFTCAPPING).sum().to(torch.bfloat16)
    return loss

@torch.compile(dynamic=False, fullgraph=True)
def reference(x, lm_head_weight, target_seq, n_predict, mtp_weights):
    x = x.view(-1, x.shape[-1])
    logits = torch.ops.nanogpt.mm_t(x, lm_head_weight, x_s=x_s, w_s=w_s, grad_s=grad_s)[0]
    if USE_SOFTCAPPING:
        logits = 23 * torch.sigmoid((logits + 5) / 7.5)
    logits_flat = logits.view(-1, logits.size(-1))
    idx = F.pad(target_seq, (0, n_predict - 1)).unfold(0, n_predict, 1)
    target_logits = logits_flat.gather(1, idx)
    cross_entropy = torch.logsumexp(logits_flat, dim=-1).unsqueeze(1) - target_logits
    for k in range(1, n_predict):
        cross_entropy[-k:, k] = 0
    loss = (cross_entropy * mtp_weights).sum()
    return loss

x_ref = x.clone()
lm_head_weight_ref = lm_head_weight.clone()
x_ref.retain_grad()
lm_head_weight_ref.retain_grad()

x_kernel = x.clone()
lm_head_weight_kernel = lm_head_weight.clone()
x_kernel.retain_grad()
lm_head_weight_kernel.retain_grad()

loss_ref = reference(x_ref, lm_head_weight_ref, target_seq, n_predict, mtp_weights).to(torch.bfloat16)

loss_ref.backward()

loss_kernel = kernel(x_kernel, lm_head_weight_kernel, target_seq, n_predict, mtp_weights)

loss_kernel.backward()

print("loss ref: ", loss_ref)
print("loss kernel: ", loss_kernel)

print("x_ref.grad:", x_ref.grad)
print("x_kernel.grad:", x_kernel.grad)

print("lm_head_ref.grad:", lm_head_weight_ref.grad)
print("lm_head_kernel.grad:", lm_head_weight_kernel.grad)

weight_matrix_size_gb_bf16 = dim * N * 2 / 1e9
SOL_weight_matrix_conversion_ms = 1.5 * weight_matrix_size_gb_bf16 / 3350 * 1000

print("LM head weight matrix size GB bf16:", weight_matrix_size_gb_bf16)
print("SOL weight matrix conversion ms:", SOL_weight_matrix_conversion_ms)

#torch.testing.assert_close(loss_ref, loss_kernel)
#torch.testing.assert_close(x_ref.grad, x_kernel.grad)
#torch.testing.assert_close(lm_head_weight_ref.grad, lm_head_weight_kernel.grad)

#print("PASS")

warmups = 5
iters = 100

for i in range(warmups):
    loss_ref = reference(x_ref, lm_head_weight_ref, target_seq, n_predict, mtp_weights).to(torch.bfloat16)
    loss_ref.backward()


torch.cuda.synchronize()
for i in range(iters):
    loss_ref = reference(x_ref, lm_head_weight_ref, target_seq, n_predict, mtp_weights).to(torch.bfloat16)
    loss_ref.backward()
torch.cuda.synchronize()

for i in range(warmups):
    loss_kernel = kernel(x_kernel, lm_head_weight_kernel, target_seq, n_predict, mtp_weights)
    loss_kernel.backward()

torch.cuda.synchronize()
start = time.time()
for i in range(iters):
    loss_kernel = kernel(x_kernel, lm_head_weight_kernel, target_seq, n_predict, mtp_weights)
    loss_kernel.backward()
torch.cuda.synchronize()
end = time.time()
avg_fwd_bwd_ms = (end - start) / iters * 1000

print("Average fwd bwd ms:", avg_fwd_bwd_ms)

