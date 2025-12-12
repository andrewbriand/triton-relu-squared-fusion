import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.cuda.nvtx import range_push, range_pop
import time

batch = 24 * 2048 * 8
dim = 768
hdim = 4 * dim

dtype = torch.bfloat16

x = torch.randn((batch, dim), dtype=dtype, device="cuda")
W1 = torch.randn((dim, hdim), dtype=dtype, device="cuda")
W2 = torch.randn((dim, hdim), dtype=dtype, device="cuda")

def reference(x, W1, W2):
  range_push("Unfused forward")
  x1 = x @ W1
  x2 = F.relu(x1).square()
  x3 = x2 @ W2.T
  range_pop()
  return x3

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


def get_autotune_config():
    return get_cuda_autotune_config()

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config()[:1],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr, c_pre_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_pre = accumulator.to(a_ptr.type.element_ty)
    c_post = tl.maximum(c_pre, 0)
    c_post = c_post * c_post

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c_post, mask=c_mask)

    c_pre_ptrs = c_pre_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_pre_ptrs, c_pre, mask=c_mask)


def forward_kernel(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    pre = torch.empty((M, N), device=a.device, dtype=a.dtype)
    post = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, post, pre, #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        post.stride(0), post.stride(1),  #
    )
    return pre, post

@triton.autotune(
    configs=get_autotune_config()[:1],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_bwd(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr, c_pre_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_pre_ptrs = c_pre_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    c_pre = tl.load(c_pre_ptrs, mask=c_mask)

    dpost = accumulator.to(a_ptr.type.element_ty)
    dpre = 2 * dpost * tl.where(c_pre > 0, c_pre, 0)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, dpre, mask=c_mask)

def bwd_kernel(a, b, pre):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    assert(pre.shape == (M, N))
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    assert pre.stride(0) == c.stride(0)
    assert pre.stride(1) == c.stride(1)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel_bwd[grid](
        a, b, c, pre, #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
    )
    return c


def custom_kernels(x, W1, W2):
  pre, x1 = forward_kernel(x, W1)
  x3 = x1 @ W2.T
  return pre, x3

class FusedLinearReLUSquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2):
        range_push("fused fwd")
        pre, post = forward_kernel(x, W1)
        x3 = post @ W2.T
        ctx.save_for_backward(x, W1, W2, pre, post)
        range_pop()
        return x3

    @staticmethod
    def backward(ctx, grad_output):
        range_push("Fused bwd")
        x, W1, W2, pre, post = ctx.saved_tensors

        # grad_output is [batch x dim]
        # post is [batch x hdim]
        # dW2 is dim x hdim
        dW2 = grad_output.T @ post

        # d / dx (relu(x))^2
        # 2 * relu(x) * (x > 0)
        # grad_output is [batch x dim]
        # W2 is [dim x hdim]
        # dpost is [batch x hdim]
        #dpost = grad_output @ W2
        #dpre = 2 * dpost * F.relu(pre)
        dpre = bwd_kernel(grad_output, W2, pre)

        # dpre is [batch x hdim]
        # x is [batch x dim]
        # dW1 is [dim x hdim]
        dW1 = x.T @ dpre

        # dpre is [batch x hdim]
        # W1 is [dim x hdim]
        # dx is [batch x dim]
        dx = dpre @ W1.T

        range_pop()
        
        return dx, dW1, dW2


x_ref = x.detach().clone().requires_grad_(True)
W1_ref = W1.detach().clone().requires_grad_(True)
W2_ref = W2.detach().clone().requires_grad_(True)

out_ref = reference(x_ref, W1_ref, W2_ref)

x_kernel = x.detach().clone().requires_grad_(True)
W1_kernel = W1.detach().clone().requires_grad_(True)
W2_kernel = W2.detach().clone().requires_grad_(True)
out_kernel = FusedLinearReLUSquareFunction.apply(x_kernel, W1_kernel, W2_kernel)

grad_out = torch.randn_like(out_ref)

range_push("Unfused bwd")
out_ref.backward(grad_out)
range_pop()

out_kernel.backward(grad_out)

torch.testing.assert_close(out_ref, out_kernel)

torch.testing.assert_close(W2_ref.grad, W2_kernel.grad)
torch.testing.assert_close(W1_ref.grad, W1_kernel.grad)
torch.testing.assert_close(x_ref.grad, x_kernel.grad)

print("PASS")

# Benchmark fwd

iters = 100

bw_4090_gb_s = 1000
tflops_4090 = 165.2

bw_h100_gb_s = 3350
tflops_h100 = 989

torch.cuda.synchronize()
start = time.time()
for i in range(iters):
   pre, post = forward_kernel(x, W1)
torch.cuda.synchronize()
end = time.time()

avg_time_ms = (end - start) / iters * 1000
print("Average fwd time (ms):", avg_time_ms)

fwd_traffic_input_elements = batch * dim + dim * hdim
fwd_traffic_output_elements = 2 * batch * hdim
fwd_traffic_input_gb = 2 * fwd_traffic_input_elements / 1e9
fwd_traffic_output_gb = 2 * fwd_traffic_output_elements / 1e9
fwd_traffic_gb = fwd_traffic_input_gb + fwd_traffic_output_gb 
fwd_bw_gb_s = fwd_traffic_gb / (avg_time_ms / 1000)
fwd_bw_util = int(fwd_bw_gb_s / bw_4090_gb_s * 100)
fwd_bw_util_h100 = int(fwd_bw_gb_s / bw_h100_gb_s * 100)

fwd_tflops = ((2 * batch * dim * hdim) / 1e12) / (avg_time_ms / 1000)
fwd_tflops_util = int(fwd_tflops / tflops_4090 * 100)
fwd_tflops_util_h100 = int(fwd_tflops / tflops_h100 * 100)

print("Forward input traffic (GB):", fwd_traffic_input_gb)
print("Forward output traffic (GB):", fwd_traffic_output_gb)
print("Forward BW (GB / s):", fwd_bw_gb_s)
print("Forward BW util RTX 4090 (%):", fwd_bw_util)
print("Forward BW util H100 (%):", fwd_bw_util_h100)
print("Forward TFLOPS:", fwd_tflops)
print("Forward TFLOPS util RTX 4090 (%):", fwd_tflops_util)
print("Forward TFLOPS util H100 (%):", fwd_tflops_util_h100)
print()

# Benchmark bwd
torch.cuda.synchronize()

start = time.time()
for i in range(iters):
  bwd_kernel(grad_out, W2, pre)
torch.cuda.synchronize()
end = time.time()

avg_time_ms_bwd = (end - start) / iters * 1000
print("Average bwd time (ms):", avg_time_ms_bwd)

bwd_traffic_input_elements = batch * dim + dim * hdim + batch * hdim
bwd_traffic_output_elements = batch * hdim
bwd_traffic_input_gb = bwd_traffic_input_elements * 2 / 1e9
bwd_traffic_output_gb = bwd_traffic_output_elements * 2 / 1e9
bwd_traffic_gb = bwd_traffic_input_gb + bwd_traffic_output_gb
bwd_bw_gb_s = bwd_traffic_gb / (avg_time_ms_bwd / 1000)
bwd_bw_util = int(bwd_bw_gb_s / bw_4090_gb_s * 100)
bwd_bw_util_h100 = int(bwd_bw_gb_s / bw_h100_gb_s * 100)

bwd_tflops = ((2 * batch * dim * hdim) / 1e12) / (avg_time_ms_bwd / 1000)
bwd_tflops_util = int(bwd_tflops / tflops_4090 * 100)
bwd_tflops_util_h100 = int(bwd_tflops / tflops_h100 * 100)

print("Backward input traffic (GB):", bwd_traffic_input_gb)
print("Backward output traffic (GB):", bwd_traffic_output_gb)
print("Backward BW (GB / s):", bwd_bw_gb_s)
print("Backward BW util RTX 4090 (%):", bwd_bw_util)
print("Backward BW util H100 (%):", bwd_bw_util_h100)
print("Backward TFLOPS:", bwd_tflops)
print("Backward TFLOPS util RTX 4090 (%):", bwd_tflops_util)
print("Backward TFLOPS util H100:", bwd_tflops_util_h100)

