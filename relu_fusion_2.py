import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.cuda.nvtx import range_push, range_pop
import time
from triton.tools.tensor_descriptor import TensorDescriptor

batch = 24 * 2048
dim = 768
hdim = 4 * dim

dtype = torch.bfloat16

x = torch.randn((batch, dim), dtype=dtype, device="cuda")
W1 = torch.randn((hdim, dim), dtype=dtype, device="cuda")
W2 = torch.randn((hdim, dim), dtype=dtype, device="cuda")

def reference(x, W1, W2):
  range_push("Unfused forward")
  x1 = x @ W1.T
  x2 = F.relu(x1).square()
  x3 = x2 @ W2
  range_pop()
  return x3

@triton.jit
def matmul_kernel_tma_persistent(a_desc, b_desc, c_desc, aux_desc,  #
                                 M, N, K,  #
                                 BLOCK_SIZE_M: tl.constexpr,  #
                                 BLOCK_SIZE_N: tl.constexpr,  #
                                 BLOCK_SIZE_K: tl.constexpr,  #
                                 GROUP_SIZE_M: tl.constexpr,  #
                                 NUM_SMS: tl.constexpr,  #
                                 FORWARD: tl.constexpr,
                                 ):
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)

        c0 = acc0.to(dtype)
        if not FORWARD:
            c0_pre = aux_desc.load([offs_am_c, offs_bn_c])
            c0 = 2 * c0 * tl.where(c0_pre > 0, c0_pre, 0)

        c_desc.store([offs_am_c, offs_bn_c], c0)

        if FORWARD:
            c0_post = tl.maximum(c0, 0)
            c0_post = c0_post * c0_post
            aux_desc.store([offs_am_c, offs_bn_c], c0_post)

        c1 = acc1.to(dtype)
        if not FORWARD:
            c1_pre = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
            c1 = 2 * c1 * tl.where(c1_pre > 0, c1_pre, 0)

        c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)

        if FORWARD:
            c1_post = tl.maximum(c1, 0)
            c1_post = c1_post * c1_post
            aux_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1_post)


def matmul_tma_persistent(a, b, aux=None):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    FORWARD = False
    if aux is None:
        FORWARD = True
        aux = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    num_stages = 4 if FORWARD else 3
    num_warps = 8

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    aux_desc = TensorDescriptor.from_tensor(aux, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])

    def grid(META):
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
        ), )

    matmul_kernel_tma_persistent[grid](
        a_desc, b_desc, c_desc, aux_desc,#
        M, N, K,  #
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=1,
        NUM_SMS=NUM_SMS,  #
        FORWARD=FORWARD,
        num_stages=num_stages,
        num_warps=num_warps
    )

    if FORWARD:
        return c, aux
    else:
        return c

class FusedLinearReLUSquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2):
        range_push("fused fwd")
        pre, post = matmul_tma_persistent(x, W1)
        x3 = post @ W2
        ctx.save_for_backward(x, W1, W2, pre, post)
        range_pop()
        return x3

    @staticmethod
    def backward(ctx, grad_output):
        range_push("Fused bwd")
        x, W1, W2, pre, post = ctx.saved_tensors

        # grad_output is [batch x dim]
        # post is [batch x hdim]
        # dW2 is hdim x dim
        dW2 = post.T @ grad_output

        # d / dx (relu(x))^2
        # 2 * relu(x) * (x > 0)
        # grad_output is [batch x dim]
        # W2 is [hdim x dim]
        # dpost is [batch x hdim]
        #dpost = grad_output @ W2
        #dpre = 2 * dpost * F.relu(pre)
        #dpre = bwd_kernel(grad_output, W2, pre)
        dpre = matmul_tma_persistent(grad_output, W2, aux=pre)

        # dpre is [batch x hdim]
        # x is [batch x dim]
        # dW1 is [hdim x dim]
        dW1 = dpre.T @ x

        # dpre is [batch x hdim]
        # W1 is [hdim x dim]
        # dx is [batch x dim]
        dx = dpre @ W1

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

print("Max abs diff out: ", torch.max(torch.abs(out_ref - out_kernel)))

torch.testing.assert_close(W2_ref.grad, W2_kernel.grad)
torch.testing.assert_close(W1_ref.grad, W1_kernel.grad)
torch.testing.assert_close(x_ref.grad, x_kernel.grad)

print("Max abs diff W2:", torch.max(torch.abs(W2_ref.grad - W2_kernel.grad)))
print("Max abs diff W1:", torch.max(torch.abs(W1_ref.grad - W1_kernel.grad)))
print("Max abs diff x:", torch.max(torch.abs(x_ref.grad - x_kernel.grad)))

print("PASS")

# Benchmark fwd

iters = 100

bw_4090_gb_s = 1000
tflops_4090 = 165.2

bw_h100_gb_s = 3350
tflops_h100 = 989

for i in range(5):
    pre, post = matmul_tma_persistent(x, W1)

torch.cuda.cudart().cudaProfilerStart()
torch.cuda.synchronize()
start = time.time()
for i in range(iters):
   pre, post = matmul_tma_persistent(x, W1)
torch.cuda.synchronize()
end = time.time()
torch.cuda.cudart().cudaProfilerStop()

avg_time_ms = (end - start) / iters * 1000
print("Average fwd time (ms):", avg_time_ms)

# unfused matmul time
torch.cuda.synchronize()
start = time.time()
for i in range(iters):
   pre = x @ W1.T
torch.cuda.synchronize()
end = time.time()
avg_time_ms_unfused_matmul = (end - start) / iters * 1000
print("Average unfused matmul time (ms):", avg_time_ms_unfused_matmul)

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

time.sleep(3)

# unfused matmul time
torch.cuda.synchronize()
start = time.time()
for i in range(iters):
   pre = grad_out @ W2.T
torch.cuda.synchronize()
end = time.time()
avg_time_ms_unfused_matmul = (end - start) / iters * 1000
print("Average unfused matmul time bwd (ms):", avg_time_ms_unfused_matmul)

# Benchmark bwd
torch.cuda.synchronize()

start = time.time()
for i in range(iters):
  matmul_tma_persistent(grad_out, W2, aux=pre)
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

