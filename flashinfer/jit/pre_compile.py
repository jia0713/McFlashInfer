import torch
import flashinfer
import warnings

warnings.filterwarnings("ignore", module="torch")

device = 'cpu'
batch_size = 1
head_dim_qk = 192
head_dim_vo = 128
qo_len = 3201
kv_len = 3201
nheads_q = 128
nheads_k = 128
kv_layout = "NHD"
# ===================================================================================== #
#                          pre_compile normal prefill bfloat16                          #
# ===================================================================================== #
dtype = torch.bfloat16

q_indptr = torch.arange(0, batch_size + 1, device='cpu').int() * qo_len
kv_indptr = torch.arange(0, batch_size + 1, device='cpu').int() * kv_len

workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device='cpu')

wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
    workspace_buffer, kv_layout, backend="fa2"
)
wrapper.pre_complie_plan(
    q_indptr,
    kv_indptr,
    nheads_q,
    nheads_k,
    head_dim_qk,
    head_dim_vo=head_dim_vo,
    causal=True,
    q_data_type=dtype,
    kv_data_type=dtype,
)

wrapper.pre_complie_plan(
    q_indptr,
    kv_indptr,
    nheads_q,
    nheads_k,
    head_dim_qk,
    head_dim_vo=head_dim_vo,
    causal=False,
    q_data_type=dtype,
    kv_data_type=dtype,
)

# ===================================================================================== #
#                          pre_compile normal prefill float16                          #
# ===================================================================================== #
dtype = torch.float16

q_indptr = torch.arange(0, batch_size + 1, device='cpu').int() * qo_len
kv_indptr = torch.arange(0, batch_size + 1, device='cpu').int() * kv_len

workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device='cpu')

wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
    workspace_buffer, kv_layout, backend="fa2"
)

wrapper.pre_complie_plan(
    q_indptr,
    kv_indptr,
    nheads_q,
    nheads_k,
    head_dim_qk,
    head_dim_vo=head_dim_vo,
    causal=True,
    q_data_type=dtype,
    kv_data_type=dtype,
)

wrapper.pre_complie_plan(
    q_indptr,
    kv_indptr,
    nheads_q,
    nheads_k,
    head_dim_qk,
    head_dim_vo=head_dim_vo,
    causal=False,
    q_data_type=dtype,
    kv_data_type=dtype,
)