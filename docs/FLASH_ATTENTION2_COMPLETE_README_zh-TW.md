# FlashAttention-2（Python + CUDA）完整 README（繁體中文）

本文件是此倉庫 FlashAttention-2 backward 路徑的完整使用與實作說明。

涵蓋範圍：
- Python API 與 blocked 參考實作
- CUDA 封裝層與 CUDA kernel 入口
- 建置與執行流程（本機 + PSC SLURM）
- 各檔案架構與使用方式

---

## 1) 快速開始

### A. CPU / 參考路徑
使用 NumPy blocked 參考實作做正確性驗證：
- 呼叫 `flash_attention2_backward(..., use_cuda_kernel=False)`

### B. CUDA 路徑
1. 先建置共享函式庫：
   - `module load cuda`
   - `bash ./compile_cuda.sh`
2. 呼叫：
   - `flash_attention2_backward(..., use_cuda_kernel=True)`

### C. PSC 批次執行
使用：
- `run_fa2_cuda_tests.slurm`

提交：
- `sbatch run_fa2_cuda_tests.slurm`

---

## 2) 張量規格

### 必要形狀
- `q, k, v, out, dout`：`(B, H, T, D)`
- `logsumexp`：`(B, H, T)` 或 `(B, H, T, 1)`

符號：
- `B`：batch size
- `H`：attention head 數
- `T`：序列長度
- `D`：head 維度

### scale
- `softmax_scale` 必須與 forward 完全一致。
- 未指定時預設為 $1/\sqrt{D}$。

---

## 3) backward 使用數學

令
$$
S = \text{scale} \cdot QK^T + \text{mask},\quad
P = \text{softmax}(S),\quad
O = PV
$$

已知 `dO`：
$$
\begin{aligned}
dV &= P^T dO \\
dP &= dO V^T \\
D_i &= \langle dO_i, O_i \rangle \\
dS &= P \odot (dP - D) \\
dQ &= \text{scale} \cdot dS K \\
dK &= \text{scale} \cdot dS^T Q
\end{aligned}
$$

---

## 4) 檔案逐一說明

### 4.1 `minitorch/flash_attention2.py`
主要 API 與參考實作。

關鍵元素：
- `FlashAttention2ForwardContext`
  - forward/backward 串接的輕量 context
- `_validate_attention_shapes(...)`
  - 檢查 shape/rank 合約
- `flash_attention2_backward(...)`
  - 分派到 CUDA 或 NumPy blocked 路徑
- `flash_attention2_backward_from_context(...)`
  - 未來 forward 串接便利 API

執行流程：
1. 可選 CUDA dispatch
2. 轉 contiguous NumPy
3. query/key 分塊迴圈
4. 用 `logsumexp` 重建局部機率
5. 累積 `dq/dk/dv`
6. 轉回 miniTorch tensor

---

### 4.2 `minitorch/cuda_kernel_ops.py`
Python 到 CUDA 的 `ctypes` 橋接層。

關鍵元素：
- `HAS_FLASH_ATTENTION2_BWD`
  - 檢查 `combine.so` 是否匯出 `FlashAttention2Backward`
- `CudaKernelOps.flash_attention2_backward(...)`
  - shape 驗證
  - `logsumexp` 正規化
  - 設定 C ABI `argtypes`
  - 呼叫 C wrapper
  - 包裝輸出成 tensor

運作方式：
- Python 傳入 contiguous 1D 記憶體 + 額外 `(B,H,T,D)` 維度。
- C wrapper 內部做 H2D / kernel launch / D2H。

---

### 4.3 `src/combine.cu`
CUDA kernel 實作 + C ABI wrapper。

關鍵元素：
- `FlashAttention2BackwardKernel(...)`
  - 每個 thread 負責一個 `(b,h,i,j)` attention pair
  - 計算局部 score / probability
  - 計算 `dS`
  - 用 atomic 累積 `dQ/dK/dV`
- `extern "C" void FlashAttention2Backward(...)`
  - 供 `ctypes` 呼叫的入口
  - alloc/copy/launch/copy-back/free

備註：
- 目前設計以正確性優先
- 因為梯度會多對一累積，因此使用 atomic

---

### 4.4 `compile_cuda.sh`
CUDA 共享函式庫建置腳本：
- 產物：`minitorch/cuda_kernels/combine.so`

---

### 4.5 `tests/test_flash_attention2_backward.py`
正確性與 API 合約測試。

涵蓋：
- non-causal / causal 與 dense reference 比對
- 不可整除 tile 大小
- 接受 `(B,H,T,1)` 的 `logsumexp`
- 非法 shape / tile 的拒絕測試
- forward-context helper 等價性

---

### 4.6 `run_fa2_cuda_tests.slurm`
PSC GPU 批次驗證腳本。

流程：
1. 載入 CUDA module
2. 啟用 venv
3. 建置共享函式庫
4. 驗證 symbol
5. 執行 FA2 測試
6. 執行 explicit CUDA runtime check

---

## 5) 使用方式

### A. 直接 API
- 準備 `q,k,v,out,logsumexp,dout`
- 呼叫 `flash_attention2_backward(...)`

### B. Context API
- forward 保存 `FlashAttention2ForwardContext`
- backward 呼叫 `flash_attention2_backward_from_context(...)`

### C. 強制路徑
- CUDA：`use_cuda_kernel=True`
- 參考：`use_cuda_kernel=False`

---

## 6) 典型整合流程

1. forward 計算並保存：
   - `out`
   - `logsumexp`
   - `causal`
   - `softmax_scale`
2. backward 取得 `dout`
3. 呼叫 FA2 backward（直接或 context helper）
4. 把 `dq, dk, dv` 接回 autograd 流程

---

## 7) 疑難排解

### `FlashAttention2Backward CUDA symbol not found`
- 重新建置：`bash ./compile_cuda.sh`
- 驗證：`nm -D minitorch/cuda_kernels/combine.so | grep FlashAttention2Backward`

### Shell 看不到 GPU
- 先申請 GPU 資源（`salloc`/`sbatch`）
- 建置/執行前先 `module load cuda`

### 數值不一致
- 確認 forward/backward 完全一致：
  - `causal`
  - `softmax_scale`
  - 同一次 forward 產生的 `logsumexp`

---

## 8) 相關文件

- `docs/FLASH_ATTENTION2_BACKWARD.md`
- `docs/FLASH_ATTENTION2_BACKWARD_zh-TW.md`
- `docs/FLASH_ATTENTION2_BACKWARD_DEVELOPER_GUIDE.md`
- `docs/FLASH_ATTENTION2_BACKWARD_DEVELOPER_GUIDE_zh-TW.md`
- `docs/FLASH_ATTENTION2_BACKWARD_CUDA.md`
- `docs/FLASH_ATTENTION2_BACKWARD_CUDA_zh-TW.md`
