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
- 未指定時預設為 `1 / sqrt(D)`。

---

## 3) backward 使用數學

為了在不同 Markdown 環境（GitHub、IDE 預覽、文件網站）都穩定顯示，
這裡改用純文字公式：

```text
S = scale * (Q K^T) + mask
P = softmax(S)
O = P V

Given dO:
dV = P^T dO
dP = dO V^T
D_i = dot(dO_i, O_i)
dS = P * (dP - D)
dQ = scale * dS K
dK = scale * dS^T Q
```

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
  - 每個 block 擁有一個 K/V tile `(b,h,k_tile)`
  - key tile 大小自適應（`D<=64` 用 `BK=64`，否則 `BK=32`）
  - shared memory 暫存 `K_j`、`V_j`，並以 padding（`D+1`）降低 bank conflict
  - block 內合作 reduction 計算：
    - `dot(q_i, k_j)`（score）
    - `dP_ij = dot(dO_i, V_j)`
    - `D_i = dot(dO_i, O_i)`
  - `dK` / `dV` 先在 shared tile 累積，再一次寫回（大幅降低 global atomic）
  - `dQ` 因 query row 會被多個 `k_tile` block 造訪，仍需 atomic
- `extern "C" void FlashAttention2Backward(...)`
  - 供 `ctypes` 呼叫的入口
  - alloc/copy/launch/copy-back/free

備註：
- 目前已是較實務的 tiled kernel（不再是單純 pairwise kernel）
- 透過 shared memory 提升資料區域性並降低 global memory 流量
- 主要 global 競爭已從 `dK/dV` 轉移；剩餘明顯競爭點為 `dQ` atomic

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

## 8) 最新 GPU 驗證結果（PSC）

最新驗證工作：
- Job ID：`37923696`
- 狀態：`COMPLETED`
- ExitCode：`0:0`
- Log：`fa2_cuda_test_37923696.log`

觀察結果：
- FA2 backward 測試：`6 passed`
- 額外 CUDA 路徑檢查：成功
- 簡易時間量測（單次、含 Python/封裝層開銷）：
  - shape `(B=2,H=4,T=128,D=64)`
  - `cuda_ms=2397.886`
  - `ref_ms=2346.258`
  - 速度比（`ref/cuda`）`= 0.98x`

解讀：
- kernel 已改為 K/V-tile ownership，且 `dK/dV` 先在 shared memory 累積後再寫回，
- 但端到端時間仍受封裝層 H2D/D2H copy 與 `dQ` atomic 成本影響。

---

## 9) 相關文件

- `docs/FLASH_ATTENTION2_COMPLETE_README.md`
- `docs/FLASH_ATTENTION2_COMPLETE_README_zh-TW.md`
