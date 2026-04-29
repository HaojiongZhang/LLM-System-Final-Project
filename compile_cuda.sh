mkdir -p minitorch/cuda_kernels
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
nvcc -O2 -shared -Xcompiler -fPIC \
    -o minitorch/cuda_kernels/paged_attn.so \
    minitorch/cuda_kernels/paged_attn.cu
