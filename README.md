# Minimal Flash Attention - CUDA Implementation

## Overview

**Minimal Flash Attention** is an educational CUDA implementation of the Flash Attention algorithm for inference workloads. The project demonstrates progressive optimization through 8 kernel variants (1-8), each building upon the previous with advanced CUDA techniques.

The implementation follows the standard Flash Attention algorithm with tiling and online softmax to avoid storing the full attention matrix, reducing memory bandwidth requirements.

## Kernel Variants

| Kernel | Optimization Focus | Key Techniques | Block Size | Register Tiling | Performance* |
|--------|-------------------|----------------|------------|-----------------|--------------|
| **1** | Baseline | Basic tiling, online softmax | Br=32, Bc=32 | 1×1 | 258.18 ms |
| **2** | Memory Coalescing | Shared memory optimization, better access patterns | Br=32, Bc=32 | 1×1 | 32.90 ms |
| **3** | Register Usage | Warp-level reductions, instruction-level parallelism | Br=32, Bc=32 | Rq=2, Rv=4 | 6.11 ms |
| **4** | Shared Memory Reuse | Improved buffer reuse, efficient shared memory | Br=32, Bc=32 | Rq=3, Rv=4 | 4.93 ms |
| **5** | Vectorization | `float4` vectorized memory operations | Br=32, Bc=32 | Rq=3, Rv=4 | 5.76 ms |
| **6** | Bank Conflict Free | Padded shared memory, reduced conflicts | Br=16, Bc=16 | Rq=8, Rv=8 | 2.41 ms |
| **7** | Pipeline Optimization | Double-buffered shared memory, compute/memory overlap | Br=16, Bc=16 | Rq=8, Rv=8 | 2.12 ms |
| **8** | Tensor Cores | WMMA API, FP16 precision, 16×16 tile operations | 32 threads (1 warp) | 16×16 tiles | 15.76 ms |

*Performance measured on N=1024, d=1024 matrices (average of 10 runs on NVIDIA RTX 4500 Ada)

**Note**: Kernel 8 uses FP16 precision for Tensor Core operations, resulting in higher numerical error (max diff: 1.65e-02) compared to FP32 reference.

## Algorithm

The implementation follows the Flash Attention algorithm:
1. **Tiling**: Split Q, K, V matrices into blocks (Br × d, Bc × d)
2. **Online Softmax**: Compute softmax incrementally to avoid storing full attention matrix
3. **Memory Hierarchy**: Efficient use of registers, shared memory, and global memory
4. **Numerical Stability**: Safe softmax computation with max subtraction

**Matrix Dimensions**: N = 1024 (sequence length), d = 1024 (embedding dimension)

## Prerequisites

- **NVIDIA GPU** with Compute Capability ≥ 7.0 (Volta or newer)
- **CUDA Toolkit** (version 11.0 or newer)
- **CMake** (version 3.0 or newer)
- **C++ Compiler** with C++11 support

## Build & Usage

### Quick Start
```bash
chmod +x run.sh    # Make script executable
./run.sh           # Build, test, and validate all kernels
```

The `run.sh` script automates the entire process:
1. **Build**: Compiles the project using CMake, generating `flash` executable
2. **Test**: Runs all 8 kernels with N=1024, d=1024 matrices
3. **Validate**: Compares outputs with Python reference (tolerance: 1e-4)

### Running Individual Kernels
```bash
./flash <kernel_number>  # Run specific kernel (1-8)

# Examples:
./flash 1  # Baseline implementation
./flash 3  # Register tiling optimization  
./flash 8  # Tensor Core (FP16) implementation
```

**Output**: Kernel number, execution time (ms), and output file (`kernel<number>_output.txt`)

## Validation

The validation system ensures correctness across all kernels:

### Process
1. Python reference (`flash.py`) generates ground truth using standard self-attention
2. Each CUDA kernel (1-8) executes and produces output
3. Outputs are compared with reference using `compare.py` (tolerance: 1e-4)

### Results
- **Kernels 1-7**: ✅ PASS (all values within 1e-4 tolerance)
- **Kernel 8**: ⚠️ Higher error (max diff: 1.65e-02) due to FP16 Tensor Core precision

**Tools**: `compare.py` (comparison), `flash.py` (reference), `run.sh` (automated testing)

## Implementation

### Code Structure
- **`src/kernel/common.cuh`**: Warp-level reduction templates (`WarpAllReduce`, `MaxOp`, `SumOp`)
- **`src/kernel/kernel_*.cuh`**: 8 kernel variants with progressive optimizations
- **`flash.cu`**: Main driver for kernel execution and timing
- **`src/utils.cu/.cuh`**: Utility functions for matrix operations and error checking

### Optimization Progression
1. **Kernels 1-2**: Memory access patterns and shared memory usage
2. **Kernels 3-5**: Register tiling, vectorization, and warp-level operations
3. **Kernels 6-7**: Advanced techniques (bank conflict avoidance, pipelining)
4. **Kernel 8**: Tensor Core utilization with WMMA API (FP16 precision)

## Performance Summary

The kernel table includes actual performance data from terminal_output (N=1024, d=1024, average of 10 runs). Key observations:

- **Optimization Impact**: Progressive optimizations yield over 120× speedup from Kernel 1 to Kernel 7
- **Tensor Cores**: Kernel 8 uses FP16 Tensor Cores (15.76 ms) - faster than baseline but with precision trade-off
- **Best Performance**: Kernel 7 achieves 2.12 ms (121.8× speedup) with advanced pipelining

**Note**: Performance varies with GPU architecture and CUDA version.

## References

1. Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

2. ggluo: [The details of flash attention - algorithm](https://ggluo.github.io/blog/flash-attention-1)

## License

This project is intended for educational purposes. Please refer to the original repository for specific licensing information.