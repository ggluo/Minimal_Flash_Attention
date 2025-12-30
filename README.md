# Minimal Flash Attention - CUDA Implementation

## Overview

**Minimal Flash Attention** is a minimal, educational implementation of the Flash Attention algorithm in CUDA for inference workloads. This project provides a clear, step-by-step implementation of the Flash Attention algorithm with progressive optimizations across 8 different kernel variants (0-7), showcasing various CUDA optimization techniques.

The Flash Attention algorithm is an efficient attention mechanism that reduces memory bandwidth requirements by computing attention scores in a tiled manner with online softmax computation, avoiding the need to store the full attention matrix.

## Key Features

- **8 Progressive Kernel Implementations**: From basic to highly optimized versions including Tensor Core utilization
- **Educational Focus**: Clean, well-commented code suitable for learning CUDA optimization techniques
- **Performance Benchmarking**: Compare performance across different optimization strategies
- **Algorithm Implementation**: Full implementation of Flash Attention with tiling and online softmax
- **Modular Design**: Separate kernels for different optimization strategies

## Project Structure

```
Minimal_Flash_Attention/
├── CMakeLists.txt          # CMake build configuration
├── flash.cu               # Main driver program
├── run.sh                 # Automated build and test script
├── README.md              # This file
└── src/
    ├── kernel.cuh         # Kernel header aggregator
    ├── utils.cu           # Utility function implementations
    ├── utils.cuh          # Utility function declarations
    └── kernel/            # Kernel implementations
        ├── common.cuh     # Common reduction operations (MaxOp, SumOp, WarpAllReduce)
        ├── kernel_0.cuh   # WMMA Tensor Core implementation
        ├── kernel_1.cuh   # Basic Flash Attention implementation
        ├── kernel_2.cuh   # Shared memory optimizations
        ├── kernel_3.cuh   # Loop unrolling (Rq=2, Rv=4)
        ├── kernel_4.cuh   # Further unrolling (Rq=3, Rv=4)
        ├── kernel_5.cuh   # Using float4
        ├── kernel_6.cuh   # Optimized warp-level reductions
        └── kernel_7.cuh   # Optimized pipeline implementation
```

## Kernel Variants

| Kernel | Description | Key Optimizations | Block Sizes |
|--------|-------------|-------------------|-------------|
| **0** | WMMA Tensor Core Implementation | Uses NVIDIA's WMMA API for 16x16 matrix operations on Tensor Cores | Block: 32 threads (1 warp)<br>Tile: 16x16 |
| **1** | Basic Implementation | Initial Flash Attention with tiling and online softmax | Br=32, Bc=32 |
| **2** | Shared Memory Optimized | Block tiling using shared memory usage patterns with better memory coalescing, one thread handle an element in a tile | Br=32, Bc=32 |
| **3** | Register Tiling | Register tiling, one thread handle a tile, better instruction-level parallelism，warp-level reductions | Br=32, Bc=32<br>Rq=2, Rv=4 |
| **4** | Better Register Tiling | Kernel 3 + shared memory reuse | Br=32, Bc=32<br>Rq=3, Rv=4 |
| **5** | Vectorized Instructions | Kernel 4 + Vectorized memory operations using `float4` | Br=32, Bc=32<br>Rq=3, Rv=4 |
| **6** | bank-conflict-free | padded shared memory | Br=16, Bc=16<br>Rq=8, Rv=8<br>Bk=8, Bd=8 |
| **7** | Optimized Pipeline Implementation | Double-buffered shared memory pipeline, advanced vectorization, optimized computation patterns | Br=16, Bc=16<br>Rq=8, Rv=8<br>Bk=8, Bd=8 |

## Algorithm Details

### Flash Attention Algorithm
The implementation follows the standard Flash Attention algorithm:

1. **Tiling**: Split Q, K, V matrices into blocks (Br × d, Bc × d)
2. **Online Softmax**: Compute softmax incrementally to avoid storing full attention matrix
3. **Memory Hierarchy**: Efficient use of shared memory and registers
4. **Numerical Stability**: Safe softmax computation with max subtraction

### Matrix Dimensions
- **N** = 1024 (sequence length)
- **d** = 1024 (embedding dimension)
- **Br** = 32 (Q block rows)
- **Bc** = 32 (K/V block columns)

## Prerequisites

- **NVIDIA GPU** with Compute Capability ≥ 7.0 (Volta architecture or newer)
- **CUDA Toolkit** (version 11.0 or newer recommended)
- **CMake** (version 3.0 or newer)
- **C++ Compiler** with C++11 support

## Build Instructions

### Method 1: Using the provided script (Recommended)
```bash
# Make the script executable
chmod +x run.sh

# Run the script (builds and tests all kernels)
./run.sh
```

### Method 2: Manual build with CMake
```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the project
make

# Return to project root
cd ..
```

The build process will generate an executable named `flash` in the project root directory.

## Usage

### Basic Usage
```bash
# Run a specific kernel (0-7)
./flash <kernel_number>
```

### Examples
```bash
# Run WMMA Tensor Core implementation (kernel 0)
./flash 0

# Run basic Flash Attention implementation (kernel 1)
./flash 1

# Run optimized implementation (kernel 4)
./flash 4

# Run advanced vectorized implementation (kernel 6)
./flash 6

# Run optimized pipeline implementation (kernel 7)
./flash 7
```

### Testing All Kernels
The `run.sh` script automatically builds the project and runs all kernels (0-7), displaying their execution times:
```bash
./run.sh
```

### Program Output
The program will display:
- Selected kernel number
- Average execution time over 10 runs (in milliseconds)
- Matrix dimensions and block sizes used

## Performance Characteristics

Each kernel represents different optimization levels with expected performance improvements:

1. **Kernel 0**: WMMA Tensor Core implementation - utilizes NVIDIA Tensor Cores for matrix operations
2. **Kernel 1**: Basic implementation - demonstrates the core Flash Attention algorithm
3. **Kernel 2**: Shared memory optimizations - reduces global memory accesses
4. **Kernel 3-4**: Loop unrolling - improves instruction-level parallelism
5. **Kernel 5-7**: Advanced optimizations - vectorization, warp-level reductions, and pipeline optimizations

The performance progression from kernel 1 to 7 demonstrates the impact of various CUDA optimization techniques on the Flash Attention algorithm.

## Educational Value

This project serves as an excellent educational resource for:

### CUDA Programming Concepts
- **Shared Memory Usage**: Efficient data sharing between threads
- **Warp-Level Operations**: Using `__shfl_xor_sync` for reductions
- **Memory Coalescing**: Optimized memory access patterns
- **Kernel Launch Configuration**: Grid and block dimension tuning

### Algorithm Implementation
- **Online Softmax**: Incremental computation for numerical stability
- **Tiling Strategies**: Block matrix operations for large matrices
- **Numerical Precision**: Handling floating-point precision issues

### Optimization Techniques
- **Loop Unrolling**: Manual and compiler-assisted unrolling
- **Instruction-Level Parallelism**: Using FMA (fused multiply-add) instructions
- **Memory Hierarchy**: Effective use of registers, shared memory, and global memory

## Implementation Details

### Key Components

#### 1. Common Operations (`src/kernel/common.cuh`)
- `WarpAllReduce`: Template for warp-level reductions
- `MaxOp`, `SumOp`: Functors for max and sum operations

#### 2. Utility Functions (`src/utils.cu/.cuh`)
- Matrix initialization and verification
- CUDA error checking
- Timing functions
- Device information display

#### 3. Main Driver (`flash.cu`)
- Command-line argument parsing
- Memory allocation and data transfer
- Kernel execution and timing
- Cleanup and result reporting

#### 4. Kernel Implementations
Each kernel in `src/kernel/` demonstrates different optimization strategies while maintaining the same algorithmic correctness.

## References

1. Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

2. ggluo: [The details of flash attention - algorithm](https://ggluo.github.io/blog/flash-attention-1)


## Future Work

Potential enhancements and extensions:

1. **Flexible Matrix Sizes**: Support for arbitrary N and d dimensions
2. **Batch Processing**: Extension to batch attention computation
3. **Mixed Precision**: Support for FP16 and BF16 precision
4. **Multi-GPU Support**: Distributed attention computation
5. **Attention Variants**: Implementation of different attention mechanisms
6. **Performance Profiling**: Detailed performance analysis tools
7. **Python Bindings**: PyTorch/CUDA extensions for easy integration

## License

This project is intended for educational purposes. Please refer to the original repository for specific licensing information.

## Contributing

This is an educational project. For contributions, please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear documentation of changes

---

**Note**: This implementation is optimized for clarity and educational value. For production use, consider using optimized libraries like FlashAttention-2 or vendor-optimized implementations.
