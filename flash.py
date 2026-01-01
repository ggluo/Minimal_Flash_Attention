#!/usr/bin/env python3
"""
Flash Attention Python Reference Implementation

This script implements the Flash Attention algorithm in Python for verification
of CUDA kernel correctness. It generates the same random data as the CUDA
version and computes the attention output using the standard Flash Attention
algorithm with online softmax.
"""

import sys
import numpy as np
import time

def randomize_matrix_python(shape, seed=42):
    """
    Generate random matrix with the same distribution as CUDA's randomize_matrix
    
    CUDA implementation:
    - Uses time.tv_usec as seed
    - Generates values: (rand() % 5) + 0.01 * (rand() % 5)
    - Randomly makes values negative with 50% probability
    
    For reproducibility, we use a fixed seed but mimic the distribution.
    """
    rng = np.random.RandomState(seed)
    N = np.prod(shape)
    
    # Generate random integers 0-4 for both parts
    int_part = rng.randint(0, 5, size=N)
    frac_part = rng.randint(0, 5, size=N) * 0.01
    values = int_part + frac_part
    
    # Randomly make values negative with 50% probability
    signs = rng.choice([1, -1], size=N, p=[0.5, 0.5])
    values = values * signs
    
    return values.reshape(shape)

def standard_self_attention(Q, K, V):
    """
    1. Standard Self-Attention with Safe Softmax.
    Memory Bound: Computes the full N x N attention matrix.
    Based on attn.py's standard_self_attention function
    """
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    
    # 1. Compute Scores (N x N memory usage)
    scores = Q @ K.T * scale  # N × N
    
    # 2. Safe Softmax
    m = np.max(scores, axis=1, keepdims=True)  # N × 1
    exp_scores = np.exp(scores - m)  # N × N
    d_sum = np.sum(exp_scores, axis=1, keepdims=True)  # N × 1
    probs = exp_scores / d_sum  # N × N
    
    # 3. Compute Output
    output = probs @ V  # N × d
    
    return output

def online_softmax_attention(Q, K, V):
    """
    2. Self-Attention with Online Softmax (Row-wise processing).
    Based on attn.py's online_softmax_attention function
    """
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    
    O = np.zeros((N, d), dtype=np.float32)
    m = np.full((N, 1), -np.inf, dtype=np.float32)  # row-wise max
    d_sum = np.zeros((N, 1), dtype=np.float32)  # denominator
    
    for j in range(N):
        k_j = K[j:j+1, :]  # 1 × d
        v_j = V[j:j+1, :]  # 1 × d
        
        x_j = Q @ k_j.T * scale  # N × 1
        
        m_prev = m
        m_new = np.maximum(m_prev, x_j)  # N × 1
        
        correction = np.exp(m_prev - m_new)  # N × 1
        d_sum = d_sum * correction + np.exp(x_j - m_new)  # N × 1
        O = O * correction + np.exp(x_j - m_new) * v_j  # N × d
        
        m = m_new

    return O / d_sum

def flash_attention_tiled(Q, K, V, block_size=32):
    """
    3. FlashAttention (Tiled Simulation) - Corrected
    Based on attn.py's flash_attention_tiled function
    """
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    
    O = np.zeros((N, d), dtype=np.float32)
    l = np.zeros((N, 1), dtype=np.float32)  # denominator
    m = np.full((N, 1), -np.inf, dtype=np.float32)  # row-wise max

    # Outer loop (Load Q blocks)
    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        q_block = Q[i:i_end, :]  # Br × d
        
        # Initialize block statistics
        m_block = m[i:i_end, :]  # Br × 1
        l_block = l[i:i_end, :]  # Br × 1
        o_block = O[i:i_end, :]  # Br × d
        
        # Inner loop (Load K, V blocks)
        for j in range(0, N, block_size):
            j_end = min(j + block_size, N)
            k_block = K[j:j_end, :]  # Bc × d
            v_block = V[j:j_end, :]  # Bc × d
            
            # 1. Compute Attention Scores: S_ij = Q_i @ K_j^T * scale
            s_ij = q_block @ k_block.T * scale  # Br × Bc
            
            # 2. Local max for this block
            m_ij = np.max(s_ij, axis=1, keepdims=True)  # Br × 1
            
            # 3. Update global running max
            m_new = np.maximum(m_block, m_ij)  # Br × 1
            
            # 4. Compute correction factors
            scale_old = np.exp(m_block - m_new)  # Br × 1
            
            # 5. Compute P_ij = exp(S_ij - m_new)
            p_ij = np.exp(s_ij - m_new)  # Br × Bc
            
            # 6. Update running sum (l)
            l_block = l_block * scale_old + np.sum(p_ij, axis=1, keepdims=True)  # Br × 1
            
            # 7. Update Output accumulator: O_new = O_old * scale_old + P_ij @ V_j
            o_block = o_block * scale_old + p_ij @ v_block  # Br × d
            
            # Update running max for next iteration
            m_block = m_new

        # Write updated blocks back
        O[i:i_end, :] = o_block
        l[i:i_end, :] = l_block
        m[i:i_end, :] = m_block
        
    return O / l

def save_matrix_to_file(matrix, filename):
    """Save matrix to text file in row-major order"""
    np.savetxt(filename, matrix.flatten(), fmt='%.8f')

def main():
    print("=== Flash Attention Python Reference Implementation ===")
    
    # Set parameters to match CUDA implementation
    N = 1024  # sequence length
    d = 1024  # embedding dimension
    seed = 42  # fixed seed for reproducibility
    
    print(f"Matrix dimensions: N={N}, d={d}")
    print(f"Total elements per matrix: {N*d:,}")
    print(f"Using random seed: {seed}")
    print()
    
    # Generate random matrices (same as CUDA)
    print("Generating random matrices Q, K, V...")
    Q = randomize_matrix_python((N, d), seed)
    K = randomize_matrix_python((N, d), seed + 1)  # Different seed for different matrices
    V = randomize_matrix_python((N, d), seed + 2)
    
    # Save input matrices
    print("Saving input matrices to files...")
    save_matrix_to_file(Q, 'python_input_Q.txt')
    save_matrix_to_file(K, 'python_input_K.txt')
    save_matrix_to_file(V, 'python_input_V.txt')
    print("  - python_input_Q.txt")
    print("  - python_input_K.txt")
    print("  - python_input_V.txt")
    
    # Compute all three reference outputs
    print("\nComputing reference outputs for three attention algorithms...")
    
    # 1. Standard Self-Attention
    print("1. Computing Standard Self-Attention...")
    start_time = time.time()
    O_std = standard_self_attention(Q, K, V)
    elapsed_std = time.time() - start_time
    print(f"   Completed in {elapsed_std:.3f} seconds")
    
    # 2. Online Softmax Attention
    print("2. Computing Online Softmax Attention...")
    start_time = time.time()
    O_online = online_softmax_attention(Q, K, V)
    elapsed_online = time.time() - start_time
    print(f"   Completed in {elapsed_online:.3f} seconds")
    
    # 3. Flash Attention (Tiled)
    print("3. Computing Flash Attention (Tiled)...")
    start_time = time.time()
    O_flash = flash_attention_tiled(Q, K, V, block_size=32)
    elapsed_flash = time.time() - start_time
    print(f"   Completed in {elapsed_flash:.3f} seconds")
    
    # Save reference outputs
    print("\nSaving reference outputs to files...")
    save_matrix_to_file(O_std, 'python_output_standard.txt')
    save_matrix_to_file(O_online, 'python_output_online.txt')
    save_matrix_to_file(O_flash, 'python_output_flash.txt')
    print("  - python_output_standard.txt (Standard Self-Attention)")
    print("  - python_output_online.txt (Online Softmax Attention)")
    print("  - python_output_flash.txt (Flash Attention Tiled)")
    
    # Print some statistics
    print("\n=== Output Statistics ===")
    print("Standard Self-Attention:")
    print(f"  Min value: {np.min(O_std):.6f}, Max value: {np.max(O_std):.6f}")
    print(f"  Mean: {np.mean(O_std):.6f}, Std: {np.std(O_std):.6f}")
    
    print("\nOnline Softmax Attention:")
    print(f"  Min value: {np.min(O_online):.6f}, Max value: {np.max(O_online):.6f}")
    print(f"  Mean: {np.mean(O_online):.6f}, Std: {np.std(O_online):.6f}")
    
    print("\nFlash Attention (Tiled):")
    print(f"  Min value: {np.min(O_flash):.6f}, Max value: {np.max(O_flash):.6f}")
    print(f"  Mean: {np.mean(O_flash):.6f}, Std: {np.std(O_flash):.6f}")
    
    # Verify correctness between different implementations
    print("\n=== Correctness Verification ===")
    
    # Compare Standard vs Online
    diff_std_online = np.max(np.abs(O_std - O_online))
    print(f"Standard vs Online: max difference = {diff_std_online:.2e} {'✓' if diff_std_online < 1e-5 else '✗'}")
    
    # Compare Standard vs Flash
    diff_std_flash = np.max(np.abs(O_std - O_flash))
    print(f"Standard vs Flash:  max difference = {diff_std_flash:.2e} {'✓' if diff_std_flash < 1e-5 else '✗'}")
    
    # Compare Online vs Flash
    diff_online_flash = np.max(np.abs(O_online - O_flash))
    print(f"Online vs Flash:    max difference = {diff_online_flash:.2e} {'✓' if diff_online_flash < 1e-5 else '✗'}")
    
    # Verify softmax property (sum of attention weights should be 1 for each query)
    print("\n=== Softmax Property Verification ===")
    test_queries = [0, 100, 500]
    scale = 1.0 / np.sqrt(d)
    for q_idx in test_queries:
        # Compute attention weights for this query (with scale factor)
        Q_single = Q[q_idx:q_idx+1, :]  # 1 × d
        S = Q_single @ K.T * scale  # 1 × N
        P = np.exp(S - np.max(S)) / np.sum(np.exp(S - np.max(S)))
        
        # The output should be P @ V
        O_single_ref = P @ V
        
        # Compare with all three implementations
        diff_std = np.max(np.abs(O_single_ref - O_std[q_idx:q_idx+1, :]))
        diff_online = np.max(np.abs(O_single_ref - O_online[q_idx:q_idx+1, :]))
        diff_flash = np.max(np.abs(O_single_ref - O_flash[q_idx:q_idx+1, :]))
        
        print(f"Query {q_idx}:")
        print(f"  vs Standard: {diff_std:.2e} {'✓' if diff_std < 1e-5 else '✗'}")
        print(f"  vs Online:   {diff_online:.2e} {'✓' if diff_online < 1e-5 else '✗'}")
        print(f"  vs Flash:    {diff_flash:.2e} {'✓' if diff_flash < 1e-5 else '✗'}")
    
    print("\n=== Summary ===")
    print("Python reference implementations completed successfully.")
    print("Input and output files have been saved for comparison with CUDA kernels.")
    print("\nTo compare with CUDA kernel output:")
    print("  python3 compare.py python_output_standard.txt kernelX_output.txt")
    print("  python3 compare.py python_output_online.txt kernelX_output.txt")
    print("  python3 compare.py python_output_flash.txt kernelX_output.txt")
    print("\nNote: Different CUDA kernels may implement different algorithms.")
    print("      Compare with the appropriate reference implementation.")

if __name__ == "__main__":
    main()
