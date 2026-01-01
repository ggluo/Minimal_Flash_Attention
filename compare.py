#!/usr/bin/env python3
import sys
import numpy as np

def compare_files(file1, file2, tolerance=1e-5):
    """
    比较两个包含浮点数的文本文件
    
    参数:
        file1: 第一个文件路径
        file2: 第二个文件路径
        tolerance: 容差阈值
    """
    print(f"=== Comparing Files ===")
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")
    print(f"Tolerance: {tolerance}")
    print()
    
    # 读取文件
    try:
        data1 = np.loadtxt(file1)
        data2 = np.loadtxt(file2)
    except Exception as e:
        print(f"Error reading files: {e}")
        return False
    
    # 检查数据形状
    print(f"File 1 shape: {data1.shape}")
    print(f"File 2 shape: {data2.shape}")
    
    if data1.shape != data2.shape:
        print("❌ Error: Files have different shapes!")
        print(f"  File 1 has {len(data1)} elements")
        print(f"  File 2 has {len(data2)} elements")
        
        # 如果形状不同，比较前 min(n1, n2) 个元素
        n = min(len(data1), len(data2))
        print(f"\nComparing first {n} elements only:")
        data1 = data1[:n]
        data2 = data2[:n]
    
    n = len(data1)
    
    # 计算误差
    abs_diff = np.abs(data1 - data2)
    rel_diff = np.abs(data1 - data2) / (np.abs(data1) + 1e-10)  # 避免除以0
    
    # 统计信息
    max_abs_error = np.max(abs_diff)
    mean_abs_error = np.mean(abs_diff)
    max_rel_error = np.max(rel_diff)
    mean_rel_error = np.mean(rel_diff)
    
    # 找出误差最大的位置
    max_abs_idx = np.argmax(abs_diff)
    max_rel_idx = np.argmax(rel_diff)
    
    print(f"\n=== Error Statistics ===")
    print(f"Maximum absolute error: {max_abs_error:.2e} at index {max_abs_idx}")
    print(f"Mean absolute error:    {mean_abs_error:.2e}")
    print(f"Maximum relative error: {max_rel_error:.2e} at index {max_rel_idx}")
    print(f"Mean relative error:    {mean_rel_error:.2e}")
    
    # 检查是否在容差范围内
    all_within_tolerance = np.all(abs_diff <= tolerance)
    
    print(f"\n=== Validation Result ===")
    if all_within_tolerance:
        print(f"✅ PASS: All {n} values are within tolerance ({tolerance})")
        return True
    else:
        num_errors = np.sum(abs_diff > tolerance)
        print(f"❌ FAIL: {num_errors} out of {n} values exceed tolerance ({tolerance})")
        
        # 显示误差最大的几个值
        print(f"\n=== Top 5 Largest Differences ===")
        print("Index | Value1      | Value2      | Abs Diff    | Rel Diff")
        print("------|-------------|-------------|-------------|-------------")
        
        # 获取误差最大的5个索引
        top_indices = np.argsort(abs_diff)[-5:][::-1]
        for idx in top_indices:
            v1 = data1[idx]
            v2 = data2[idx]
            abs_err = abs_diff[idx]
            rel_err = rel_diff[idx]
            print(f"{idx:5d} | {v1:11.6f} | {v2:11.6f} | {abs_err:11.2e} | {rel_err:11.2e}")
        
        return False

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python compare.py <file1> <file2> [tolerance]")
        print("Example: python compare.py python_output_kernel2.txt kernel0_output.txt")
        print("Example: python compare.py python_output_kernel2.txt kernel0_output.txt 1e-4")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    # 设置默认容差
    tolerance = 1e-5
    if len(sys.argv) == 4:
        try:
            tolerance = float(sys.argv[3])
        except ValueError:
            print(f"Error: Invalid tolerance value '{sys.argv[3]}'. Using default {tolerance}")
    
    success = compare_files(file1, file2, tolerance)
    
    # 返回适当的退出代码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
