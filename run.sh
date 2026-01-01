#!/bin/bash

echo "=== Flash Attention Kernel Testing Script ==="
echo ""

# Configuration
VERBOSE=true
TOLERANCE=1e-4  # Tolerance for comparison
REFERENCE_FILE="python_output_standard.txt"  # Only compare with standard output

# Step 1: Run Python reference implementation
echo "Step 1: Running Python reference implementation..."
echo ""

if ! python3 flash.py; then
    echo "❌ Error: Python reference flash.py implementation failed!"
    exit 1
fi

# Step 2: Build CUDA program
echo "Step 2: Building CUDA program..."
echo ""

mkdir -p build
cd build

echo "Running CMake..."
if ! cmake .. > cmake.log 2>&1; then
    echo "❌ CMake failed. Check build/cmake.log for details."
    exit 1
fi

echo "Running make..."
if ! make -j4 > make.log 2>&1; then
    echo "❌ Make failed. Check build/make.log for details."
    exit 1
fi

cd ..
echo "✅ Build completed successfully."
echo ""

# Step 3: Test each kernel and compare results
echo "Step 3: Testing each CUDA kernel (0-7) and comparing with Python reference..."
echo ""

# Array to track test results
declare -a TEST_RESULTS
ALL_PASSED=true

# Test each kernel
for ((i=1; i<=8; i++))
do
    echo "--- Testing Kernel ${i} ---"
    
    # Check if executable exists
    if [ ! -f "./flash" ]; then
        echo "❌ Error: Executable 'flash' not found!"
        exit 1
    fi
    
    # Run CUDA kernel
    echo "Running CUDA kernel ${i}..."
    if ! ./flash ${i} 2>&1; then
        echo "❌ Error: CUDA kernel ${i} execution failed!"
        TEST_RESULTS[$i]="FAIL (execution error)"
        ALL_PASSED=false
        echo ""
        continue
    fi
    
    # Check if output file was created
    OUTPUT_FILE="kernel${i}_output.txt"
    if [ ! -f "${OUTPUT_FILE}" ]; then
        echo "❌ Error: Output file ${OUTPUT_FILE} not found!"
        TEST_RESULTS[$i]="FAIL (no output)"
        ALL_PASSED=false
        echo ""
        continue
    fi
    
    echo "✅ Kernel ${i} executed successfully. Output saved to ${OUTPUT_FILE}"
    
    if $VERBOSE; then
        echo ""
        echo "Comparing CUDA kernel ${i} output with Python references..."
        
        # Compare with standard reference only
        if [ -f "${REFERENCE_FILE}" ]; then
            echo "  - Comparing with ${REFERENCE_FILE}..."
            
            # Run comparison
            if python3 compare.py "${REFERENCE_FILE}" "${OUTPUT_FILE}" "${TOLERANCE}" 2>/dev/null; then
                echo "    ✅ Matches ${REFERENCE_FILE} within tolerance ${TOLERANCE}"
                TEST_RESULTS[$i]="PASS (matches ${REFERENCE_FILE})"
            else
                # Try to get the max difference for reporting
                DIFF_OUTPUT=$(python3 compare.py "${REFERENCE_FILE}" "${OUTPUT_FILE}" "${TOLERANCE}" 2>&1 || true)
                MAX_DIFF=$(echo "$DIFF_OUTPUT" | grep -o "Maximum absolute error: [0-9.e+-]*" | cut -d' ' -f4 || echo "N/A")
                
                echo "    ❌ Does not match ${REFERENCE_FILE} (max diff: ${MAX_DIFF})"
                TEST_RESULTS[$i]="FAIL (max diff: ${MAX_DIFF})"
                ALL_PASSED=false
            fi
        else
            echo "❌ Error: Reference file ${REFERENCE_FILE} not found!"
            TEST_RESULTS[$i]="FAIL (no reference file)"
            ALL_PASSED=false
        fi
    else
        TEST_RESULTS[$i]="EXECUTED (not compared)"
    fi
    
    echo ""
done

# Step 4: Summary
echo "=== Test Summary ==="
echo ""

echo "Kernel Test Results:"
for ((i=1; i<=8; i++))
do
    if [[ "${TEST_RESULTS[$i]}" == PASS* ]]; then
        echo "  Kernel ${i}: ✅ ${TEST_RESULTS[$i]}"
    elif [[ "${TEST_RESULTS[$i]}" == FAIL* ]]; then
        echo "  Kernel ${i}: ❌ ${TEST_RESULTS[$i]}"
    else
        echo "  Kernel ${i}: ⚠️  ${TEST_RESULTS[$i]}"
    fi
done

echo ""
if $ALL_PASSED && $VERBOSE; then
    echo "✅ All kernels passed validation!"
else
    if $VERBOSE; then
        echo "❌ Some kernels failed validation."
    else
        echo "⚠️  Kernels executed but not compared (verbose mode disabled)."
    fi
fi