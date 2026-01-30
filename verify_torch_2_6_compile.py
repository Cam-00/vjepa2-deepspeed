import torch
import warnings


def verify_2_6_features():
    """Verify PyTorch 2.6 features and environment"""

    print("=== PyTorch 2.6 Feature Verification ===")

    # Basic Information
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"SM Count: {torch.cuda.get_device_properties(0).multi_processor_count}")

    # Check Compilation Features
    print(f"\nCompilation Features:")
    print(f"  torch.compile available: {hasattr(torch, 'compile')}")

    # Test small model compilation
    try:
        test_model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        ).cuda()

        test_input = torch.randn(32, 128).cuda()
        # Using model.compile (standard in PyTorch 2.x+)
        test_model.compile(mode="max-autotune")
        output = test_model(test_input)

        print("  ✅ Model compilation test passed")

    except Exception as e:
        print(f"  ❌ Model compilation test failed: {e}")


def setup_ultimate_performance():
    """Complete performance optimization configuration"""

    print(f"PyTorch Version: {torch.__version__}")

    # 1. Floating point precision settings (Resolves TensorFloat32 warnings)
    torch.set_float32_matmul_precision('high')  # Options: 'high' or 'medium'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 2. PyTorch Inductor Core Optimizations
    torch._inductor.config.max_autotune = True
    torch._inductor.config.max_autotune_gemm = True  # Bypasses SM count warnings
    torch._inductor.config.epilogue_fusion = True
    torch._inductor.config.coordinate_descent_tuning = True

    # 3. Triton Optimizations
    torch._inductor.config.triton.cudagraphs = True

    # 4. PyTorch 2.6 Specific Configurations
    if hasattr(torch._inductor.config, 'fx_graph_cache'):
        torch._inductor.config.fx_graph_cache = True
        print("fx_graph_cache activated")

    if hasattr(torch._inductor.config, 'pattern_matcher'):
        torch._inductor.config.pattern_matcher = True
        print("pattern_matcher activated")

    # 5. Environment Variables
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"

    # 6. Threading Settings
    torch.set_num_threads(4)

    # 7. Warning Suppression
    # warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gemm mode.*")
    # warnings.filterwarnings("ignore", message=".*TensorFloat32 tensor cores.*")

    print("✅ PyTorch 2.6 performance configuration completed successfully")


def main():
    setup_ultimate_performance()
    verify_2_6_features()
    return


if __name__ == "__main__":
    main()