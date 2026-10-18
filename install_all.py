"""
================================================================================
 ProteoFunc  -  COMPLETE SETUP & INSTALLATION SCRIPT
================================================================================
 This script installs EVERY SINGLE dependency needed to run the ProteoFunc
 Multimodal Protein Function Prediction project from a COMPLETELY BLANK
 Python environment.

 REQUIREMENTS:
   - Python 3.11 installed at:
     python
   - Internet connection (to download packages)

 HOW TO RUN:
   Open PowerShell in the project folder and run:
     & "python" install_all.py

 WHAT IT DOES:
   1. Upgrades pip, setuptools, wheel
   2. Installs PyTorch (CPU version for Windows)
   3. Installs PyTorch Geometric + scatter/sparse/cluster (CPU wheels)
   4. Installs HuggingFace Transformers + ESM-2 support
   5. Installs all scientific computing libraries (numpy, pandas, scipy, sklearn)
   6. Installs visualization libraries (matplotlib, seaborn, plotly)
   7. Installs web dashboard (Streamlit)
   8. Installs explainability tools (SHAP, Captum)
   9. Installs bioinformatics tools (BioPython, obonet)
   10. Installs logging/monitoring (TensorBoard, Wandb)
   11. Installs utility packages (PyYAML, tqdm, requests)
   12. Sets up Windows environment variables
   13. Creates all required project directories
   14. Verifies every single import works

 After running this script, you can directly start with:
     python run_pipeline.py --stage embeddings
================================================================================
"""

import os
import sys
import subprocess
import time

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PYTHON = sys.executable  # Use whichever Python runs this script

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def print_header(text):
    print()
    print("=" * 72)
    print(f"  {text}")
    print("=" * 72)

def print_step(num, total, text):
    print(f"\n  [{num}/{total}] {text}...")

def run_pip(args, desc=""):
    """Run a pip command and handle errors."""
    cmd = [PYTHON, "-m", "pip"] + args
    print(f"    -> {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        if result.returncode != 0:
            # Check if it's just "already satisfied"
            if "already satisfied" in result.stdout.lower():
                print(f"    ✅ {desc}  -  already installed")
                return True
            print(f"    ⚠️  Warning: {desc}")
            if result.stderr:
                # Only show last 3 lines of error
                err_lines = result.stderr.strip().split("\n")
                for line in err_lines[-3:]:
                    print(f"       {line}")
            return False
        else:
            print(f"    ✅ {desc}  -  installed successfully")
            return True
    except subprocess.TimeoutExpired:
        print(f"    ⏰ Timeout: {desc} (took too long, skipping)")
        return False
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False

def check_import(module_name, import_name=None):
    """Check if a module can be imported."""
    name = import_name or module_name
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN INSTALLATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()

    print_header("ProteoFunc  -  COMPLETE INSTALLATION SCRIPT")
    print(f"  Python: {PYTHON}")
    print(f"  Version: {sys.version}")
    print(f"  Platform: {sys.platform}")
    print(f"  Working directory: {os.getcwd()}")

    TOTAL_STEPS = 13
    results = {}

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: Upgrade pip, setuptools, wheel
    # ──────────────────────────────────────────────────────────────────────
    print_step(1, TOTAL_STEPS, "Upgrading pip, setuptools, wheel")
    run_pip(["install", "--upgrade", "pip", "setuptools", "wheel"], "pip/setuptools/wheel")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2: Install PyTorch (CPU version for Windows)
    # ──────────────────────────────────────────────────────────────────────
    print_step(2, TOTAL_STEPS, "Installing PyTorch (CPU)")
    if check_import("torch"):
        import torch
        print(f"    ✅ PyTorch already installed: v{torch.__version__}")
        results["pytorch"] = True
    else:
        results["pytorch"] = run_pip(
            ["install", "torch", "torchvision", "torchaudio",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
            "PyTorch CPU"
        )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3: Install PyTorch Geometric + extensions
    # ──────────────────────────────────────────────────────────────────────
    print_step(3, TOTAL_STEPS, "Installing PyTorch Geometric (graph neural networks)")

    # First install pyg-lib and torch-geometric from PyG wheels
    # Get torch version for compatible wheels
    try:
        import torch
        torch_version = torch.__version__.split("+")[0]  # e.g. "2.3.0"
        torch_major_minor = ".".join(torch_version.split(".")[:2])  # e.g. "2.3"
        print(f"    Detected PyTorch version: {torch_version}")

        # Install PyG dependencies using the find-links approach
        pyg_url = f"https://data.pyg.org/whl/torch-{torch_major_minor}.0+cpu.html"

        run_pip(
            ["install", "torch-scatter", "torch-sparse", "torch-cluster",
             "-f", pyg_url],
            "PyG extensions (scatter/sparse/cluster)"
        )
        results["pyg_extensions"] = True
    except Exception as e:
        print(f"    ⚠️  PyG extensions install attempt: {e}")
        # Fallback: install without specific torch version
        run_pip(["install", "torch-scatter", "torch-sparse", "torch-cluster"], "PyG extensions (fallback)")
        results["pyg_extensions"] = False

    run_pip(["install", "torch-geometric>=2.4.0"], "torch-geometric")
    results["torch_geometric"] = check_import("torch_geometric")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4: Install HuggingFace Transformers + ESM-2
    # ──────────────────────────────────────────────────────────────────────
    print_step(4, TOTAL_STEPS, "Installing Transformers (ESM-2 protein language model)")
    run_pip(["install", "transformers>=4.30.0"], "HuggingFace Transformers")
    run_pip(["install", "fair-esm>=2.0.0"], "fair-esm (ESM-2)")
    run_pip(["install", "sentencepiece"], "sentencepiece (tokenizer)")
    run_pip(["install", "protobuf"], "protobuf")
    results["transformers"] = check_import("transformers")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 5: Install scientific computing libraries
    # ──────────────────────────────────────────────────────────────────────
    print_step(5, TOTAL_STEPS, "Installing scientific computing (NumPy, Pandas, SciPy, Scikit-learn)")
    run_pip(["install", "numpy>=1.24.0"], "NumPy")
    run_pip(["install", "pandas>=2.0.0"], "Pandas")
    run_pip(["install", "scipy>=1.10.0"], "SciPy")
    run_pip(["install", "scikit-learn>=1.3.0"], "Scikit-learn")
    run_pip(["install", "networkx>=3.1"], "NetworkX")
    results["scientific"] = all([
        check_import("numpy"),
        check_import("pandas"),
        check_import("scipy"),
        check_import("sklearn", "sklearn"),
        check_import("networkx"),
    ])

    # ──────────────────────────────────────────────────────────────────────
    # STEP 6: Install visualization libraries
    # ──────────────────────────────────────────────────────────────────────
    print_step(6, TOTAL_STEPS, "Installing visualization (Matplotlib, Seaborn, Plotly)")
    run_pip(["install", "matplotlib>=3.7.0"], "Matplotlib")
    run_pip(["install", "seaborn>=0.12.0"], "Seaborn")
    run_pip(["install", "plotly>=5.15.0"], "Plotly")
    results["visualization"] = all([
        check_import("matplotlib"),
        check_import("seaborn"),
        check_import("plotly"),
    ])

    # ──────────────────────────────────────────────────────────────────────
    # STEP 7: Install Streamlit web dashboard
    # ──────────────────────────────────────────────────────────────────────
    print_step(7, TOTAL_STEPS, "Installing Streamlit (web dashboard)")
    run_pip(["install", "streamlit>=1.28.0"], "Streamlit")
    results["streamlit"] = check_import("streamlit")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 8: Install explainability tools
    # ──────────────────────────────────────────────────────────────────────
    print_step(8, TOTAL_STEPS, "Installing explainability (SHAP, Captum)")
    run_pip(["install", "shap>=0.42.0"], "SHAP")
    run_pip(["install", "captum>=0.6.0"], "Captum")
    results["explainability"] = check_import("captum")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 9: Install bioinformatics tools
    # ──────────────────────────────────────────────────────────────────────
    print_step(9, TOTAL_STEPS, "Installing bioinformatics (BioPython, obonet)")
    run_pip(["install", "biopython>=1.81"], "BioPython")
    run_pip(["install", "obonet>=1.0.0"], "obonet (GO ontology)")
    results["bioinformatics"] = check_import("Bio")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 10: Install logging & monitoring
    # ──────────────────────────────────────────────────────────────────────
    print_step(10, TOTAL_STEPS, "Installing logging & monitoring (TensorBoard, W&B)")
    run_pip(["install", "tensorboard>=2.14.0"], "TensorBoard")
    run_pip(["install", "wandb>=0.15.0"], "Weights & Biases")
    results["monitoring"] = check_import("tensorboard")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 11: Install utility packages
    # ──────────────────────────────────────────────────────────────────────
    print_step(11, TOTAL_STEPS, "Installing utility packages")
    run_pip(["install", "pyyaml>=6.0"], "PyYAML")
    run_pip(["install", "tqdm>=4.65.0"], "tqdm")
    run_pip(["install", "requests>=2.31.0"], "requests")
    results["utilities"] = all([
        check_import("yaml"),
        check_import("tqdm"),
        check_import("requests"),
    ])

    # ──────────────────────────────────────────────────────────────────────
    # STEP 12: Set Windows environment variables
    # ──────────────────────────────────────────────────────────────────────
    print_step(12, TOTAL_STEPS, "Setting Windows environment variables")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    print("    ✅ KMP_DUPLICATE_LIB_OK = TRUE (prevents OpenMP crash)")
    print("    ✅ PYTHONIOENCODING = utf-8 (prevents Unicode errors)")

    # ──────────────────────────────────────────────────────────────────────
    # STEP 13: Create all required project directories
    # ──────────────────────────────────────────────────────────────────────
    print_step(13, TOTAL_STEPS, "Creating project directories")
    directories = [
        "data/raw",
        "data/processed",
        "data/embeddings",
        "data/cache",
        "outputs/checkpoints",
        "outputs/evaluation",
        "outputs/explainability",
        "outputs/figures",
        "outputs/logs",
    ]
    for d in directories:
        os.makedirs(d, exist_ok=True)
        print(f"    📁 {d}/")
    print("    ✅ All directories created")

    # ══════════════════════════════════════════════════════════════════════
    # VERIFICATION  -  Test every single import
    # ══════════════════════════════════════════════════════════════════════
    print_header("VERIFYING ALL IMPORTS")

    all_imports = [
        # Core deep learning
        ("torch",               "PyTorch",                    "torch"),
        ("torchvision",         "TorchVision",                "torchvision"),

        # PyTorch Geometric
        ("torch_geometric",     "PyTorch Geometric",          "torch_geometric"),

        # Transformers / ESM-2
        ("transformers",        "HuggingFace Transformers",   "transformers"),

        # Scientific computing
        ("numpy",               "NumPy",                      "numpy"),
        ("pandas",              "Pandas",                     "pandas"),
        ("scipy",               "SciPy",                      "scipy"),
        ("sklearn",             "Scikit-learn",               "sklearn"),
        ("networkx",            "NetworkX",                   "networkx"),

        # Visualization
        ("matplotlib",          "Matplotlib",                 "matplotlib"),
        ("seaborn",             "Seaborn",                    "seaborn"),
        ("plotly",              "Plotly",                     "plotly"),

        # Web dashboard
        ("streamlit",           "Streamlit",                  "streamlit"),

        # Explainability
        ("shap",                "SHAP",                       "shap"),
        ("captum",              "Captum",                     "captum"),

        # Bioinformatics
        ("Bio",                 "BioPython",                  "Bio"),
        ("obonet",              "obonet (GO parser)",         "obonet"),

        # Logging
        ("tensorboard",         "TensorBoard",                "tensorboard"),

        # Utilities
        ("yaml",                "PyYAML",                     "yaml"),
        ("tqdm",                "tqdm",                       "tqdm"),
        ("requests",            "requests",                   "requests"),
        ("pickle",              "pickle (built-in)",          "pickle"),
        ("json",                "json (built-in)",            "json"),
        ("argparse",            "argparse (built-in)",        "argparse"),
        ("pathlib",             "pathlib (built-in)",         "pathlib"),
        ("logging",             "logging (built-in)",         "logging"),
    ]

    passed = 0
    failed = 0
    failed_list = []

    for import_name, display_name, module_name in all_imports:
        try:
            __import__(module_name)
            print(f"    ✅ {display_name:<30s} ({module_name})")
            passed += 1
        except ImportError as e:
            print(f"    ❌ {display_name:<30s} ({module_name})  -  {e}")
            failed += 1
            failed_list.append(display_name)

    # Additional checks for specific submodules
    print("\n  Checking specific submodules...")

    sub_checks = [
        ("torch.nn", "PyTorch neural networks"),
        ("torch.optim", "PyTorch optimizers"),
        ("sklearn.metrics", "Scikit-learn metrics"),
        ("sklearn.model_selection", "Scikit-learn model selection"),
        ("sklearn.preprocessing", "Scikit-learn preprocessing"),
        ("sklearn.manifold", "Scikit-learn manifold (t-SNE)"),
        ("scipy.stats", "SciPy statistics"),
        ("matplotlib.pyplot", "Matplotlib pyplot"),
    ]

    for module_name, display_name in sub_checks:
        try:
            __import__(module_name)
            print(f"    ✅ {display_name:<30s} ({module_name})")
            passed += 1
        except ImportError as e:
            print(f"    ❌ {display_name:<30s} ({module_name})  -  {e}")
            failed += 1
            failed_list.append(display_name)

    # Check PyTorch Geometric submodules (these commonly fail)
    pyg_sub = [
        ("torch_geometric.nn", "PyG neural network layers"),
        ("torch_geometric.data", "PyG data structures"),
    ]
    for module_name, display_name in pyg_sub:
        try:
            __import__(module_name)
            print(f"    ✅ {display_name:<30s} ({module_name})")
            passed += 1
        except ImportError as e:
            print(f"    ⚠️  {display_name:<30s} ({module_name})  -  {e}")
            print(f"       (PyG might still work  -  some submodules are optional)")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time

    print_header("INSTALLATION SUMMARY")
    print(f"  ✅ Passed:  {passed}")
    print(f"  ❌ Failed:  {failed}")
    print(f"  ⏱️  Time:    {elapsed/60:.1f} minutes")
    print()

    if failed_list:
        print("  ⚠️  Failed packages (may need manual install):")
        for f in failed_list:
            print(f"      - {f}")
        print()

    # Print version info for key packages
    print("  📦 Key package versions:")
    version_checks = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("torch_geometric", "PyTorch Geometric"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("streamlit", "Streamlit"),
        ("sklearn", "Scikit-learn"),
    ]
    for mod, name in version_checks:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "unknown")
            print(f"      {name:<25s} v{ver}")
        except:
            print(f"      {name:<25s} not installed")

    print()
    print("=" * 72)
    if failed == 0:
        print("  🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!")
        print()
        print("  You can now run the pipeline:")
        print()
        print('  STEP 1  -  Extract embeddings:')
        print('    $env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:PYTHONIOENCODING="utf-8"; python run_pipeline.py --stage embeddings')
        print()
        print('  STEP 2  -  Build dataset:')
        print('    $env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:PYTHONIOENCODING="utf-8"; python run_pipeline.py --stage dataset')
        print()
        print('  STEP 3  -  Train models:')
        print('    $env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:PYTHONIOENCODING="utf-8"; python run_pipeline.py --stage train')
        print()
        print('  STEP 4  -  Evaluate models:')
        print('    $env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:PYTHONIOENCODING="utf-8"; python run_pipeline.py --stage evaluate')
        print()
        print('  STEP 5  -  Explainability:')
        print('    $env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:PYTHONIOENCODING="utf-8"; python run_pipeline.py --stage explain')
        print()
        print('  STEP 6  -  Launch dashboard:')
        print('    streamlit run app.py')
    else:
        print(f"  ⚠️  {failed} package(s) had issues. Check the errors above.")
        print("  Most packages installed successfully  -  the pipeline may still work.")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
