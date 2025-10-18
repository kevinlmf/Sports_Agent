# Sports Injury Risk Prediction - Demo Scripts

This directory contains comprehensive demonstration scripts showcasing all seven dimensions of the Sports Injury Risk Prediction system.

## 📋 Overview

The complete demo system covers:

1. **Algorithm Coverage** - Traditional, DL, and Multimodal methods
2. **Architecture Integrity** - Unified interface across all components
3. **Distributed Training** - Multi-GPU support with DeepSpeed/Ray
4. **MLOps Support** - Experiment tracking and profiling
5. **API/Deployment** - Production-ready FastAPI service
6. **Enterprise Features** - Interpretability and risk management
7. **Research Extensions** - Future roadmap and capabilities

## 🚀 Quick Start

### Run Complete Demo

```bash
# Run all seven dimensions
./scripts/run_complete_demo.sh
```

This will systematically test and demonstrate all system capabilities.

### Run Individual Demos

```bash
# MLOps: Experiment tracking and profiling
python scripts/demo_mlops.py

# API: Deployment architecture
python scripts/demo_api.py

# Enterprise: Interpretability and validation
python scripts/demo_enterprise.py
```

## 📁 Script Descriptions

### `run_complete_demo.sh`
Master script that runs all seven dimension tests.

**Features:**
- Color-coded output for easy reading
- Progressive execution with status updates
- Checks system capabilities (GPU, dependencies)
- Generates comprehensive summary report

**Output:**
- System status summary table
- Performance benchmarks
- Test coverage report
- Next steps guidance

### `demo_mlops.py`
Demonstrates MLOps infrastructure and practices.

**Components:**
- **MLflow Integration**: Experiment tracking, model registry
- **Weights & Biases**: Real-time visualization, hyperparameter tuning
- **PyTorch Profiler**: Performance analysis, bottleneck detection
- **Model Versioning**: Checkpoint management strategy
- **Monitoring**: Production metrics and alerting

**Usage:**
```python
python scripts/demo_mlops.py
```

### `demo_api.py`
Showcases production deployment architecture.

**Components:**
- **FastAPI Service**: REST API endpoints
- **Request/Response Examples**: Multimodal inputs
- **Docker Deployment**: Containerization strategy
- **Kubernetes**: Orchestration and auto-scaling
- **Model Optimization**: ONNX, TorchScript, quantization

**Usage:**
```python
python scripts/demo_api.py
```

### `demo_enterprise.py`
Demonstrates enterprise-grade features.

**Components:**
- **SHAP Explainability**: Feature importance analysis
- **Grad-CAM**: Visual interpretability for images
- **Attention Visualization**: Cross-modal attention patterns
- **Uncertainty Quantification**: Confidence intervals
- **Model Calibration**: Reliability assessment
- **Fairness Analysis**: Bias detection and mitigation
- **Clinical Validation**: Medical metrics and thresholds
- **Model Governance**: Documentation and compliance

**Usage:**
```python
python scripts/demo_enterprise.py
```

## 🎯 What Each Demo Shows

### 1️⃣ Algorithm Coverage Demo

**Traditional Methods:**
- ✅ CHIME Model
- ✅ Logistic Regression
- ✅ Random Forest
- ✅ XGBoost

**Deep Learning Methods:**
- ✅ TransCHIME (Transformer-based)
- ✅ LSTM
- ✅ GRU
- ✅ Transformer

**Multimodal Models:**
- ✅ Vision-Language Model (CLIP + BERT)
- ✅ LoRA Adapters

### 2️⃣ Architecture Integrity Demo

Shows unified interface across:
- Data loading (`SportInjuryDataset`)
- Feature engineering (`FeatureEngineer`)
- Model interface (all models share common API)
- Training pipeline (`Trainer`)
- Evaluation metrics (`compute_auc_roc`, `calibration`)

### 3️⃣ Distributed Training Demo

**Strategies:**
- Data Parallel (DistributedDataParallel)
- Model Parallel (DeepSpeed ZeRO)
- Pipeline Parallel
- Hybrid Parallel

**Optimizations:**
- Gradient accumulation
- Mixed precision (FP16/BF16)
- Gradient checkpointing
- CPU offloading

### 4️⃣ MLOps Support Demo

**Experiment Tracking:**
- MLflow: Parameters, metrics, artifacts
- W&B: Real-time dashboards, sweeps

**Profiling:**
- PyTorch Profiler: CPU/GPU utilization
- Memory profiling
- Bottleneck detection

**Model Management:**
- Version control
- Checkpoint strategies
- Model registry

### 5️⃣ API/Deployment Demo

**API Endpoints:**
```
POST /predict          - Single prediction
POST /batch_predict    - Batch processing
GET  /health          - Health check
GET  /model_info      - Model metadata
```

**Deployment:**
- Docker containerization
- Kubernetes orchestration
- Horizontal auto-scaling
- ONNX optimization

### 6️⃣ Enterprise Features Demo

**Interpretability:**
- SHAP feature importance
- Grad-CAM visual heatmaps
- Attention weight visualization

**Validation:**
- Uncertainty quantification
- Model calibration (ECE, Brier score)
- Fairness metrics (demographic parity)
- Clinical metrics (sensitivity, specificity)

**Governance:**
- Model cards
- Ethical considerations
- Compliance documentation

### 7️⃣ Research Extensions Demo

**Implemented:**
- ✅ Vision-Language Models
- ✅ LoRA adapters
- ✅ TransCHIME architecture
- ✅ Multimodal fusion

**In Progress:**
- 🔄 Video sequence analysis
- 🔄 Adapter fusion strategies
- 🔄 Sparse attention

**Planned:**
- 🚀 Continual learning
- 🚀 Causal inference
- 🚀 Federated learning

## 📊 Expected Output

### System Status Summary
```
┌─────────────────────────────────────────────────────────────────────┐
│                      SYSTEM STATUS SUMMARY                          │
├─────────────────────────────────────────────────────────────────────┤
│ Dimension                 │ Status │ Key Component                  │
├───────────────────────────┼────────┼────────────────────────────────┤
│ 1️⃣  Algorithm Coverage    │   ✅   │ Traditional + DL + Multimodal  │
│ 2️⃣  Architecture          │   ✅   │ Unified Interface              │
│ 3️⃣  Distributed Training  │   ⚙️   │ Multi-GPU Ready                │
│ 4️⃣  MLOps Support         │   ✅   │ MLflow + W&B                   │
│ 5️⃣  API / Deployment      │   🔄   │ FastAPI + Docker               │
│ 6️⃣  Enterprise Features   │   ✅   │ SHAP + Grad-CAM                │
│ 7️⃣  Research Extensions   │   🚀   │ VLM + LoRA + TransCHIME        │
└─────────────────────────────────────────────────────────────────────┘
```

### Performance Benchmarks
```
┌───────────────────┬────────┬─────────┬──────────┬─────────┐
│ Model             │ Params │ AUC-ROC │ Inference│ Memory  │
├───────────────────┼────────┼─────────┼──────────┼─────────┤
│ XGBoost (Baseline)│   2M   │  0.85   │   3ms    │  0.05GB │
│ TransCHIME        │  10M   │  0.89   │  15ms    │  1.0GB  │
│ VLM (Full)        │ 130M   │  0.93   │  50ms    │  2.0GB  │
│ VLM + LoRA        │  11M   │  0.92   │  40ms    │  1.5GB  │
└───────────────────┴────────┴─────────┴──────────┴─────────┘
```

## 🔧 Prerequisites

### Required Dependencies
```bash
# Core ML
pip install torch torchvision transformers

# Multimodal
pip install peft bitsandbytes

# Interpretability
pip install shap captum

# Data processing
pip install pandas numpy scikit-learn

# Optional: MLOps
pip install mlflow wandb
```

### System Requirements
- Python 3.8+
- 8GB+ RAM (16GB+ recommended for multimodal models)
- GPU optional but recommended (CUDA 11.8+ if using GPU)

## 📝 Usage Examples

### Basic Demo Run
```bash
# Quick test - individual components
python scripts/demo_mlops.py
python scripts/demo_api.py
python scripts/demo_enterprise.py

# Full system demo
./scripts/run_complete_demo.sh
```

### With Dependencies Check
```bash
# Install dependencies first
pip install -r requirements.txt

# Run demo
./scripts/run_complete_demo.sh
```

### Customize Output
```bash
# Run specific dimensions only
# Edit run_complete_demo.sh and comment out sections you want to skip

# Example: Skip distributed training demo
# Just comment out the "Dimension 3" section
```

## 🐛 Troubleshooting

### Script Permission Error
```bash
chmod +x scripts/run_complete_demo.sh
chmod +x scripts/demo_*.py
```

### Missing Dependencies
```bash
pip install torch transformers peft shap captum
```

### GPU Not Detected
- The demo will automatically fall back to CPU mode
- Multi-GPU features will be skipped with appropriate messages

### Import Errors
- Ensure you're running from the project root directory
- Check that `src/` is in your Python path

## 📚 Related Documentation

- **README.md** - Main project overview
- **docs/MULTIMODAL_SYSTEM_ARCHITECTURE.md** - Complete architecture
- **docs/DL_EXPANSION_PLAN.md** - Research roadmap
- **PROJECT_STRUCTURE.md** - Directory structure
- **examples/chime_example.py** - CHIME usage example
- **examples/quick_start.py** - Quick start guide

## 🎓 Learning Path

**Beginners:**
1. Run `./scripts/run_complete_demo.sh` to see overview
2. Study `demo_enterprise.py` for interpretability basics
3. Explore `examples/chime_example.py` for hands-on training

**Intermediate:**
1. Deep dive into `demo_mlops.py` for experiment tracking
2. Review `demo_api.py` for deployment patterns
3. Study docs/MULTIMODAL_SYSTEM_ARCHITECTURE.md

**Advanced:**
1. Implement custom distributed training strategies
2. Extend multimodal fusion architectures
3. Contribute to research extensions roadmap

## 🤝 Contributing

If you add new features, please update the corresponding demo script:
- New models → Update Algorithm Coverage section
- New interpretability → Update `demo_enterprise.py`
- New deployment → Update `demo_api.py`
- New MLOps tools → Update `demo_mlops.py`

## 📧 Support

For issues or questions:
1. Check this README and related docs
2. Review demo script output for hints
3. Inspect script source code for implementation details
4. Check project documentation in `docs/`

---

**Last Updated:** 2025-10-17
**Status:** ✅ All demos functional
**Version:** 1.0.0
