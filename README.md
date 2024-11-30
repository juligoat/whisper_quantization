# Whisper Quantization Project

This project explores different quantization approaches for the Whisper ASR model using HuggingFace Transformers.

## Project Structure
```
whisper_quantization/
├── notebooks/          # Jupyter notebooks
│   ├── 01_baseline_evaluation.ipynb
│   ├── 02_dynamic_quantization.ipynb
│   └── 03_static_quantization.ipynb
├── src/               # Helper functions
├── data/              # Dataset files
├── models/            # Saved models
└── results/           # Evaluation results
```

## Setup
1. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Jupyter:
   ```bash
   jupyter notebook
   ```

## Project Steps
1. Baseline evaluation of Whisper-small
2. Dynamic quantization implementation
3. Static quantization implementation
4. Performance comparison and analysis
