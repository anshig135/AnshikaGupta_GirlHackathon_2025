# RTL Depth Predictor

## Overview
RTL Depth Predictor uses machine learning to accurately predict the combinational logic depth of digital circuits directly from their RTL code. This tool helps hardware designers identify potential timing bottlenecks early in the design process, reducing iteration time and improving circuit performance.

## Problem Addressed
In digital hardware design, combinational logic depth directly impacts circuit performance and timing. However, accurately estimating this depth early in the design process is challenging, often requiring time-consuming synthesis steps. Our solution provides:

- Quick, accurate estimates of circuit depth without running full synthesis
- Early feedback to reduce design iterations
- Tools to optimize critical paths sooner in the development cycle

## Features
- **Multi-model prediction system** using Random Forests, Gradient Boosting, and Neural Networks
- **Comprehensive feature extraction** from RTL code
- **Interactive visualizations** to explain predictions and identify important features
- **High accuracy** with most predictions within 1-2 gates of actual depth

## Technical Approach

### Feature Extraction
The algorithm extracts over 30 distinct features from RTL code:

1. **Basic operator analysis** (counting logical operators like AND, OR, XOR)
2. **Control flow analysis** (if/else, case statements)
3. **Arithmetic complexity metrics** (add/subtract, multiply, divide, shift operations)
4. **Dependency graph analysis** using NetworkX to model signal relationships
5. **Structural analysis** of the RTL code (module depth, always blocks, etc.)

### Machine Learning Models
Multiple models are implemented and compared:
- Random Forest (best overall performance)
- Gradient Boosting
- Neural Networks
- Deep Networks

### Performance Metrics

| Model Type       | MAE  | RMSE | R²  | Within 1 Gate (%) | Within 2 Gates (%) |
|------------------|------|------|------|--------------------|---------------------|
| Random Forest   | 0.45 | 0.67 | 0.92 | 85.3               | 97.2                |
| Gradient Boosting | 0.49 | 0.71 | 0.91 | 83.7               | 96.5                |

## Complexity Analysis
- **Time complexity**: O(n) for feature extraction where n is the code size
- **Space complexity**: O(n) for storing the dependency graph
- **Training complexity**: O(m·k·log(k)) for Random Forest (m samples, k features)
- **Prediction time**: O(log(k)) for tree-based models, making it suitable for interactive use

## Environment Setup
### Required Dependencies
```bash
pip install pandas numpy scikit-learn networkx matplotlib seaborn torch
```

### Clone the Repository
```bash
git clone https://github.com/your-username/rtl-depth-predictor.git
cd rtl-depth-predictor
```

### Optional: Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Running the Code
### Basic Usage
```bash
python rtl_depth_predictor.py --rtl_file path/to/rtl.v --target_signal result
```

### Training a New Model
```bash
python rtl_depth_predictor.py --train --dataset path/to/dataset.csv --model_type random_forest
```

### Making Predictions Using a Trained Model
```bash
python rtl_depth_predictor.py --predict --model_path rtl_depth_predictor.pkl --rtl_file path/to/rtl.v
```

## Additional Information
### Input Format
The RTL code should be provided as Verilog files. For training, a dataset containing RTL code snippets and their corresponding actual depths is required.

### Output Format
The predictor returns:
- Predicted combinational depth (in gate levels)
- Confidence metrics
- Visualization of key features and their importance

### Parameters
- `model_type`: Type of model to use ('random_forest', 'gradient_boosting')
- `deep_learning_params`: Configuration for deep learning models (epochs, learning rate, etc.)

## Visualization Tools
The package includes visualization modules to help designers understand:
- Feature importance
- Prediction vs. actual comparison
- Error distribution
- Feature correlation matrix

## Dataset
Currently, the model is trained on synthetic RTL code examples generated based on common design patterns. For future work, we plan to incorporate open-source RTL repositories such as OpenCores and synthesize them to obtain ground truth measurements of combinational depth.

## Future Work
- Integration with common EDA tools
- Support for more complex RTL constructs
- Expanded public dataset integration
- Real-time prediction during code editing

