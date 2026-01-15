# ERTECNet

**ERTECNet** (Enhanced Rapid Training Echo Convolution Network) is a hybrid deep learning architecture that combines a lightweight Convolutional Neural Network (CNN) backbone with a Deep Echo State Network (ESN) readout layer. It is designed for efficient image classification, particularly suitable for medical imaging tasks (like PCam, H&E metastatic and non-metastatic images) and standard benchmarks (MNIST, CIFAR-10). Currently developing a Multimodal Version of ERTECNet. Upcoming features will include: Support vector machine (SVM) integration and Weighted majority voting mechanisms for ensemble decision-making. 

To fully understand the theoretical foundations and the specific architectural novelties introduced in this project, we recommend the following reading order:

Foundation: Rapid Training Echo Convolution Network for Image Recognition [https://doi.org/10.1016/j.ins.2024.121750]. This paper outlines the base concepts of combining convolutions with echo state networks.

Project Specifics: Please refer to the PDF included in this repository pages between 20 and 38. This document details the unique modifications, hyperparameter updates, and structural changes implemented in ERTECNet compared to the baseline architecture.

## Key Features

- **Hybrid Architecture**: 
  - **CNN Backbone**: Efficient MBConv blocks (similar to EfficientNet) with Efficient Channel Attention (ECA) or Squeeze-Excite (SE) and DropPath regularization.
  - **ESN Readout**: A Deep Echo State Network (MESN) that processes spatial features as a sequence, providing a robust and efficient classification head.
  - **Attention Mechanism**: Optional Self-Attention mechanism within the ESN to weight important spatial features.
- **Robust Training**:
  - **Ridge Regression Solver**: Uses a robust Cholesky solver with jitter and fallback to Eigendecomposition for stable training of the ESN readout.
  - **Warm Start**: Option to warm-start the ESN output weights (`W_out`) before end-to-end training to ensure meaningful gradients flow back to the CNN from epoch 1.
- **Metrics & Visualization**:
  - precise tracking of Accuracy, F1-Score, Precision, Recall, and AUC-ROC.
  - Automatic generation of Confusion Matrices and ROC Curves.
  - TensorBoard support for model graph visualization.

## Requirements

Ensure you have a Python environment (3.8+) with the following packages:

```bash
pip install torch torchvision numpy scikit-learn
```

Optional dependencies for advanced features:

```bash
# For model summary
pip install torchinfo

# For TensorBoard logging
pip install tensorboard

# For plotting Confusion Matrices and ROC Curves
pip install matplotlib seaborn

# For PCam dataset
pip install datasets
```

## Usage

 The main script is `ERTECNet.py`. You can run it from the command line with various arguments.

### Basic Command

```bash
python ERTECNet.py --dataset mnist --epochs 5 --batch-size 965
```
Suitable for 6GB GPU.

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset to use: `mnist`, `cifar10`, `pcam`, `random_uniklinikum` | `mnist` |
| `--dataset-root` | Root directory for datasets | `./data` |
| `--image-size` | Input image size (Height Width) | `28 28` |
| `--batch-size` | Batch size for training and evaluation | `800` |
| `--epochs` | Number of training epochs | `5` |
| `--lr` | Learning rate for the CNN backbone | `0.01` |
| `--weight-decay` | Weight decay for the optimizer | `5e-5` |
| `--attn-type` | Attention type in ESN: `none` or `softmax` | `none` |
| `--neurons-per-deep` | Number of neurons per deep reservoir layer | `20` |
| `--sub-reservoirs` | Number of sub-reservoirs per layer | `4` |
| `--warmup-batches` | Batches to warm-start `W_out` (0 for full train set) | `1` |
| `--save-cm` | Save confusion matrix images | `False` |
| `--save-roc` | Save ROC curve images | `False` |
| `--metrics-path` | CSV file to log per-epoch metrics | (Disabled) |
| `--save-best-path` | Path to save the best model checkpoint | (Disabled) |


**Train on PCam (PatchCamelyon):**

```bash
# Requires 'datasets' library
python ERTECNet.py --dataset pcam --image-size 96 96 --batch-size 200 --epochs 10 --save-roc --metrics-path pcam_metrics.csv --attn-type softmax 
```
Suitable for 6GB GPU.

## Output

- **Console**: detailed logs per epoch including Train Loss, Test Loss, Accuracy, F1, Precision, Recall, and ROC AUC.
- **Metrics CSV**: If `--metrics-path` is provided, a CSV file with all metrics is generated.
- **Plots**: Confusion Matrices and ROC curves are saved to specified directories if enabled.
- **Checkpoints**: The best model (based on test accuracy) is saved if `--save-best-path` is provided.

## Architecture Details

The model processes an image $x$ through:
1.  **CNN**: $h = \text{CNN}(x)$ resulting in feature map $[B, C, H, W]$.
2.  **Sequence Conversion**: $h$ is flattened to a sequence of spatial tokens $u = [B, T, C]$ where $T = H \times W$.
3.  **ESN**: The sequence $u$ is fed into a Deep ESN.
    - Each layer of the ESN consists of multiple sub-reservoirs.
    - The final state representations are concatenated.
4.  **Readout**: A Ridge Regression layer maps the reservoir states to class logits. `W_out` is computed using a closed-form solution (or iterative approximation) during training.
