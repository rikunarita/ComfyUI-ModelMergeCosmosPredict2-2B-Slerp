# ComfyUI-ModelMergeCosmosPredict2-2B-Slerp

A specialized ComfyUI custom node for merging **Cosmos Predict 2 2B** models using **Slerp (Spherical Linear Interpolation)** and Linear Interpolation, **also compatible with the currently trending *Anima* model developed by circlestone-labs**.

Unlike standard linear merging, Slerp preserves the magnitude of the weight vectors, resulting in sharper and more stable outputs when merging models with significantly different characteristics or styles.

##  Features

- **Slerp & Linear Support**: Seamlessly switch between Slerp and standard Linear (weighted average) interpolation via a simple dropdown menu.
- **Layer-wise Ratio Control**: Adjust the merge ratio for each specific layer (Embeddings, 28 Transformer Blocks, Final Layer) individually using dedicated sliders.
- **Safe Fallback Mechanism**: Automatically falls back to Linear interpolation if Slerp encounters numerical instability (e.g., zero-norm tensors or shape mismatches), ensuring a crash-free workflow.
- **Optimized Memory Management**: Properly handles VRAM cleanup and model patching following ComfyUI's best practices.
- **Drop-in Replacement**: Fully compatible with the original `ModelMergeCosmosPredict2_2B` node structure.

## 📦 Installation

### Option 1: ComfyUI Manager (Recommended)
1. Open **ComfyUI Manager**.
2. Click on **"Install via Git URL"**.
3. Paste the following URL and click Install:
   ```
   https://github.com/rikunarita/ComfyUI-ModelMergeCosmosPredict2-2B-Slerp.git
   ```
4. Restart ComfyUI.

### Option 2: Manual Installation
1. Navigate to your `ComfyUI/custom_nodes/` directory.
2. Clone this repository:
   ```bash
   git clone https://github.com/rikunarita/ComfyUI-ModelMergeCosmosPredict2-2B-Slerp.git
   ```
3. Restart ComfyUI.

## 🚀 Usage

1. Load two Cosmos Predict 2B checkpoints using the standard **Load Checkpoint** nodes.
2. Add the **Model Merge Cosmos Predict 2 2B (Slerp)** node to your graph.
3. Connect the two models to `model1` and `model2`.
4. Set the **`merge_mode`** to either `slerp` or `linear`.
5. Adjust the individual sliders for each block to control the blending ratio (0.0 to 1.0).
6. Connect the output to a KSampler or other downstream nodes.

### Parameters Explained

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `merge_mode` | Combo | `slerp` | Selects the interpolation algorithm. `slerp` for sharp blending, `linear` for standard weighted average. |
| `pos_embedder.` | Float | 1.0 | Merge ratio for the positional embedding layer. |
| `x_embedder.` | Float | 1.0 | Merge ratio for the input (pixel) embedding layer. |
| `t_embedder.` | Float | 1.0 | Merge ratio for the time embedding layer. |
| `t_embedding_norm.` | Float | 1.0 | Merge ratio for the time embedding normalization layer. |
| `blocks.0.` ~ `blocks.27.` | Float | 1.0 | Merge ratios for the 28 individual Transformer blocks. |
| `final_layer.` | Float | 1.0 | Merge ratio for the final output projection layer. |

## 📝 Node Details

- **Node Name**: `ModelMergeCosmosPredict2_2B_Slerp`
- **Display Name**: Model Merge Cosmos Predict 2 2B (Slerp)
- **Category**: `model/merging/model specific`
- **Inputs**: 
  - `model1` (MODEL)
  - `model2` (MODEL)
- **Outputs**: 
  - `MODEL` (Merged Model)

## 🛠 Technical Notes

- **Numerical Stability**: The Slerp implementation includes strict numerical safeguards, including zero-norm prevention, dot-product clamping, and automatic fallback to Linear interpolation when the angle between vectors is too small (`DOT_THRESHOLD=0.9995`).
- **State Dict Handling**: Uses `strict=False` when loading the merged state dictionary to gracefully handle any minor structural discrepancies between the two source models.

## 📄 License & Credits

This custom node is built upon the architecture of the official ComfyUI model merging nodes (`comfy_extras.nodes_model_merging`). 

- **ComfyUI**: [ComfyUI Official Repository](https://github.com/comfyanonymous/ComfyUI)
- **Slerp Math**: Standard Spherical Linear Interpolation adapted for high-dimensional PyTorch tensors.

---
*Made with ❤️ for the ComfyUI Community.*
