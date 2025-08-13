# Ultrasound Muscle Segmentation with U-Net++

This project implements a U-Net++ model for segmenting the rectus femoris and vastus medialis muscles from ultrasound images using PyTorch and the `segmentation-models-pytorch` library. The model is trained on pre-split `.npy` datasets and includes post-processing to refine segmentation masks. The code is organized for training, evaluation, and visualization of segmentation results, with outputs saved to specified directories.

## Prerequisites

- Python 3.8+
- Google Colab or a local environment with GPU support (recommended)
- Google Drive for data storage and model saving
- Required libraries listed in `requirements.txt`

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/unetplusplus_segmentation.git
   cd unetplusplus_segmentation
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**:
   - Place your pre-split `.npy` files in the appropriate Google Drive directories
   - Expected files: `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`

4. **Mount Google Drive** (if using Colab):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Usage

### Training
Run the training script to train U-Net++ models for both rectus femoris and vastus medialis:
```bash
train.py
```

- Training runs for 50 epochs, saving the model with the best validation Dice score.

### Evaluation
Run the evaluation script to generate and save predictions for both training and test sets:
```bash
evaluate.py
```
- Predictions are saved as PNG images
- Outputs include input images, ground truth masks, raw predictions, and post-processed masks.

## Notes
- The model uses a ResNet34 backbone with pre-trained ImageNet weights.
- Input images are grayscale (single channel).
- Post-processing keeps the largest connected component and fills holes in the segmentation masks.
- The Dice score is used to evaluate segmentation performance.
- Ensure sufficient Google Drive storage for saving models and prediction images.

## Dependencies
See `requirements.txt` for a full list of dependencies. Key libraries include:
- `torch`
- `segmentation-models-pytorch`
- `numpy`
- `scipy`
- `matplotlib`
- `tqdm`

## Acknowledgments
- Built using the `segmentation-models-pytorch` library.
- Developed in Google Colab with GPU support.
