import torch
import numpy as np
import os
from scipy import ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import UltrasoundNpyDataset_NoTransforms
from model import load_trained_model

def post_process_mask(mask):
    labels, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    component_sizes = np.bincount(labels.ravel())
    if len(component_sizes) > 1:
        largest_component_label = component_sizes[1:].argmax() + 1
        processed_mask = (labels == largest_component_label)
        processed_mask = ndimage.binary_fill_holes(processed_mask)
        return processed_mask.astype(np.uint8)
    return mask

def visualize_and_save(processed_img, gt_mask, pred_raw, pred_post, save_path, title):
    if processed_img.shape[0] == 1:
        processed_img_display = processed_img.cpu().squeeze(0).numpy()
        processed_img_display = (processed_img_display - processed_img_display.min()) / (processed_img_display.max() - processed_img_display.min())
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        processed_img_display = processed_img.cpu().permute(1, 2, 0).numpy()
        processed_img_display = std * processed_img_display + mean
        processed_img_display = np.clip(processed_img_display, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(processed_img_display, cmap='gray')
    axes[0].set_title("Input Image"); axes[0].axis('off')
    axes[1].imshow(np.squeeze(gt_mask), cmap='gray'); axes[1].set_title("Ground Truth"); axes[1].axis('off')
    axes[2].imshow(np.squeeze(pred_raw), cmap='gray'); axes[2].set_title("Raw Prediction"); axes[2].axis('off')
    axes[3].imshow(np.squeeze(pred_post), cmap='gray'); axes[3].set_title("Post-Processed"); axes[3].axis('off')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def evaluate_model(model, data_loader, save_dir, dataset_type, device):
    total_dice_before = 0
    total_dice_after = 0
    num_samples = 0
    smooth = 1e-6
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, gt_masks) in enumerate(tqdm(data_loader, desc=f"{dataset_type} Prediction")):
            images = images.to(device)
            gt_masks_np = gt_masks.cpu().numpy()
            preds_logits = model(images)
            preds_sigmoid = torch.sigmoid(preds_logits)
            preds_before_np = (preds_sigmoid > 0.5).cpu().numpy()
            preds_after_np = np.array([post_process_mask(np.squeeze(p)) for p in preds_before_np])

            for j in range(images.shape[0]):
                image_idx = i * data_loader.batch_size + j
                gt = np.squeeze(gt_masks_np[j]).flatten()
                pred_before = np.squeeze(preds_before_np[j]).flatten()
                pred_after = np.squeeze(preds_after_np[j]).flatten()

                intersection_before = (pred_before * gt).sum()
                total_dice_before += (2. * intersection_before + smooth) / (pred_before.sum() + gt.sum() + smooth)
                intersection_after = (pred_after * gt).sum()
                total_dice_after += (2. * intersection_after + smooth) / (pred_after.sum() + gt.sum() + smooth)
                num_samples += 1

                save_path = os.path.join(save_dir, f"{dataset_type.lower()}_prediction_{image_idx+1}.png")
                visualize_and_save(
                    processed_img=images[j],
                    gt_mask=gt_masks_np[j],
                    pred_raw=preds_before_np[j],
                    pred_post=preds_after_np[j],
                    save_path=save_path,
                    title=f"{dataset_type} Set - Prediction {image_idx+1}"
                )

    avg_dice_before = total_dice_before / num_samples
    avg_dice_after = total_dice_after / num_samples
    print(f"\n--- {dataset_type} Set Evaluation Complete ---")
    print(f"Total {dataset_type} Images Processed: {num_samples}")
    print(f"Average Dice (Before Post-Processing): {avg_dice_before:.4f}")
    print(f"Average Dice (After Post-Processing): {avg_dice_after:.4f}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Rectus Femoris Evaluation
    rf_data_folder = '/content/drive/MyDrive/intern RF transverse latest file/'
    rf_x_test_path = os.path.join(rf_data_folder, 'X_test.npy')
    rf_y_test_path = os.path.join(rf_data_folder, 'y_test.npy')
    rf_x_train_path = os.path.join(rf_data_folder, 'X_train.npy')
    rf_y_train_path = os.path.join(rf_data_folder, 'y_train.npy')
    rf_model_path = '/content/drive/MyDrive/internship models/unet++ model/rectus femoris/unet++_resnet34_best.pth'
    rf_base_save_dir = '/content/drive/MyDrive/internship models/unet++ model/rectus femoris/segmentation_results_with_preprocessing'
    rf_train_save_dir = os.path.join(rf_base_save_dir, 'train_set_predictions')
    rf_test_save_dir = os.path.join(rf_base_save_dir, 'test_set_predictions')

    # Load data
    x_test = np.load(rf_x_test_path)
    y_test = np.load(rf_y_test_path)
    x_train = np.load(rf_x_train_path)
    y_train = np.load(rf_y_train_path)

    # Create datasets and dataloaders
    test_dataset = UltrasoundNpyDataset_NoTransforms(x_test, y_test)
    train_dataset = UltrasoundNpyDataset_NoTransforms(x_train, y_train)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)

    # Load and evaluate model
    model = load_trained_model(rf_model_path, device)
    print("Evaluating rectus femoris model...")
    evaluate_model(model, test_loader, rf_test_save_dir, "Test", device)
    evaluate_model(model, train_loader, rf_train_save_dir, "Train", device)

    # Vastus Medialis Evaluation
    vm_data_folder = '/content/drive/MyDrive/intern RF longitudinal latest file/'
    vm_x_test_path = os.path.join(vm_data_folder, 'X_test.npy')
    vm_y_test_path = os.path.join(vm_data_folder, 'y_test.npy')
    vm_x_train_path = os.path.join(vm_data_folder, 'X_train.npy')
    vm_y_train_path = os.path.join(vm_data_folder, 'y_train.npy')
    vm_model_path = '/content/drive/MyDrive/internship models/unet++ model/vastus medialis/unet++_resnet34_best.pth'
    vm_base_save_dir = '/content/drive/MyDrive/internship models/unet++ model/vastus medialis/segmentation_results_with_preprocessing'
    vm_train_save_dir = os.path.join(vm_base_save_dir, 'train_set_predictions')
    vm_test_save_dir = os.path.join(vm_base_save_dir, 'test_set_predictions')

    # Load data
    x_test = np.load(vm_x_test_path)
    y_test = np.load(vm_y_test_path)
    x_train = np.load(vm_x_train_path)
    y_train = np.load(vm_y_train_path)

    # Create datasets and dataloaders
    test_dataset = UltrasoundNpyDataset_NoTransforms(x_test, y_test)
    train_dataset = UltrasoundNpyDataset_NoTransforms(x_train, y_train)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)

    # Load and evaluate model
    model = load_trained_model(vm_model_path, device)
    print("Evaluating vastus medialis model...")
    evaluate_model(model, test_loader, vm_test_save_dir, "Test", device)
    evaluate_model(model, train_loader, vm_train_save_dir, "Train", device)

if __name__ == "__main__":
    main()