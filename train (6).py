import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from dataset import UltrasoundNpyDataset_NoTransforms

def dice_score(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice

def train_model(model, train_loader, val_loader, num_epochs, device, model_save_path):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val_dice = 0.0

    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")

        # Training Phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        for images, masks in tqdm(train_loader, desc="Training"):
            images = images.to(device)
            masks = masks.float().to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_dice += dice_score(outputs, masks).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                masks = masks.float().to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()
                val_dice += dice_score(outputs, masks).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        # Save best model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved at epoch {epoch+1} with Val Dice: {avg_val_dice:.4f} to {model_save_path}")

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}, Val Dice:   {avg_val_dice:.4f}\n")

    print("Training finished!")
    print(f"Best model was saved with Validation Dice: {best_val_dice:.4f} to {model_save_path}")

def main():
    # Define file paths for rectus femoris
    rf_data_folder = '/content/drive/MyDrive/intern RF transverse latest file/'
    rf_x_train_path = os.path.join(rf_data_folder, 'X_train.npy')
    rf_y_train_path = os.path.join(rf_data_folder, 'y_train.npy')
    rf_x_val_path = os.path.join(rf_data_folder, 'X_val.npy')
    rf_y_val_path = os.path.join(rf_data_folder, 'y_val.npy')
    rf_model_save_path = '/content/drive/MyDrive/internship models/unet++ model/rectus femoris/unet++_resnet34_best.pth'

    # Load rectus femoris data
    x_train = np.load(rf_x_train_path)
    y_train = np.load(rf_y_train_path)
    x_val = np.load(rf_x_val_path)
    y_val = np.load(rf_y_val_path)

    # Create datasets and dataloaders
    train_dataset = UltrasoundNpyDataset_NoTransforms(x_train, y_train)
    val_dataset = UltrasoundNpyDataset_NoTransforms(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Train rectus femoris model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from model import create_unetplusplus_model
    model = create_unetplusplus_model().to(device)
    print("Training U-Net++ model for rectus femoris...")
    train_model(model, train_loader, val_loader, num_epochs=50, device=device, model_save_path=rf_model_save_path)

    # Define file paths for vastus medialis
    vm_data_folder = '/content/drive/MyDrive/intern RF longitudinal latest file/'
    vm_x_train_path = os.path.join(vm_data_folder, 'X_train.npy')
    vm_y_train_path = os.path.join(vm_data_folder, 'y_train.npy')
    vm_x_val_path = os.path.join(vm_data_folder, 'X_val.npy')
    vm_y_val_path = os.path.join(vm_data_folder, 'y_val.npy')
    vm_model_save_path = '/content/drive/MyDrive/internship models/unet++ model/vastus medialis/unet++_resnet34_best.pth'

    # Load vastus medialis data
    x_train = np.load(vm_x_train_path)
    y_train = np.load(vm_y_train_path)
    x_val = np.load(vm_x_val_path)
    y_val = np.load(vm_y_val_path)

    # Create datasets and dataloaders
    train_dataset = UltrasoundNpyDataset_NoTransforms(x_train, y_train)
    val_dataset = UltrasoundNpyDataset_NoTransforms(x_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Train vastus medialis model
    model = create_unetplusplus_model().to(device)
    print("Training U-Net++ model for vastus medialis...")
    train_model(model, train_loader, val_loader, num_epochs=50, device=device, model_save_path=vm_model_save_path)

if __name__ == "__main__":
    main()