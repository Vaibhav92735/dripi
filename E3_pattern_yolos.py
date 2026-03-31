import os
import shutil
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# ==========================================
# 1. GPU Check & Configuration
# ==========================================
# Check if GPU is available
if torch.cuda.is_available():
    device = 0  # Use the first GPU
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU Detected: {gpu_name}")
else:
    device = 'cpu'
    print("⚠️  No GPU detected. Training will run on CPU (this will be slow).")

# Input settings
# csv_file = 'pattern_dataset.csv'
# source_img_dir = 'pattern_dataset/'
yolo_dataset_dir = 'pattern_yolo_format' # Output folder for organized images

# Model settings
# Options: yolov8n-cls.pt (nano), yolov8s-cls.pt (small), yolov8m-cls.pt (medium)
model_ver = 'yolo26n-cls.pt' 
num_epochs = 10
batch_size = 8
img_size = 224

# # ==========================================
# # 2. Data Preparation
# # ==========================================
# def prepare_yolo_data():
#     """
#     Reads the CSV, performs the split, and copies images into
#     train/val/test folders organized by class.
#     """
#     if os.path.exists(yolo_dataset_dir):
#         print(f"Folder '{yolo_dataset_dir}' already exists. Skipping data prep.")
#         return

#     print("Reading CSV and splitting data...")
#     df = pd.read_csv(csv_file)
    
#     # Cleaning
#     df = df.dropna(subset=['class'])
#     class_counts = df['class'].value_counts()
#     rare_classes = class_counts[class_counts < 3].index
#     df = df[~df['class'].isin(rare_classes)]
    
#     print(f"Total valid images: {len(df)}")

#     # Stratified Split (80/20 then 80/20)
#     temp_df, test_df = train_test_split(
#         df, test_size=0.20, stratify=df['class'], random_state=42
#     )
#     train_df, val_df = train_test_split(
#         temp_df, test_size=0.20, stratify=temp_df['class'], random_state=42
#     )

#     splits = {'train': train_df, 'val': val_df, 'test': test_df}

#     print("Copying images to YOLO folder structure...")
#     for split_name, split_df in splits.items():
#         for index, row in split_df.iterrows():
#             filename = row['filename']
#             class_name = row['class']
            
#             # Paths
#             src_path = os.path.join(source_img_dir, filename)
#             dest_folder = os.path.join(yolo_dataset_dir, split_name, class_name)
#             dest_path = os.path.join(dest_folder, filename)
            
#             os.makedirs(dest_folder, exist_ok=True)
            
#             try:
#                 shutil.copy2(src_path, dest_path)
#             except Exception as e:
#                 print(f"Error copying {filename}: {e}")

#     print("Data preparation complete!")

# ==========================================
# 3. Training Loop (GPU Enabled)
# ==========================================
def train_yolo():
    print(f"Loading model: {model_ver}...")
    model = YOLO(model_ver) 

    print(f"Starting YOLO training on device: {device}...")
    
    # Train the model
    results = model.train(
        data=yolo_dataset_dir,  # Path to the folder we created
        epochs=num_epochs,
        imgsz=img_size,
        batch=batch_size,
        project='yolo_pattern_project', # Output folder name
        name='pattern_run',             # Run name
        device=device,                  # <--- THIS FORCES GPU USAGE (0)
        augment=True,
        plots=True,
        exist_ok=True                   # Overwrite existing run if needed
    )
    
    print("Training finished.")
    print(f"Results saved to: {results.save_dir}")

    # Validation
    metrics = model.val()
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")

if __name__ == "__main__":
    # prepare_yolo_data()
    train_yolo()