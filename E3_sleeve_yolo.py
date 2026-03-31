import os
import shutil
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# ==========================================
# 1. GPU Check & Configuration
# ==========================================
if torch.cuda.is_available():
    device = 0  
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU Detected: {gpu_name}")
else:
    device = 'cpu'
    print("⚠️  No GPU detected. Training will run on CPU.")

# --- PATH CONFIGURATION ---
csv_file = 'f_s_1n.csv'
source_img_dir = 'f_s_1n/'          
yolo_dataset_dir = 'f_s_1n_yolo_format' 

# --- MODEL HYPERPARAMETERS ---
model_ver = 'yolov8s-cls.pt' 
num_epochs = 10
batch_size = 16
img_size = 224

# ==========================================
# 2. Data Preparation Function
# ==========================================
def prepare_yolo_data():
    """
    Reads CSV, samples 800 images per class, and creates 
    Train/Val/Test splits in subfolders.
    """
    if os.path.exists(yolo_dataset_dir):
        print(f"Directory '{yolo_dataset_dir}' already exists. Please delete it to re-run sampling.")
        return

    print(f"Reading {csv_file} and sampling 800 images per class...")
    df = pd.read_csv(csv_file)
    
    # 1. Cleaning
    df = df.dropna(subset=['class'])
    
    # 2. Filter out classes with fewer than 5 images to allow stratification
    class_counts = df['class'].value_counts()
    rare_classes = class_counts[class_counts < 5].index
    df = df[~df['class'].isin(rare_classes)]
    
    # 3. Sample exactly 800 images per class (or max available if < 800)
    df = df.groupby('class').apply(
        lambda x: x.sample(n=min(len(x), 500), random_state=42)
    ).reset_index(drop=True)
    
    print(f"Total images after sampling: {len(df)}")
    print("Samples per class:\n", df['class'].value_counts())

    # 4. Stratified Split (80% Train/Val, 20% Test)
    temp_df, test_df = train_test_split(
        df, test_size=0.20, stratify=df['class'], random_state=42
    )
    # Split temp into Train (80%) and Val (20%)
    train_df, val_df = train_test_split(
        temp_df, test_size=0.20, stratify=temp_df['class'], random_state=42
    )

    splits = {'train': train_df, 'val': val_df, 'test': test_df}

    # 5. Physical File Copying
    print("Copying files to YOLO structure...")
    for split_name, split_df in splits.items():
        for index, row in split_df.iterrows():
            filename = row['filename']
            class_name = row['class']
            
            # Source: fitn/Class_Name/img_xxxx.jpg
            src_path = os.path.join(source_img_dir, class_name, filename)
            
            # Destination: fitn_yolo_format/split/class/img_xxxx.jpg
            dest_folder = os.path.join(yolo_dataset_dir, split_name, class_name)
            dest_path = os.path.join(dest_folder, filename)
            
            os.makedirs(dest_folder, exist_ok=True)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_path)
            else:
                # Log only every 50th missing file to avoid cluttering console
                if index % 50 == 0:
                    print(f"Warning: File not found: {src_path}")

    print("✅ Data preparation complete!")

# ==========================================
# 3. Training Function
# ==========================================
def train_yolo():
    print(f"Loading weights: {model_ver}...")
    model = YOLO(model_ver) 

    print(f"🚀 Starting training on {device}...")
    
    # Training
    model.train(
        data=yolo_dataset_dir,
        epochs=num_epochs,
        imgsz=img_size,
        batch=batch_size,
        project='fs1_project',
        name='fs1_200',
        device=device,
        augment=True,
        plots=True,
        exist_ok=True
    )
    print("✅ Training finished.")
    # Validation
    print("Running final validation on test set...")
    metrics = model.val()
    print(f"Final Top-1 Accuracy: {metrics.top1:.4f}")
# ==========================================
# 4. Execution
# ==========================================
if __name__ == "__main__":
    prepare_yolo_data()
    train_yolo()