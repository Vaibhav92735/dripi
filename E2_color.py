import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. Configuration
# ==========================================
csv_file = 'subset_dataset.csv'
img_dir = 'subset_images/'
batch_size = 16
learning_rate = 0.001
num_epochs = 10

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 2. Custom Dataset Class (CORRECTED)
# ==========================================
class ClothingcolourDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        # We pass the whole dataframe (or a slice) so we can access columns by name
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # FIX: Use iloc to get the row by index, then access columns by name
        row = self.dataframe.iloc[idx]
        img_name = row['image_name']      # Access the 'image_name' column
        label = row['label_encoded']      # Access the 'label_encoded' column
        img_path = os.path.join(self.root_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Missing file: {img_path}")
            # Generate a black image or skip (crashing is usually safer for debugging)
            raise

        if self.transform:
            image = self.transform(image)
        return image, label

# ==========================================
# 3. Data Preparation (Fixed)
# ==========================================
df = pd.read_csv(csv_file)

# FIX 1: Drop rows where 'colour' is missing (NaN)
print(f"Original rows: {len(df)}")
df = df.dropna(subset=['colour'])
print(f"Rows after dropping NaN: {len(df)}")

# FIX 2: Remove rare colours (Count < 2)
colour_counts = df['colour'].value_counts()
rare_colours = colour_counts[colour_counts < 2].index
df = df[~df['colour'].isin(rare_colours)]
print(f"Rows after removing rare colours: {len(df)}")

# Encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['colour'])
num_classes = len(le.classes_)
print(f"Classes found: {le.classes_}")

# Split Data
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_encoded'])

# (Transforms and Datasets remain the same as before...)
# train_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
train_transforms = transforms.Compose([
    # 1. RandomResizedCrop: Forces model to look at parts of the item, not just the whole shape
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    
    # 2. Brightness/Contrast: Simulates different lighting conditions (Do NOT change hue)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
    # 3. RandomErasing: Randomly blacks out a rectangle. Forces model to use other visual cues.
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ClothingcolourDataset(train_df, img_dir, train_transforms)
val_dataset = ClothingcolourDataset(val_df, img_dir, val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ==========================================
# 4. Model Setup (Fixed GPU Placement)
# ==========================================
# model = models.resnet50(pretrained=True)

# # Unfreeze layers for fine-tuning
# for param in model.parameters():
#     param.requires_grad = True

# # Replace the head
# num_ftrs = model.fc.in_features
# model.fc = nn.Sequential(
#     nn.Dropout(0.5),
#     nn.Linear(num_ftrs, num_classes)
# )
# # FIX 3: Move model to GPU *after* all modifications
# model = model.to(device) 


model = models.resnet50(pretrained=True)

# 1. Freeze EVERYTHING first
for param in model.parameters():
    param.requires_grad = False

# 2. Unfreeze ONLY the last block (layer4) and the head (fc)
# 'layer4' is the final convolutional block in ResNet
for param in model.layer4.parameters():
    param.requires_grad = True

# 3. Replace and unfreeze the Head
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5), # Keep your dropout
    nn.Linear(num_ftrs, num_classes)
)

model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4) 
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)# Lower learning rate for fine-tuning
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)

# ==========================================
# 5. Training Loop
# ==========================================
print("Starting training...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.long().to(device) # Labels are now LongTensors from label_encoded
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = 100 * val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.2f}%")

print("Training complete.")
torch.save(model.state_dict(), 'clothing_colour_resnet.pth')
print("Model saved.")