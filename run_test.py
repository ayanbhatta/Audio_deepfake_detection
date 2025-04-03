import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Create directories
os.makedirs("data/test_data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Test data generation
def generate_sine_wave(duration, sample_rate, frequency):
    """Generate a sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    return np.sin(2 * np.pi * frequency * t)

def generate_test_data(num_samples=100, duration=1.0, sample_rate=16000):
    """Generate test data: real and fake audio samples."""
    print("Generating test data...")
    real_samples = []
    fake_samples = []
    
    for i in range(num_samples):
        # Generate real audio (clean sine wave)
        real_audio = generate_sine_wave(duration, sample_rate, 440)  # A4 note
        real_samples.append(real_audio)
        
        # Generate fake audio (distorted sine wave)
        fake_audio = generate_sine_wave(duration, sample_rate, 440)
        fake_audio = fake_audio + 0.1 * np.random.randn(len(fake_audio))
        fake_samples.append(fake_audio)
    
    return real_samples, fake_samples

# Feature extraction
def extract_features(audio, feature_size=128):
    """Simple feature extraction."""
    # Pad or truncate to fixed length
    if len(audio) > feature_size:
        return audio[:feature_size]
    elif len(audio) < feature_size:
        return np.pad(audio, (0, feature_size - len(audio)))
    else:
        return audio

# Dataset class
class AudioDataset(Dataset):
    def __init__(self, real_samples, fake_samples, feature_size=128):
        self.real_features = [extract_features(audio, feature_size) for audio in real_samples]
        self.fake_features = [extract_features(audio, feature_size) for audio in fake_samples]
        self.labels = [1] * len(self.real_features) + [0] * len(self.fake_features)
        self.features = self.real_features + self.fake_features
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx]).unsqueeze(0)  # Add channel dimension
        label = torch.FloatTensor([self.labels[idx]])
        return feature, label

# Model definition
class SimpleDeepfakeDetector(nn.Module):
    def __init__(self, input_size=128):
        super(SimpleDeepfakeDetector, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(32 * (input_size // 4), 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Input shape: (batch_size, 1, input_size)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# Compute Equal Error Rate
def compute_eer(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[idx]
    return eer, thresholds[idx]

# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for data, target in tqdm(train_loader, desc="Training"):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(output.detach().cpu().numpy())
        all_labels.extend(target.cpu().numpy())
    
    eer, _ = compute_eer(all_labels, all_preds)
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss, eer

# Evaluation function
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            all_preds.extend(output.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    eer, threshold = compute_eer(all_labels, all_preds)
    avg_loss = total_loss / len(test_loader)
    
    return avg_loss, eer, all_preds, all_labels, threshold

# Plot ROC curve
def plot_roc(y_true, y_pred, save_path="results/roc_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    print(f"ROC curve saved to {save_path}")

def main():
    # Parameters
    num_samples = 500
    feature_size = 128
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate data
    real_samples, fake_samples = generate_test_data(num_samples=num_samples)
    
    # Create datasets
    dataset = AudioDataset(real_samples, fake_samples, feature_size=feature_size)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = SimpleDeepfakeDetector(input_size=feature_size).to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    train_losses = []
    train_eers = []
    test_losses = []
    test_eers = []
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_eer = train(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_eers.append(train_eer)
        
        # Evaluate
        test_loss, test_eer, all_preds, all_labels, threshold = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_eers.append(test_eer)
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train EER: {train_eer:.4f} - "
              f"Test Loss: {test_loss:.4f}, Test EER: {test_eer:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "models/simple_deepfake_detector.pth")
    print("Model saved to models/simple_deepfake_detector.pth")
    
    # Plot ROC curve for the final epoch
    plot_roc(all_labels, all_preds)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train')
    plt.plot(range(1, num_epochs+1), test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_eers, label='Train')
    plt.plot(range(1, num_epochs+1), test_eers, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('EER')
    plt.title('Training and Testing EER')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/training_curves.png")
    print("Training curves saved to results/training_curves.png")
    
    # Final evaluation
    print(f"\nFinal Test EER: {test_eer:.4f} at threshold {threshold:.4f}")
    
    # Example inference
    print("\nExample inference:")
    model.eval()
    
    # Get one real and one fake sample
    real_sample = torch.FloatTensor(extract_features(real_samples[0], feature_size)).unsqueeze(0).unsqueeze(0)
    fake_sample = torch.FloatTensor(extract_features(fake_samples[0], feature_size)).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        real_pred = model(real_sample.to(device)).item()
        fake_pred = model(fake_sample.to(device)).item()
    
    print(f"Real audio prediction: {real_pred:.4f} (Expected: 1.0)")
    print(f"Fake audio prediction: {fake_pred:.4f} (Expected: 0.0)")
    print(f"Using threshold {threshold:.4f}:")
    print(f"Real audio classification: {'Real' if real_pred >= threshold else 'Fake'}")
    print(f"Fake audio classification: {'Real' if fake_pred >= threshold else 'Fake'}")

if __name__ == "__main__":
    main() 