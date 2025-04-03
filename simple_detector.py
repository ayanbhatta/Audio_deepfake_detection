import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

# Create directories
os.makedirs("data/test_data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Test data generation
def generate_sine_wave(duration, sample_rate, frequency):
    """Generate a sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    return np.sin(2 * np.pi * frequency * t)

def generate_test_data(num_samples=500, duration=1.0, sample_rate=16000):
    """Generate test data: real and fake audio samples."""
    print("Generating test data...")
    real_samples = []
    fake_samples = []
    
    for i in tqdm(range(num_samples), desc="Generating samples"):
        # Generate real audio (clean sine wave)
        real_audio = generate_sine_wave(duration, sample_rate, 440)  # A4 note
        real_samples.append(real_audio)
        
        # Generate fake audio (distorted sine wave)
        fake_audio = generate_sine_wave(duration, sample_rate, 440)
        fake_audio = fake_audio + 0.1 * np.random.randn(len(fake_audio))
        fake_samples.append(fake_audio)
    
    return real_samples, fake_samples

# Feature extraction
def extract_features(audio, num_features=20):
    """Extract simple statistical features from audio."""
    features = []
    # Mean
    features.append(np.mean(audio))
    # Standard deviation
    features.append(np.std(audio))
    # Min
    features.append(np.min(audio))
    # Max
    features.append(np.max(audio))
    # Range
    features.append(np.max(audio) - np.min(audio))
    # Energy
    features.append(np.sum(audio**2))
    # Zero crossing rate
    features.append(np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio))
    # Median
    features.append(np.median(audio))
    # Quartiles
    features.append(np.percentile(audio, 25))
    features.append(np.percentile(audio, 75))
    # RMS
    features.append(np.sqrt(np.mean(audio**2)))
    # Skewness
    features.append(np.mean((audio - np.mean(audio))**3) / (np.std(audio)**3) if np.std(audio) > 0 else 0)
    # Kurtosis
    features.append(np.mean((audio - np.mean(audio))**4) / (np.std(audio)**4) if np.std(audio) > 0 else 0)
    
    # Fast Fourier Transform features
    fft_features = np.abs(np.fft.fft(audio))[:len(audio)//2]
    fft_features = fft_features[:5]  # Use only the first 5 frequency components
    features.extend(fft_features)
    
    # Pad if necessary
    while len(features) < num_features:
        features.append(0)
    
    # Truncate if necessary
    features = features[:num_features]
    
    return np.array(features)

# Compute Equal Error Rate
def compute_eer(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[idx]
    return eer, thresholds[idx]

# Plot ROC curve
def plot_roc(y_true, y_pred_proba, save_path="results/roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
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
    num_features = 20
    
    # Generate data
    real_samples, fake_samples = generate_test_data(num_samples=num_samples)
    
    # Extract features
    print("Extracting features...")
    X = []
    y = []
    
    for sample in tqdm(real_samples, desc="Processing real samples"):
        X.append(extract_features(sample, num_features))
        y.append(1)  # Label for real samples
    
    for sample in tqdm(fake_samples, desc="Processing fake samples"):
        X.append(extract_features(sample, num_features))
        y.append(0)  # Label for fake samples
    
    X = np.array(X)
    y = np.array(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    eer, threshold = compute_eer(y_test, y_pred_proba)
    
    # Print results
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print(f"\nEqual Error Rate (EER): {eer:.4f} at threshold {threshold:.4f}")
    
    # Plot ROC curve
    plot_roc(y_test, y_pred_proba)
    
    # Plot confusion matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Fake', 'Real'])
    plt.yticks(tick_marks, ['Fake', 'Real'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    print("Confusion matrix saved to results/confusion_matrix.png")
    
    # Feature importance
    feature_imp = model.feature_importances_
    indices = np.argsort(feature_imp)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(X.shape[1]), feature_imp[indices])
    plt.xticks(range(X.shape[1]), indices)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig("results/feature_importance.png")
    print("Feature importance plot saved to results/feature_importance.png")
    
    # Example inference
    print("\nExample inference:")
    real_features = extract_features(real_samples[0], num_features)
    fake_features = extract_features(fake_samples[0], num_features)
    
    real_pred = model.predict_proba([real_features])[0, 1]
    fake_pred = model.predict_proba([fake_features])[0, 1]
    
    print(f"Real audio prediction: {real_pred:.4f} (Expected: 1.0)")
    print(f"Fake audio prediction: {fake_pred:.4f} (Expected: 0.0)")
    print(f"Using threshold {threshold:.4f}:")
    print(f"Real audio classification: {'Real' if real_pred >= threshold else 'Fake'}")
    print(f"Fake audio classification: {'Real' if fake_pred >= threshold else 'Fake'}")

if __name__ == "__main__":
    main() 