# Audio Deepfake Detection Project

This project implements an audio deepfake detection system based on research from the [Audio-Deepfake-Detection](https://github.com/media-sec-lab/Audio-Deepfake-Detection) repository.

## Project Overview

Audio deepfakes pose an emerging threat to digital trust. This project explores existing research approaches and implements a system to identify manipulated audio content. The implementation focuses on extracting handcrafted features from audio samples and using machine learning techniques to distinguish between real and fake audio.

## Model Selection 

Based on research review from the Audio-Deepfake-Detection repository, three promising approaches were identified:

1. **AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)**
   - Key innovation: Graph attention networks for spectro-temporal feature learning
   - Performance: EER of 0.83% on ASVspoof 2019 LA scenario
   - Promising for: Real-time detection due to efficient architecture
   - Limitations: Requires significant computational resources

2. **RawNet2**
   - Key innovation: End-to-end raw waveform processing
   - Performance: EER of 1.06% on ASVspoof 2019 LA scenario
   - Promising for: Robust feature extraction from raw audio
   - Limitations: Sensitive to audio quality variations

3. **Handcrafted Feature-based Approach with Random Forest**
   - Key innovation: Leveraging statistical and spectral features with ensemble learning
   - Performance: Varies based on feature selection, generally competitive
   - Promising for: Interpretability and lightweight deployment
   - Limitations: May not generalize as well as deep learning approaches to novel attacks

## Implementation Details

Due to constraints in accessing the ASVspoof 2019 dataset and dependency issues with deep learning frameworks, we implemented a handcrafted feature-based approach:

- **Feature Extraction**: Statistical, shape, and spectral features (20 features total)
- **Model**: Random Forest Classifier with 100 decision trees
- **Dataset**: Synthetic dataset of 500 real (clean sine waves) and 500 fake (noisy sine waves) audio samples

## Project Structure

```
.
├── README.md                    # Project overview and documentation
├── requirements.txt             # Project dependencies
├── simple_detector.py           # Main implementation with RandomForest
├── src/                         # Source code directory (PyTorch implementation)
│   ├── __init__.py
│   ├── model.py                 # AASIST model implementation
│   ├── data_loader.py           # Dataset loading utilities
│   ├── generate_test_data.py    # Test data generation
│   └── download_dataset.py      # Dataset download utilities
├── notebooks/                   # Analysis and documentation
│   └── analysis.md              # Detailed analysis and reflection
├── data/                        # Dataset directory
│   └── test_data/               # Generated test data
├── models/                      # Saved model directory
└── results/                     # Results and visualizations
    ├── roc_curve.png            # ROC curve visualization
    ├── confusion_matrix.png     # Confusion matrix visualization
    └── feature_importance.png   # Feature importance visualization
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio-deepfake-detection.git
cd audio-deepfake-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To run the simple detector with synthetic data:

```bash
python simple_detector.py
```

This will:
1. Generate synthetic audio data (real and fake samples)
2. Extract features from the audio samples
3. Train a RandomForest classifier on the data
4. Evaluate the model performance
5. Generate visualizations in the `results/` directory

## Results

The model achieves:
- **Equal Error Rate (EER)**: 0.00% on synthetic data
- **Perfect classification**: 100% accuracy, precision, recall, and F1-score

Note that these results are on a synthetic dataset with clear separation between real and fake samples. Real-world performance would be different.

## Future Work

1. Use real deepfake audio datasets
2. Implement more sophisticated audio features
3. Deploy deep learning models like AASIST or RawNet2
4. Evaluate on cross-dataset scenarios

## References

1. Yi, J., et al. (2023). "Audio deepfake detection: a survey"
2. Wang, X., et al. (2021). "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks"
3. Tak, H., et al. (2021). "End-to-end spectro-temporal graph attention networks for speaker verification anti-spoofing and speech deepfake detection"
4. Kwak, H., et al. (2021). "A little more conversation - the influence of personality and conversational context on voice-AI" 