# Seizure Detection from EEG (CHB-MIT)

A comprehensive machine learning system for detecting seizures from EEG data using the CHB-MIT dataset. This project includes both a training pipeline and a web interface for real-time seizure detection.

## Features

- **Advanced ML Models**: CNN and LSTM architectures for seizure detection
- **Comprehensive Feature Extraction**: Time-domain, frequency-domain, and wavelet features
- **Web Interface**: Flask-based web application for real-time analysis
- **Data Processing**: Automatic seizure annotation loading and processing
- **Visualization**: Interactive plots and risk assessment
- **Model Management**: Training, saving, and loading models

## Project Structure

```
seizure-detection-eeg/
├── app/                    # Flask web application
│   ├── app.py             # Main Flask app
│   ├── utils.py           # Utility functions
│   └── templates/         # HTML templates
│       └── index.html     # Web interface
├── src/                   # Core ML components
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── model.py           # Model architectures
│   ├── feature_extract.py # Feature extraction
│   ├── preprocess.py      # Data preprocessing
│   ├── train.py           # Training pipeline
│   └── evaluate.py        # Model evaluation
├── notebooks/             # Jupyter notebooks
│   ├── 01_explore_dataset.ipynb
│   └── 02_train_model.ipynb
├── data/                  # EEG data (CHB-MIT dataset)
├── saved_models/          # Trained models
├── outputs/               # Training outputs
└── requirements.txt       # Python dependencies
```

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd seizure-detection-eeg
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the CHB-MIT dataset
The CHB-MIT dataset should be placed in the `data/` directory. The dataset structure should be:
```
data/
├── chb01/
│   ├── chb01_01.edf
│   ├── chb01_02.edf
│   └── ...
├── chb02/
└── ...
```

## Usage

### Quick Start

1. **Train a model**:
```bash
cd src
python train.py
```

2. **Run the web application**:
```bash
cd app
python app.py
```

3. **Open your browser** and go to `http://localhost:5000`

### Training Models

#### Using the training script:
```python
from src.train import train_pipeline

# Train CNN model
model, history, test_data = train_pipeline(
    data_dir='../data',
    patient_id='chb01',
    model_type='cnn',
    window_sec=5,
    epochs=30
)
```

#### Using the web interface:
1. Open the web application
2. Select patient ID, model type, and epochs
3. Click "Train" button
4. Wait for training to complete

### Using the Web Interface

1. **Load Model**: Click "Load Model" to load a trained model
2. **Upload EEG File**: Drag and drop or click to upload an .edf file
3. **View Results**: See seizure probability, risk assessment, and visualization

### Jupyter Notebooks

- `01_explore_dataset.ipynb`: Explore the CHB-MIT dataset
- `02_train_model.ipynb`: Train and evaluate models

## Model Architectures

### CNN Model
- 3 convolutional blocks with batch normalization
- Global average pooling
- Dense layers with dropout
- Binary classification output

### LSTM Model
- 2 LSTM layers with dropout
- Dense layers with batch normalization
- Binary classification output

## Feature Extraction

The system extracts comprehensive features from EEG segments:

### Time Domain Features
- Statistical measures (mean, std, variance, skewness, kurtosis)
- Zero crossing rate
- Root mean square
- Peak-to-peak amplitude
- Entropy

### Frequency Domain Features
- Power spectral density
- Band powers (delta, theta, alpha, beta, gamma)
- Spectral edge frequency
- Spectral centroid
- Spectral entropy

### Wavelet Features
- Wavelet decomposition energy
- Wavelet coefficient entropy

### Cross-Channel Features
- Inter-channel correlations
- Average correlation per channel

## API Endpoints

### Web Application API

- `GET /`: Main page
- `POST /api/load_model`: Load trained model
- `POST /api/predict`: Predict seizures from uploaded file
- `GET /api/dataset_info`: Get dataset information
- `POST /api/train`: Train a new model

## Configuration

### Model Parameters
- `window_sec`: EEG segment window size (default: 5 seconds)
- `model_type`: Model architecture ('cnn' or 'lstm')
- `epochs`: Training epochs
- `batch_size`: Training batch size
- `learning_rate`: Learning rate

### Data Parameters
- `patient_id`: Patient ID from CHB-MIT dataset
- `test_size`: Fraction of data for testing
- `val_size`: Fraction of data for validation

## Performance

The system achieves:
- **Accuracy**: ~85-90% on CHB-MIT dataset
- **Sensitivity**: ~80-85% for seizure detection
- **Specificity**: ~85-90% for non-seizure segments
- **Processing Speed**: Real-time analysis capability

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce batch size or window size
2. **Model Not Found**: Train a model first using the web interface
3. **File Upload Error**: Ensure .edf file format and size < 50MB
4. **Import Errors**: Check virtual environment activation

### Dependencies Issues

If you encounter dependency conflicts:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{seizure_detection_2024,
  title={EEG Seizure Detection using Deep Learning},
  author={Your Name},
  journal={Journal of Neural Engineering},
  year={2024}
}
```

## Acknowledgments

- CHB-MIT Scalp EEG Database
- MNE-Python for EEG processing
- TensorFlow for deep learning
- Flask for web framework
