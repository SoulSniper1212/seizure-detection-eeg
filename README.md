# Seizure Detection from EEG (CHB-MIT)

This project provides a pipeline to train a CNN model for seizure detection from the CHB-MIT scalp EEG database.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd seizure-detection-eeg
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data Download

1.  Download the CHB-MIT dataset from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/).
2.  Specifically, you need the data for `chb01`.
3.  Organize the data into the following structure:
    ```
    seizure-detection-eeg/
    └── data/
        └── chb01/
            ├── chb01_01.edf
            ├── chb01_02.edf
            └── ...
    ```

## Usage

### 1. Preprocessing

To preprocess the data, you can run the segmentation script. *(Note: A separate script for this will be added)*

### 2. Training

To train the model, run the training script:
```bash
python -c "from src.train import train_model; from src.data_loader import load_chbmit_patient; from src.preprocess import segment_eeg; eeg_data, raw_objects = load_chbmit_patient('data'); X = segment_eeg(eeg_data, raw_objects); # Add dummy labels for now; y = [0]*len(X); train_model(X, y, X, y)"
```
*(Note: This is a placeholder command. A more robust training script will be provided.)*

### 3. Evaluation

To evaluate the model, run the evaluation script. *(Note: A separate script for this will be added)*

### 4. Streamlit App

A Streamlit app will be available to visualize the results. To run it:
```bash
streamlit run app/app.py
