
# ICA Audio Source Separation and Classification

## Description
This project demonstrates the use of Independent Component Analysis (ICA) to separate mixed audio signals into their original sources and subsequently classify them using a custom-developed classification algorithm. The project covers the following main steps:

1. Downloading and reading audio files.
2. Displaying the spectrogram of each audio signal.
3. Creating a random mixing matrix and generating mixed audio signals.
4. Applying ICA to separate the mixed signals into reconstructed signals.
5. Normalizing the reconstructed signals.
6. Displaying the spectrogram of each reconstructed signal.
7. Creating a training and test dataset from the original audio signals.
8. Proposing features for classification, training a classifier, and evaluating its performance.
9. Classifying the reconstructed signals.

## Files and Directories
- `source1.wav, source2.wav, source3.wav, source4.wav, source5.wav, source6.wav`: Original audio files.
- `mixed_signal1.wav, mixed_signal2.wav, ..., mixed_signal6.wav`: Mixed audio signals.
- `separated_signal_normalized_1.wav, separated_signal_normalized_2.wav, ..., separated_signal_normalized_6.wav`: Reconstructed audio signals after ICA.
- `src/`: Source code for the project.
  - `main.py`: Main script for the project, handling signal mixing, ICA application, normalization, and spectrogram plotting.
  - `classification.py`: Script for augmenting data, extracting features, training the classifier, and classifying reconstructed signals.
- `README.md`: This file.

## Setup and Installation
1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-username/ICA-Audio-Source-Separation-and-Classification.git
   cd ICA-Audio-Source-Separation-and-Classification
   ```

2. **Create and activate a virtual environment**:
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### 1. Running the Main Script
Run the main script to process the audio signals:
```sh
python src/main.py
```
This will:
- Load the original audio files.
- Display the spectrogram of each original audio signal.
- Create mixed signals using a random mixing matrix.
- Save the mixed signals.
- Apply ICA to separate the mixed signals.
- Normalize the separated signals.
- Save the normalized separated signals.
- Display the spectrogram of each normalized separated signal.

### 2. Running the Classification Script
Run the classification script to train the classifier and classify the reconstructed signals:
```sh
python src/classification.py
```
This will:
- Augment the original audio data with noise and pitch variations.
- Convert the audio data into spectrogram matrices.
- Extract features from the spectrograms.
- Train a logistic regression model.
- Classify the reconstructed signals using the trained model.

