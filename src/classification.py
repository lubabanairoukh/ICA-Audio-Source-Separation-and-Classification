


from sklearn.model_selection import train_test_split
from sklearn import metrics, linear_model, preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from scipy.io import wavfile
from scipy import signal
import numpy as np
import librosa


def add_noise(sample, noise_factor):
    """
    Adds Gaussian noise to a sample.

    This function generates Gaussian noise based on the length of the input sample
    and adds it to the original sample scaled by a specified noise factor. The augmented
    sample is then cast back to the original data type of the input sample.

    Parameters:
    - sample (np.array): The original sample to which noise will be added. This should be a NumPy array.
    - noise_factor (float): The factor by which the generated noise is scaled before being added to the sample.

    Returns:
    - np.array: The augmented sample with added Gaussian noise, in the same data type as the input sample.
    """
    noise = np.random.randn(len(sample))
    augmented_sample = sample + noise_factor * noise
    augmented_sample = augmented_sample.astype(sample.dtype)
    return augmented_sample



def manipulate(data, pitch_factor):
    """
    Shifts the pitch of an audio signal.

    This function alters the pitch of the given audio data by a specified number of semitones.
    The data is first converted to float32 type, then the pitch shift is applied. The function
    relies on the `librosa.effects.pitch_shift` method for the pitch shifting process.

    Parameters:
    - data (np.array): The audio data to be pitch shifted. It should be a 1D NumPy array.
    - pitch_factor (float): The number of semitones by which the pitch of the audio data will be shifted.
      Positive values increase the pitch, while negative values decrease it.

    Returns:
    - np.array: The pitch-shifted audio data as a NumPy array of type float32.
    """
    return librosa.effects.pitch_shift(data.astype(np.float32), sr=8000, n_steps=pitch_factor)



def varMatrix(sample):
    """
    Calculate the variance of a sample.

    This function computes the variance of a given sample, which represents the average
    of the squared differences from the mean. It first calculates the mean of the sample,
    then computes the squared differences of each element from the mean, and finally,
    sums these squared differences and divides by the size of the sample to get the variance.

    Parameters:
    - sample (np.array): The sample for which variance will be calculated. This should be a 1D NumPy array.

    Returns:
    - float: The variance of the sample.
    """
    sample_mean = np.mean(sample)
    squared_diffs = np.square(sample - sample_mean)
    variance = np.sum(squared_diffs) / sample.size
    return variance

def peak_frequency(sample):
    """
    Find the index of the peak frequency in a sample.

    This function calculates the index of the peak frequency in the given sample.
    It returns the index of the maximum value along the specified axis.

    Parameters:
    - sample (np.array): The sample containing frequency data. This should be a 1D NumPy array.

    Returns:
    - int: The index of the peak frequency in the sample.
    """
    return np.argmax(sample, axis=0)

def hnr_ratio(sample):
    """
    Calculate the Harmonics-to-Noise Ratio (HNR) of a sample.

    This function computes the Harmonics-to-Noise Ratio (HNR) of the given sample.
    It calculates the mean of the first 10 elements (assumed to represent harmonics)
    and the mean of the remaining elements (assumed to represent noise) in the sample.
    It then computes the ratio of harmonics to noise.

    Parameters:
    - sample (np.array): The sample for which the HNR will be calculated. This should be a 1D NumPy array.

    Returns:
    - float: The Harmonics-to-Noise Ratio (HNR) of the sample.
    """
    harmonics = np.mean(sample[:10])
    noise = np.mean(sample[10:])
    hnr = harmonics / noise
    return hnr



def spectral_flatness(sample):
    """
    Calculate the spectral flatness measure of a sample.

    This function computes the spectral flatness measure of the given sample, which is a
    measure of how much noise-like or tonal the signal is. It calculates both the geometric
    mean and arithmetic mean of the sample, then computes the spectral flatness as the ratio
    of the geometric mean to the arithmetic mean. In cases where the arithmetic mean is zero,
    the geometric mean is used instead.

    Parameters:
    - sample (np.array): The sample for which the spectral flatness measure will be calculated.
      This should be a 1D NumPy array.

    Returns:
    - np.array: The spectral flatness measure of the sample, computed for each element of the input array.
    """
    geometric_mean = np.exp(np.mean(np.log(sample + 1e-20), axis=0))
    arithmetic_mean = np.mean(sample, axis=0)

    # Initialize flatness array with geometric mean for all elements as the default
    flatness = np.full_like(arithmetic_mean, fill_value=geometric_mean)

    # Indices where arithmetic_mean is not zero
    non_zero_indices = arithmetic_mean != 0

    # Perform division only where arithmetic_mean is not zero
    flatness[non_zero_indices] = geometric_mean[non_zero_indices] / arithmetic_mean[non_zero_indices]

    return flatness

def mean_on_col(sample):
    """
    Calculate the mean along columns of a sample matrix.

    This function computes the mean along columns (axis 0) of the given sample matrix.

    Parameters:
    - sample (np.array): The sample matrix for which the mean along columns will be calculated.
      This should be a 2D NumPy array.

    Returns:
    - np.array: The mean along columns of the sample matrix.
    """
    return np.mean(sample, axis=0)


def mean_on_row(sample):
    """
    Calculate the mean along rows of a sample matrix.

    This function computes the mean along rows (axis 1) of the given sample matrix.

    Parameters:
    - sample (np.array): The sample matrix for which the mean along rows will be calculated.
      This should be a 2D NumPy array.

    Returns:
    - np.array: The mean along rows of the sample matrix.
    """
    return np.mean(sample, axis=1)


def std_matrix(sample):
    """
    Calculate the standard deviation of a sample matrix.

    This function computes the standard deviation of the values in the given sample matrix.

    Parameters:
    - sample (np.array): The sample matrix for which the standard deviation will be calculated.
      This should be a 2D NumPy array.

    Returns:
    - float: The standard deviation of the sample matrix.
    """
    return np.std(sample)


def sum_matrix(sample):
    """
    Calculate the sum of all elements in a matrix.

    This function computes the sum of all elements in the given matrix by iterating through
    each row and each value within the row and accumulating the sum.

    Parameters:
    - sample (np.array): The matrix for which the sum of all elements will be calculated.
      This should be a 2D NumPy array.

    Returns:
    - float: The sum of all elements in the matrix.
    """
    total_sum = 0
    for row in sample:
        for value in row:
            total_sum += value
    return total_sum





def train_logistic_regression_model(target_labels, feature_descriptions, *feature_arrays):
    """
    Train a logistic regression model using the provided features and target labels.

    This function trains a logistic regression model using the specified features and target labels.
    It first constructs the feature matrix by stacking the provided feature arrays horizontally.
    Then, it shuffles the data and splits it into training and testing sets. The logistic regression model
    is trained using the training data and evaluated on the testing data. Finally, it prints a classification
    report and confusion matrix to assess the model's performance.

    Parameters:
    - target_labels (array-like): The target labels for classification.
    - feature_descriptions (list): A list of descriptions for the features being used.
    - *feature_arrays (array-like): Variable length argument list containing the feature arrays.
      Each feature array should have the same length as the target_labels.

    Returns:
    - logistic_classifier (LogisticRegression): The trained logistic regression classifier.

    """
    feature_description_str = ", ".join(feature_descriptions)

    # Concatenate feature arrays into a single feature matrix for model training.
    X = np.column_stack(feature_arrays)

    # Shuffle the dataset to ensure that the data is randomly distributed.
    X, target_labels = shuffle(X, target_labels, random_state=1)

    # Split the dataset into training and testing sets, with 70% of the data reserved for testing.
    X_train, X_test, y_train, y_test = train_test_split(X, target_labels, test_size=0.70, random_state=1)

    # Initialize the logistic regression classifier with a maximum of 1000 iterations for convergence.
    logistic_classifier = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)

    # Train the logistic regression model using the training data.
    logistic_classifier.fit(X_train, y_train)

    expected = y_test

    # Predict the labels for the testing set using the trained model.
    predicted = logistic_classifier.predict(X_test)

    print("Logistic regression using %s features on 15 percent of the data:" % feature_description_str)
    print(metrics.classification_report(expected, predicted, zero_division=0))
    print("Confusion matrix:")
    print(metrics.confusion_matrix(expected, predicted))

    return logistic_classifier


def augment_data_with_noise(data_samples, labels):
    """
    Augment data samples with Gaussian noise.

    This function augments the provided data samples by adding Gaussian noise to each sample.
    It generates multiple noisy versions of each sample with varying noise levels and adds them
    to the augmented data set along with their corresponding labels.

    Parameters:
    - data_samples (array-like): The original data samples to be augmented.
    - labels (array-like): The labels corresponding to the original data samples.

    Returns:
    - tuple: A tuple containing the augmented data samples and their corresponding labels.
      Both augmented data samples and labels are returned as NumPy arrays.
    """
    augmented_samples = []
    augmented_labels = []

    for i, sample in enumerate(data_samples):
        augmented_samples.append(sample)
        augmented_labels.append(labels[i])

        for noise_factor in range(1, 5):
            augmented_sample = add_noise(sample, noise_factor)
            augmented_samples.append(augmented_sample)
            augmented_labels.append(labels[i])

    return np.array(augmented_samples), np.array(augmented_labels)


def augment_data_with_pitch(X, Y, pitch_factors):
    """
    Augment data samples with pitch manipulation.

    This function augments the provided data samples by applying pitch manipulation to each sample.
    It generates multiple pitch-shifted versions of each sample with varying pitch factors and adds them
    to the augmented data set along with their corresponding labels.

    Parameters:
    - X (array-like): The original data samples to be augmented.
    - Y (array-like): The labels corresponding to the original data samples.
    - pitch_factors (array-like): The pitch factors to be applied for pitch manipulation.

    Returns:
    - tuple: A tuple containing the augmented data samples and their corresponding labels.
      Both augmented data samples and labels are returned as NumPy arrays.
    """
    augmented_X = []
    augmented_Y = []

    for i in range(len(X)):
        sample = X[i]
        label = Y[i]

        augmented_X.append(sample)
        augmented_Y.append(label)

        for factor in pitch_factors:
            augmented_sample = manipulate(sample, factor)
            augmented_X.append(augmented_sample)
            augmented_Y.append(label)

    return np.array(augmented_X), np.array(augmented_Y)


def convert_wav_to_Sxx_matrix(wave_data_list, sampling_rate=8000):
    """
    Convert wave data to spectrogram matrices.

    This function converts a list of wave data samples into spectrogram matrices using
    the Short-Time Fourier Transform (STFT) with the provided sampling rate. Each wave data
    sample is transformed into its corresponding spectrogram matrix.

    Parameters:
    - wave_data_list (list): A list of wave data samples to be converted to spectrogram matrices.
    - sampling_rate (int, optional): The sampling rate of the wave data samples, in Hz. Default is 8000.

    Returns:
    - np.array: An array containing the spectrogram matrices corresponding to the input wave data samples.
      Each spectrogram matrix represents the time-frequency representation of a wave data sample.

    """
    return np.array([signal.spectrogram(wave_data, fs=sampling_rate)[2] for wave_data in wave_data_list])



def main():
 
    file_list =   ["./data/source1.wav", "./data/source2.wav", "./data/source3.wav", "./data/source4.wav", "./data/source5.wav", "./data/source6.wav"]
    tags = ["noise", "person", "person", "person", "noise", "person"]

    # Load audio samples from specified file paths.
    audio_samples = [wavfile.read(audio_file)[1] for audio_file in file_list]

    # Augment the original audio data with noise to create a more robust dataset.
    X_train, Y_train = augment_data_with_noise(audio_samples, tags)

    # Further augment the data by adjusting the pitch, increasing the dataset's variability.
    pitch_factors = list(range(2, 10))
    X_train, Y_train = augment_data_with_pitch(X_train, Y_train, pitch_factors)

    # Normalize audio data and convert it into a frequency-time representation using spectrograms.
    for i in range(len(X_train)):
        vector = X_train[i]
        X_train[i] = (vector - vector.mean()) / np.linalg.norm(vector - vector.mean())
    X_train = convert_wav_to_Sxx_matrix(X_train)

    feature_func_dict = {
        varMatrix: "variance of matrix",
        peak_frequency: "index of the peak frequency",
        hnr_ratio: "Harmonics-to-Noise Ratio (HNR)",
        mean_on_col: "mean along columns",
        mean_on_row: "mean along rows",
        spectral_flatness: "spectral flatness measure",
        std_matrix: "standard deviation of matrix",
        sum_matrix: "sum of all elements in a matrix",

    }

    # Map each feature extraction function over the spectrogram data to create feature arrays.
    features = [np.array(list(map(func, X_train))) for func in feature_func_dict.keys()]
    feature_names = list(feature_func_dict.values())

    # Train a logistic regression model using the extracted features and corresponding labels.
    l_r_model = train_logistic_regression_model(Y_train, feature_names, *features)

    return l_r_model


def main2(main_model):
    file_list_final = [
        "./output/separated_signal_normalized_1.wav", "./output/separated_signal_normalized_2.wav",
        "./output/separated_signal_normalized_3.wav", "./output/separated_signal_normalized_4.wav",
        "./output/separated_signal_normalized_5.wav", "./output/separated_signal_normalized_6.wav"
    ]
    expected = ["person", "noise", "person", "person", "person", "noise"]

    # Load audio samples for prediction, similar to the initial loading step in `main`.
    audio_samples = [wavfile.read(file)[1] for file in file_list_final]

    # Convert the loaded audio samples into spectrograms for feature extraction.
    spectrograms = [signal.spectrogram(sample, fs=8000)[2] for sample in audio_samples]
    X_temp = np.array(spectrograms)


    feature_functions = [varMatrix,peak_frequency, hnr_ratio, spectral_flatness, mean_on_col, mean_on_row, std_matrix,
                         sum_matrix]

    # Apply the defined feature extraction functions to the spectrogram data.
    X = np.column_stack([np.array(list(map(func, X_temp))) for func in feature_functions])

    # Normalize the feature matrix to ensure that all features contribute equally to the model prediction.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use the pre-trained model to predict the labels of the new dataset.
    predicted = main_model.predict(X_scaled)

    # Print a report comparing the expected labels to the model's predictions.
    print("Logistic regression on created data:\n%s\n" % (metrics.classification_report(expected,predicted)))

    # Print a confusion matrix to evaluate the model's performance visually.
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))





if __name__ == "__main__":
    main_model = main()
    main2(main_model)


