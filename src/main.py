import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from sklearn.decomposition import FastICA

def plot_spectrogram(audio_signals):
    for index, signalData in enumerate(audio_signals):
        plt.subplot(211)
        plt.title(f'Spectrogram of a wave file source {index+1}')
        plt.plot(signalData[1])
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        plt.subplot(212)
        plt.specgram(signalData[1], Fs=signalData[0])
        plt.xlabel('Time')
        plt.ylabel('Frequency')

        plt.show()

def generate_random_mixing_matrix(low=0.5, high=2.5, size=(6, 6)):
    return np.random.uniform(low, high, size)

def standardize_signals(signals):
    std_dev = signals.std(axis=0, keepdims=True)
    std_dev[std_dev == 0] = 1
    standardized_signals = signals / std_dev
    return standardized_signals

def mix_signals(standardized_signals, mixing_matrix):
    mixed_signals = np.dot(standardized_signals, mixing_matrix)
    return mixed_signals

def save_mixed_signals(mixed_signals, audio_signals, base_filename="mixed_signal"):
    sampling_rate = audio_signals[0][0]
    mixed_signals = mixed_signals.astype(np.int8)
    for i, signal in enumerate(mixed_signals.T):
        filename = f"{base_filename}_{i + 1}.wav"
        wavfile.write(filename, sampling_rate, signal)

def apply_ica_and_normalize(mixed_signals, n_components):
    ica = FastICA(n_components=n_components, random_state=0)
    separated_signals = ica.fit_transform(mixed_signals)
    for i in range(separated_signals.shape[1]):
        separated_signals[:, i] -= separated_signals[:, i].mean()
        separated_signals[:, i] /= np.linalg.norm(separated_signals[:, i])
    return separated_signals

def save_normalized_separated_signals(separated_signals, audio_signals, base_filename="separated_signal_normalized"):
    sampling_rate = audio_signals[0][0]
    separated_signals = np.int16(separated_signals / np.max(np.abs(separated_signals)) * 32767)
    for i, signal in enumerate(separated_signals.T):
        wavfile.write(f"{base_filename}_{i + 1}.wav", sampling_rate, signal)

def main():
    sources_name = ["./data/source1.wav", "./data/source2.wav", "./data/source3.wav", "./data/source4.wav", "./data/source5.wav", "./data/source6.wav"]
    audio_signals = [wavfile.read(source) for source in sources_name]
    plot_spectrogram(audio_signals)

    random_mixing_matrix = generate_random_mixing_matrix().T
    extracted_signals = np.array([signal[1] for signal in audio_signals])
    all_same_length = all(signal.shape[0] == extracted_signals[0].shape[0] for signal in extracted_signals)
    if not all_same_length:
        min_length = min(signal.shape[0] for signal in extracted_signals)
        extracted_signals = np.array([signal[:min_length] for signal in extracted_signals])

    transposed_signals = extracted_signals.T
    standardized_signals = standardize_signals(transposed_signals)
    mixed_signals = mix_signals(standardized_signals, random_mixing_matrix)

    save_mixed_signals(mixed_signals, audio_signals)
    separated_signals = apply_ica_and_normalize(mixed_signals, n_components=len(audio_signals))
    save_normalized_separated_signals(separated_signals, audio_signals)

    prepared_signals = [(audio_signals[0][0], signal) for signal in separated_signals.T]
    plot_spectrogram(prepared_signals)

if __name__ == "__main__":
    main()
