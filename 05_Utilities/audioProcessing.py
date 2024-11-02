import librosa
import re
import math
import numpy as np

class processAudio:
    ########################################################
    # function to calculat n_fft for STFT

    ## librosa documentation at: https://librosa.org/doc/main/generated/librosa.stft.html
    ## --> n_fft: int > 0 [scalar] - length of the windowed signal after padding with zeros
    ## --> In any case, we recommend setting n_fft to a power of two for optimizing the speed of the fast Fourier transform (FFT) algorithm. 

    ## calculate the next power of 2 for input x
    def next_power_of_2(self, x):   
        next_power_of_two = 2**(math.ceil(math.log(x, 2)))

        if next_power_of_two == x:
            next_power_of_two *= 2

        return next_power_of_two
    ########################################################

    ########################################################
    # function to convert power mel-spectrogram to dB-scaled mel-spectrogram

    ## D_mel - power mel-spectrogram, a_squere_min - minimum value for the power, dB_ref - reference value for dB scaling
    def power_mel_to_db(self, D_mel, a_squere_min, dB_ref):

        D_mel_dB = 10.0 * np.log10(np.maximum(a_squere_min, np.minimum(D_mel, dB_ref)/dB_ref))

        ## Dynamic range from -120 dB to 0 dB 
        ## 0 dB as reference level -> log_10(1) = 0 -> values inside the log should be smaller or equal to 1
        ## --> values over dB_ref are set to dB_ref --> np.minimum(D_mel, dB_ref)
        ## -120 dB as the minimum possible dB value -> log_10(1e-12) -> values inside the log should be larger or equal to 1e-12
        ## --> values under a_square_min are set to a_square_min --> np.maximum(a_squere_min, np.minimum(D_mel, dB_ref)

        ## also tried with linear mapping, but this results in worse resolution for small values
        ## -> D_mel_dB = -120 + (120/dB_ref) * np.maximum(a_square_min, np.minimum(D_mel, dB_ref)) 

        return D_mel_dB

    ########################################################
    # calculate the dB-scaled mel-spectrogram

    ## audio_file_path - path to the audio file, sampling_rate - sampling rate of the audio file, hop_length - hop length for the STFT calculation
    def get_mel_spectrogram(self, audio_file_path, sampling_rate, hop_length, n_mels, f_min, f_max, a_squere_min, dB_ref):

        ## initialize the dB-scaled mel-spectrogram as None
        D_mel_dB = None
        
        ## load the audio file, get the signal's wave form
        signal, sampling_rate = librosa.load(audio_file_path, sr=sampling_rate)

        n_samples = len(signal) ## number of all samples in the signal
        duration = n_samples / sampling_rate ## duration of the signal in seconds
        time = np.linspace(0, duration, n_samples) ## time vector for plotting

        n_fft = self.next_power_of_2(hop_length) ## length of the windowed signal after padding with zeros

        # padding signal on both sides with hop_length/2 * zeros -> important in stft calculation for chunks
        #signal_padded = np.pad(signal, (hop_length//2, hop_length//2), 'constant', constant_values=(0, 0))
        
        ## Short-Time Fourier Transform (STFT) to get the signal's spectrum
        ## time-frequency representation of the signal, the values in the matrix are the amplitude values of the frequencies at the given time
        signal_stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center = True)

        ## calculate the power spectrogram
        ## I = c * A^2, where c is a constant, A is the amplitude, I is the intensity (Roberts 1984)
        ## D = |STFT|^2 = |A|^2, the values in the matrix are proportional with the power values of the frequencies at the given time
        D = np.abs(signal_stft) ** 2

        # print the maximum value in the power spectrogram
        print(f"Max value in the power spectrogram: {np.max(D)}")

        ## creating the mel filter bank for mel-scaled spectrogram generation
        mel_filter_bank = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=f_min, fmax = f_max, htk=True, norm = 1) 
        ## --> if norm = 1 -> filters are normalized to sum to 1
        ## --> if htk = True -> the formula: Mel(f) = 2595 * np.log10(1 + frequency / 700) is used

        ## applying the mel filter bank to the power spectrogram
        D_mel = np.dot(mel_filter_bank, D)
        
        ## if the minimum value for the power and the reference value for dB scaling are given, calculate the dB-scaled mel-spectrogram
        if a_squere_min != None and dB_ref != None:

            D_mel_dB = self.power_mel_to_db(D_mel, a_squere_min=a_squere_min, dB_ref=dB_ref)

        return signal, time, D_mel, D_mel_dB
    ########################################################
        
