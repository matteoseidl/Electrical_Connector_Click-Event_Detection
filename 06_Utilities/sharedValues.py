import numpy as np

## This class contains the shared values used in the project
class sharedValuesConfig:
    ########################################################
    # Audio processing
    SAMPLING_RATE = 32000 ## sampling rate for the audio files
    CHUNK_SIZE = 4096 ## number of samples processed at a time
    RESOLUTION = 0.016 ## time resolution of the spectrogram
    SPECTROGRAM_COLUMNS_PER_CHUNK = (CHUNK_SIZE/SAMPLING_RATE)/RESOLUTION ## (should be int) number of spectrogram columns in one chunk of audio data
    ## --> with 0.016 seconds resolution, chunk of 4096 samples at a sampling rate of 32 kHz --> 8 spectrogram columns per chunk
    
    HOP_LENGTH = int(RESOLUTION * SAMPLING_RATE) ## the number of samples between successive frames in the STFT
    N_MELS = 128 ## number of mel bands (rows in the mel-scaled spectrogram), 128 is a common value
    F_MIN = 20 # minimum frequency in Hz, taken from the microphone specification
    F_MAX = 14000 # maximum frequency in Hz
    ## --> the microphone has a max frequency response of 20 kHz, however, the sampling rate is 32 kHz, so the max frequency is 16 kHz following the Nyquist–Shannon sampling theorem
    ## --> however the 16 kHz still resulted in distorted values on the upper side of the spectrogram plots
    ## --> therfore, 14 kHz was chosen as the maximum frequency

    DB_REF = 1e3 ## reference value for dB conversion --> value from analysing the dataset
    A_SQUERE_MIN = 1e-12 ## larger than 0 to avoid log(0)  --> value from analysing the dataset
    TOP_DB_ABS = abs(10*np.log10(A_SQUERE_MIN)) ## maximum dB value -> 10*log(a_squere_min) = -120
    
    ########################################################

    ########################################################
    # Dataset generation and detection

    ## short window size for the connectors Ethernet and HVA 630
    WINDOW_SIZE_SEC = (CHUNK_SIZE/SAMPLING_RATE) * 4 ## window size in seconds, has to be larger than the click event duration!!
    ## --> in this case 0.512 seconds, corresponds to 4 chunks with 4096 samples each at a sampling rate of 32 kHz
    WINDOW_SIZE = int(WINDOW_SIZE_SEC/RESOLUTION) ## in this case 32, number of spectrogram columns in the window
    STEP_SIZE_SEC = WINDOW_SIZE_SEC / 4 ## in this case 0.128 seconds --> overlap to avoide missing a click
    STEP_SIZE = int(STEP_SIZE_SEC/RESOLUTION) ## in this case 8, number of spectrogram columns in the step size

    ## long window size for the connector HVA 280
    WINDOW_SIZE_SEC_LONG = WINDOW_SIZE_SEC * 2 ## window size in seconds, has to be larger than the click event duration!!
    ## --> in this case 1.024 seconds, corresponds to 8 chunks with 4096 samples each at a sampling rate of 32 kHz
    WINDOW_SIZE_LONG = int(WINDOW_SIZE_SEC_LONG/RESOLUTION) ## in this case 64, number of spectrogram columns in the window

    ########################################################
    # Audio plotting

    CHUNKS_PER_PLOT = 16
    SAMPLES_PER_PLOT = CHUNK_SIZE * CHUNKS_PER_PLOT
