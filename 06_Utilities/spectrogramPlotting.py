import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

class spectrogramPlotter:

    # Apply global font settings
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 8.5,
        'axes.labelsize': 8.5,
        'axes.titlesize': 8.5,
        'legend.fontsize': 8.5,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8
    })

    global cm_to_inch
    cm_to_inch = 2.54

    ########################################################
    # plot the mel-spectrogram
    def plot_single_wave_and_mel_spectrogram(self, signal, time, D_mel_dB, top_dB_abs, f_min, f_max, n_mels, sampling_rate):

        fig_x = 14/cm_to_inch
        fig_y = 8/cm_to_inch

        fig, axs = plt.subplots(2, 1, figsize=(fig_x, fig_y))

        # plot wave from
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(time, signal)
        #ax1.set_xlim(left=0, right=time[-1])
        ax1.set_ylabel('Amplitude [-]')
        #ax1.set_xlim(left=time[0], right=time[-1])
        ax1.axes.xaxis.set_ticklabels([])
        ax1.yaxis.set_label_coords(-0.117, 0.5) 
        

        # plot mel-spectrogram
        ax2 = plt.subplot(2, 1, 2)
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Frequency [Hz]')

        ax1.set_xlim(left=time[0], right=time[-1])
        #ax2.axes.xaxis.set_ticklabels([]) 
        mel_spec_img = ax2.pcolormesh(np.linspace(0, signal.shape[0] / sampling_rate, D_mel_dB.shape[1]),
                                                np.linspace(f_min, f_max, n_mels), 
                                                D_mel_dB, shading='auto', cmap='inferno')

        mel_spec_img.set_clim(vmin=-top_dB_abs, vmax=0)

        # add colorbar on the bottom of the plot
        cbar_ax = fig.add_axes([0.15, -0.08, 0.7, 0.02])
        fig.colorbar(mel_spec_img, cax=cbar_ax, orientation='horizontal', format="%+2.0f dB")

        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #ax1.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        ax2.set_xticks(np.arange(0, time[-1], 5))

        plt.show()
    ########################################################

    ########################################################
    # plot the signal interval around the peak time of the click event
    def plot_signal_interval(self, signal, time, peak_time, interval):

        fig_x = 14/cm_to_inch
        fig_y = 8/cm_to_inch

        fig, ax = plt.subplots(1, 1, figsize=(fig_x, fig_y))

        ax = plt.subplot(1, 1, 1)
        ax.plot(time, signal)
        #ax1.set_xlim(left=0, right=time[-1])
        ax.set_ylabel('Amplitude [-]')
        ax.set_xlim(left=peak_time-interval, right=peak_time+interval)

        if peak_time is not None:
            ax.axvline(x=peak_time, color='r', linestyle='--', label='Click event')
            ax.legend()
        
        ax.set_xlabel('Time [s]')

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        plt.show()
    
    ########################################################

    ########################################################
    # plot signal interval with 2 peak times
    def plot_signal_interval_with_2_peaks(self, signal, time, peak_time_1, peak_time_2, interval):

        fig_x = 14/cm_to_inch
        fig_y = 8/cm_to_inch
        fig, ax = plt.subplots(1, 1, figsize=(fig_x, fig_y))


        ax = plt.subplot(1, 1, 1)
        ax.plot(time, signal)
        #ax1.set_xlim(left=0, right=time[-1])
        ax.set_ylabel('Amplitude [-]')
        ax.set_xlim(left=(peak_time_1+peak_time_2)/2-interval, right=(peak_time_1+peak_time_2)/2+interval)

        if peak_time_1 is not None:
            ax.axvline(x=peak_time_1, color='limegreen', linestyle='--', label='Click event 2')
            ax.axvline(x=peak_time_2, color='r', linestyle='--', label='Click event 1')
            ax.legend()
        
        ax.set_xlabel('Time [s]')

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        plt.show()


        
        """plt.figure(figsize=(16, 6))
        plt.plot(time, signal)
        print(interval)
        print((peak_time_1+peak_time_2)/2-interval, (peak_time_1+peak_time_2)/2+interval)
        plt.xlim((peak_time_1+peak_time_2)/2-interval, (peak_time_1+peak_time_2)/2+interval)
        
        if peak_time_1 is not None:
            plt.axvline(x=peak_time_1, color='r', linestyle='--', label='Click Peak 1')
            plt.axvline(x=peak_time_2, color='r', linestyle='--', label='Click Peak 2')
            plt.legend()

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Audio Signal with Click Event Peaks")
        plt.tight_layout()
        plt.show()"""
    ########################################################

    ########################################################
    # plot single mel spectrogram
    def plot_single_mel_spectrogram(self, D_mel_dB, top_dB_abs, f_min, f_max, n_mels):

        fig_x = 14/cm_to_inch
        fig_y = 8/cm_to_inch
        fig, ax = plt.subplots(1, 1, figsize=(fig_x, fig_y))

        ax = plt.subplot()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')
        ax.axes.xaxis.set_ticklabels([]) 
        mel_spec_img = ax.pcolormesh(np.linspace(0, D_mel_dB.shape[1], D_mel_dB.shape[1]),
                                                np.linspace(f_min, f_max, n_mels), 
                                                D_mel_dB, shading='auto', cmap='inferno')

        #mel_spec_img.set_clim(vmin=-top_dB_abs, vmax=dB_ref)
        mel_spec_img.set_clim(vmin=-top_dB_abs, vmax=0)

        cbar_ax = fig.add_axes([0.15, -0.05, 0.7, 0.02])
        fig.colorbar(mel_spec_img, cax=cbar_ax, orientation='horizontal', format="%+2.0f dB")

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        plt.show()
    ########################################################

    ########################################################
    # plot multiple spectrograms
    def plot_multiple_mel_spectrograms(self, spectrograms_D_mel_dB, top_dB_abs, f_min, f_max, n_mels, num_spectrograms, num_cols):

        if num_spectrograms % num_cols == 0:
            num_rows = int(num_spectrograms / num_cols)
        else:
            num_rows = int(num_spectrograms / num_cols) + 1

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 2))

        for i in range(num_spectrograms):
            row = i // num_cols
            col = i % num_cols

            if num_rows == 1:
                ax = axs[col]
            else:
                ax = axs[row, col]

            spectrogram_id = i

            spectrogram = spectrograms_D_mel_dB[spectrogram_id]

            #ax = plt.subplot()
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Frequency [Hz]')
            ax.axes.xaxis.set_ticklabels([]) 
            mel_spec_img = ax.pcolormesh(np.linspace(0, spectrogram.shape[1], spectrogram.shape[1]),
                                                    np.linspace(f_min, f_max, n_mels), 
                                                    spectrogram, shading='auto', cmap='inferno')

            #mel_spec_img.set_clim(vmin=-top_dB_abs, vmax=dB_ref)
            mel_spec_img.set_clim(vmin=-top_dB_abs, vmax=0)

        cbar_ax = fig.add_axes([0.15, -0.05, 0.7, 0.02])
        fig.colorbar(mel_spec_img, cax=cbar_ax, orientation='horizontal', format="%+2.0f dB")

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        plt.tight_layout()
        plt.show()

