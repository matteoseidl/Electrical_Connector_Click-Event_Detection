import numpy as np
import matplotlib.pyplot as plt

class spectrogramPlotter:

    ########################################################
    # plot the mel-spectrogram
    def plot_single_wave_and_mel_spectrogram(self, signal, time, D_mel_dB, top_dB_abs, f_min, f_max, n_mels, sampling_rate):

        fig_x = 16
        fig_y = 6
        fig, axs = plt.subplots(2, 1, figsize=(fig_x, fig_y))

        # plot wave from
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(time, signal)
        ax1.set_xlim(left=0, right=time[-1])
        ax1.set_ylabel('Amplitude')
        ax1.set_xlim(left=time[0], right=time[-1])

        # plot mel-spectrogram
        ax2 = plt.subplot(2, 1, 2)
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Frequency [Hz]')
        ax2.axes.xaxis.set_ticklabels([]) 
        mel_spec_img = ax2.pcolormesh(np.linspace(0, signal.shape[0] / sampling_rate, D_mel_dB.shape[1]),
                                                np.linspace(f_min, f_max, n_mels), 
                                                D_mel_dB, shading='auto', cmap='inferno')

        mel_spec_img.set_clim(vmin=-top_dB_abs, vmax=0)

        # add colorbar on the bottom of the plot
        cbar_ax = fig.add_axes([0.15, -0.05, 0.7, 0.02])
        fig.colorbar(mel_spec_img, cax=cbar_ax, orientation='horizontal', format="%+2.0f dB")

        plt.show()
    ########################################################

