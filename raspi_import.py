import numpy as np
import sys
import matplotlib.pyplot as plt

def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.

    Returns sample period and a (`samples`, `channels`) `float64` array of
    sampled data from all `channels` channels.

    Example (requires a recording named `foo.bin`):
    ```
    >>> from raspi_import import raspi_import
    >>> sample_period, data = raspi_import('foo.bin')
    >>> print(data.shape)
    (31250, 5)
    >>> print(sample_period)
    3.2e-05

    ``` """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        # The "dangling" `.astype('float64')` casts data to double precision
        # Stops noisy autocorrelation due to overflow
        data = data.reshape((-1, channels))

    # sample period is given in microseconds, so this changes units to seconds
    sample_period *= 1e-6
    return sample_period, data

# Import data from bin file
if __name__ == "__main__":
    sample_period, data = raspi_import('foo.bin')

    num_sensors = 5
    NFFT= 31250
    padded_NFFT = NFFT*10
    
    # Create a figure for subplots
    plt.figure(figsize=(10, 10))

    def channeldata(data):
        plt.plot(channel_data)
        plt.xlabel('Tid [ms]')
        plt.ylabel('Amplitude [V]')
        plt.title(f"ADC-data {sensor}")

        plt.xlim(0,200)
        return 0
    
    def compute_fft(data, NFFT, sample_period): 
        
        fft_result = np.fft.fftshift(np.fft.fft(data, NFFT))
        fft_result = np.abs(fft_result)

        fft_result = 20* np.log10(fft_result + 1e-10) #dB
        fft_result -= np.max(fft_result)
         
        frequency_axis = np.fft.fftshift(np.fft.fftfreq(NFFT, sample_period))
        plt.plot(frequency_axis, fft_result)
        plt.title(f"FFT av ADC-data {sensor}")
        plt.xlabel("Frekvens [Hz]")
        plt.ylabel("Forsterkning [dB]")
        plt.xlim(-2200,2200)
        return frequency_axis, fft_result

    def psd (data, NFFT, sample_period):
        
        fft_result = np.fft.fft(data, NFFT)
        fft_result = np.fft.fftshift(fft_result)
        fft_axis = np.fft.fftshift(np.fft.fftfreq(NFFT,sample_period))

        S_X = np.abs(fft_result) **2
        S_X_db = 20* np.log10(S_X)

        S_X_db -= np.max(S_X_db)

        plt.plot(fft_axis,S_X_db)
        plt.title(f"Effekttetthetsspekter (Periodogram)")
        plt.xlabel("Frekvens [Hz]")
        plt.ylabel("Effekttetthet [dB]")
        plt.xlim(950,1050)
        plt.ylim(-150,0)
        return 0
    def hanning_window(data, NFFT, sample_period):  #Mulig at å bruke NFFT_padded er best når funksjonen kalles på
        
        fft_non_hanning = np.fft.fftshift(np.fft.fft(data, NFFT))
        hanning_window = np.hanning(len(data))
        data_windowed = data*hanning_window
        fft_result = np.fft.fftshift(np.fft.fft(data_windowed, NFFT))

        fft_windowed_abs = np.abs(fft_result) 
        fft_non_hanning = 20*np.log10(np.abs(fft_non_hanning)) 
        fft_non_hanning = fft_non_hanning -np.max(fft_non_hanning) 

        fft_windowed = 20*np.log10(fft_windowed_abs) 
        fft_windowed = fft_windowed -np.max(fft_windowed)

        frequency_axis = np.fft.fftshift(np.fft.fftfreq(NFFT, sample_period))

        plt.plot(frequency_axis, fft_windowed, label = 'Med vindu')
        plt.plot(frequency_axis,fft_non_hanning, label = 'Uten vindu')
        plt.legend()
        plt.title(f"FFT med og uten Hanningvindu")
        plt.xlabel("Frekvens [Hz]")
        plt.ylabel("Forsterkning [dB]")
        plt.xlim(950,1050)
        #plt.ylim(-0.05,0.05)
        return 0


    for sensor in range(num_sensors):
        channel_data = data[0:, sensor]
        channel_data = channel_data*0.000806 

        
        plt.subplot(num_sensors, 1, sensor + 1)

        #channeldata(channel_data)
        #compute_fft(channel_data,NFFT,sample_period)
        psd(channel_data,padded_NFFT, sample_period)
        #hanning_window(channel_data,padded_NFFT,sample_period)

plt.tight_layout()
plt.show()



