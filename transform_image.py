import os
import numpy as np 
import matplotlib.pyplot as plt 
import neurokit2 as nk
from neurokit2.epochs import epochs_create
from neurokit2.signal import signal_rate
from neurokit2 import ecg_peaks

def calculate_bpm(record) -> int:
    """
    Calculate the heart rate in beats per minute (BPM) from an ECG signal.

    Parameters:
        record (wfdb.Record): The ECG record containing the signal and sampling frequency.

    Returns:
        int: Heart rate in BPM.

    Raises:
        ValueError: If the ECG signal or sampling frequency is invalid.
    """
    ecg_signal = record.p_signal[:, 0]       
    sampfreq = record.fs

    try:
        _, rpeaks = nk.ecg_peaks(ecg_signal, sampfreq)
    except Exception as e:
        raise ValueError(f"Error detecting R-peaks: {str(e)}")

    duration_of_record = len(ecg_signal) / sampfreq
    
    try:
        heart_rate = (len(rpeaks['ECG_R_Peaks']) * 60) / duration_of_record
    except ZeroDivisionError:
        raise ValueError("Sampling frequency is zero or duration of record is zero")

    return int(heart_rate)

def segment_cardiac_epoch(bpm):
    """
    Segments a cardiac epoch based on the given heart rate (bpm).

    Parameters:
        bpm (float): Heart rate in beats per minute.

    Returns:
        Tuple(float, float): Start and end of the cardiac epoch.
    """
    if bpm <= 0:
        raise ValueError("Heart rate (bpm) must be a positive value.")

    epoch_width = bpm / 60
    epoch_start = -0.3 / epoch_width
    epoch_end = 0.45 / epoch_width

    if bpm >= 90:
        c = 0.15
        epoch_start -= c
        epoch_end += c

    return epoch_start, epoch_end

def ecg_segment(ecg_cleaned, bpm, rpeaks=None, sampling_rate=200):
    """
    Segment ECG signal into cardiac epochs based on heart rate.

    Parameters:
        ecg_cleaned (array-like): Cleaned ECG signal.
        bpm (int): Heart rate in beats per minute (BPM).
        rpeaks (array-like, optional): R-peak indices. If not provided, detected automatically.
        sampling_rate (int, optional): Sampling rate of the ECG signal.

    Returns:
        dict: Dictionary containing cardiac epochs.

    Raises:
        ValueError: If the provided heart rate is invalid.
    """
    if rpeaks is None:
        _, rpeaks = ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, correct_artifacts=False)
        rpeaks = rpeaks["ECG_R_Peaks"]

    epochs_start, epochs_end = segment_cardiac_epoch(bpm)

    heartbeats = epochs_create(ecg_cleaned,
                               rpeaks,
                               sampling_rate=sampling_rate,
                               epochs_start=epochs_start, epochs_end=epochs_end)

    return heartbeats

def get_combined_beat_image(signal=None, bpm=None, voltage_range=[-3, 3], folder_name=None, img_name=None):
    """
    Generate a combined beat image from an ECG signal.

    Parameters:
        signal (array-like, optional): ECG signal.
        bpm (int, optional): Heart rate in beats per minute (BPM).
        voltage_range (list, optional): Voltage range for the y-axis. Defaults to [-3, 3].
        folder_name (str, optional): Folder path to save the image. Defaults to None.
        img_name (str, optional): Name of the image file. Defaults to None.

    Returns:
        str: File path of the saved image.

    Raises:
        ValueError: If signal is empty or bpm is not provided.
    """
    # Generate cardiac epochs from the ECG signal
    beats = ecg_segment(signal, bpm, rpeaks=None, sampling_rate=200)
    
    # Create a figure
    fig = plt.figure(num=1, figsize=(5, 8), dpi=300)
    ax = fig.add_subplot(111)
    ax.axis("off")
    
    # Plot each beat
    for key in beats.keys():
        ax.plot(beats[key]["Signal"], color="tab:blue", linewidth=1)
    
    # Set y-axis limits
    ax.set_ylim(voltage_range)
    fig.tight_layout()
    
    # Check if folder exists, if not create it
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    
    # Save the image
    img_path = os.path.join(folder_name, img_name + ".jpg")
    fig.savefig(img_path)
    
    # Clear the axis and close the figure
    ax.clear()
    fig.clf()
    
    return img_path


def get_spectrogram_image(): #spectrogram
    return

def get_rc_image(): ##recurrence plot
    return

def get_autocorrelation_image(): ##autocorrelation plot
    return
