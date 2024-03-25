import os
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

class Record:
    
    """Class representing an ECG record."""
    
    def __init__(self, parent, signal, symbol, aux, sample, label, sf):
        
        """
        Initialize a Record object.

        Args:
            parent (str): The parent of the record.
            signal (np.ndarray): The ECG signal.
            symbol (np.ndarray): Annotation symbols.
            aux (np.ndarray): Auxiliary information.
            sample (np.ndarray): Sample indices of annotations.
            label (str): Label or comment associated with the record.
            sf (int): Sampling frequency of the signal.
        """
        
        self.__parent = parent
        self.__signal = signal
        self.__symbol = symbol
        self.__aux = aux
        self.__sample = sample
        self.__label = label
        self.__sf = sf
        self.__collection = {"signal": self.__signal,
                             "symbol": self.__symbol,
                             "aux": self.__aux,
                             "sample": self.__sample,
                             "label": self.__label,
                             "sampling_frequency": self.__sf,
                             "has_missed_beat": self.has_missed_beat(),
                             "has_unknown_beat": self.has_unknown_beat()}
    
    def __getitem__(self, key):
        return self.__collection[key]
    
    def __str__(self):
        return "Summary\n" + \
               "Size of signal: " + str(len(self.__signal)) + \
               "\nSize of symbol: " + str(len(self.__symbol)) + \
               "\nSize of aux: " + str(len(self.__aux)) + "\n" 
    
    def get_interval(self):
        return (self.__sampfrom, self.__sampto)
    
    def get_indexes_of(self, this=None):
        
        """
        Get indexes of a specific symbol or auxiliary annotation.

        Args:
            this (str): The symbol or annotation to search for.

        Returns:
            np.ndarray: An array of indexes where the symbol or annotation is found.
        """
        
        
        if (len(self.__symbol) > 0) and (len(self.__aux)) > 0:
            if this == "+":
                return np.where(np.asarray(self.__symbol) == "+")
            elif this == "(N":
                return np.where(np.asarray(self.__aux) == '(N')
            elif this=='(AFIB':
                return np.where(np.asarray(self.__aux) == '(AFIB')
            else:
                return print("No Index for this")
            
    def get_intersect_of(self, a, b):
        """
        Get the intersection of two arrays.

        Args:
            a (np.ndarray): First array.
            b (np.ndarray): Second array.

        Returns:
            tuple: A tuple containing the intersection of the arrays and their indices.
        """
        return np.intersect1d(a, b, return_indices=True)
    
    def get_nsr_interval(self):
        rhythm_interval = list()
        
        plus_indexes = self.get_indexes_of('+')
        N_indexes = self.get_indexes_of("(N")
        if not N_indexes[0]:  # This checks if N_indexes list is empty
            print("There ain't rhythm annotation")
            return None
        if len(plus_indexes) != 0 and len(N_indexes) != 0:
            _, a_indexes, b_indexes = self.get_intersect_of(a=N_indexes, b=plus_indexes)
            for i in range(len(b_indexes)-1):
                interval_start = N_indexes[i]
                interval_end = plus_indexes[b_indexes[i]+1]
                interval = (self.__sample[interval_start], self.__sample[interval_end])
                
                rhythm_interval.append(interval)
            if plus_indexes[-1][-1] == N_indexes[-1][-1]:
                interval = (self.__sample[N_indexes[-1][-1]], len(self.__signal))
                rhythm_interval.append(interval)
        return rhythm_interval
    
    def get_afib_interval(self):
        rhythm_interval = list()
        
        plus_indexes = self.get_indexes_of('+') [0]              
        AFIB_indexes = self.get_indexes_of("(AFIB")[0]
        if not AFIB_indexes:  # This checks if N_indexes list is empty
            print("There ain't rhythm annotation")
            return None
        
        
        if len(plus_indexes) != 0 and len(AFIB_indexes) != 0:
            
            _, a_indexes, b_indexes = self.get_intersect_of(a=AFIB_indexes, b=plus_indexes)
            
            for i in range(len(b_indexes)):
                interval_start = AFIB_indexes[i]
                interval_end = plus_indexes[b_indexes[i]+1]
                interval = (self.__sample[interval_start], self.__sample[interval_end])
                rhythm_interval.append(interval)
            # Use .all() if you want to check if all elements are True
            if (plus_indexes[-1] == AFIB_indexes[-1]):
                
                interval = (self.__sample[AFIB_indexes[-1]], len(self.__signal))
                rhythm_interval.append(interval)
        return rhythm_interval


    
    def get_valid_rhythm_interval(self,duration,type=None):
        if type=='NSR':
            rhythm_intervals = self.get_nsr_interval()
        if type=='AF':
            rhythm_intervals=self.get_afib_interval()
        return [itval for itval in rhythm_intervals if self.is_interval_valid(interval=itval,
                                                                           sampling_freq=self.__sf,
                                                                           duration=duration)]
        
    def is_interval_valid(self, interval, sampling_freq, duration):
        return abs(interval[1] - interval[0]) >= (sampling_freq * duration)
    
    def find_index_of_symbol(self, symbol):
        if symbol in self.__symbol:
            return np.where(np.asarray(self.__symbol) == symbol)[0]
        return -1
    
    def find_q_index(self):
        return self.find_index_of_symbol('Q')
    
    def find_quote_index(self):
        return self.find_index_of_symbol('"')
    
    def has_unknown_beat(self):
        return ("Q" in self.__symbol)
    
    def has_missed_beat(self):
        return ('"' in self.__symbol)    
   
    def move_to_any_q_or_quote(self):
        q_index = self.find_q_index()
        quote_index = self.find_quote_index()
        move_index = q_index + quote_index
        return max(move_index)
    
    def move_to_no_pac(self):
        pac_indexes = self.find_index_of_symbol("A")
        return max(pac_indexes)
    
    def move_to_no_pvc(self):
        pvc_indexes = self.find_index_of_symbol("V")
        return max(pvc_indexes)
    
    def has_pac(self):
        return ("A" in self.__symbol)
    
    def has_pvc(self):
        return ("V" in self.__symbol)
    
    def get_pac_percentage(self):
        pac_count = self.get_pac_counts()
        if len(self.__symbol) > 0:
            return (pac_count / len(self.__symbol)) * 100
        else:
            return 0
    
    def get_pvc_percentage(self):
        pvc_count = self.get_pvc_counts()
        if len(self.__symbol) > 0:
            return (pvc_count / len(self.__symbol)) * 100
        else:
            return 0
    
    def is_positive(self, arr_type):
        if arr_type == "PAC":
            percentage = self.get_pac_percentage()
            return ((19 < percentage) and (self.get_pvc_counts() == 0))
        if arr_type == "PVC":
            percentage = self.get_pvc_percentage()
            return ((19 < percentage) and (self.get_pac_counts() == 0))            
        
    def get_pac_counts(self):
        return Counter(self.__symbol)['A']
    
    def get_pvc_counts(self):
        return Counter(self.__symbol)['V']
    
    def get_label(self):
        return self.__label
    
    def get_sampling_frequency(self):
        return self.__sf
    
    def get_duration(self):
        duration = len(self.__signal) / self.__sf
        return duration 
    
    def which(self):
        return self.__parent
    
    def plot_signal_with_annotation(self, ann_style='r.', figsize=(15, 6)):
        """
        Plot the ECG signal with annotations.

        Args:
            ann_style (str): Style of annotation markers.
            figsize (tuple): Size of the figure.

        Returns:
            None
        """
        plot_signal_with_annotation(self.__signal, self.__symbol, self.__sample,
                                    self.__sf, ann_style=ann_style, figsize=figsize)
    
class RecordReader:
    """Class for reading ECG records."""
    
    @classmethod
    def read(cls, path, number, channel, sampfrom, sampto):
        
        """
        Read an ECG record.

        This method reads an ECG record from the specified path, extracts the signal,
        annotations, sample indices, comments, and sampling frequency, and returns
        a Record object representing the record.

        Args:
            path (str): The path to the directory containing the record.
            number (str): The name or identifier of the record.
            channel (int): The channel number of the ECG signal to read.
            sampfrom (int): Starting sample index to read.
            sampto (int): Ending sample index to read.

        Returns:
            Record: A Record object representing the ECG record.

        Raises:
            ValueError: If the specified record file cannot be found or read.
            ValueError: If the specified record annotations cannot be found or read.
        """
        
        
        fullpath = os.path.join(path, number)
        signal = wfdb.rdrecord(fullpath,
                               sampfrom=sampfrom,
                               sampto=sampto).p_signal[:, channel]
        
        ann = wfdb.rdann(fullpath, 'atr',
                            shift_samps=True,
                            sampfrom=sampfrom,
                            sampto=sampto)
        
        symbol = ann.symbol
        aux = ann.aux_note
        sample = ann.sample
        comment = wfdb.rdrecord(fullpath).comments[0]
        sf = wfdb.rdrecord(fullpath).fs
        
        return Record(parent=number,
                      signal=signal,
                      symbol=symbol,
                      aux=aux,
                      sample=sample,
                      label=comment,
                      sf=sf)


def plot_signal_with_annotation(signal,annotation_symbols,annotation_indices,
                                sampling_freq,ann_style='r.',figsize=(15,6)):
        
    #create time axis
    time=np.arange(len(signal))/sampling_freq
    
    pvc_percentage=100*(Counter(annotation_symbols)['V']/Counter(annotation_symbols)['N'])
    pac_percentage=100*(Counter(annotation_symbols)['A']/Counter(annotation_symbols)['N'])
    
    
    
    plt.tight_layout()
    # Create a figure with the desired size
    plt.figure(figsize=figsize)
    plt.ylim(-3,5)
    #plot the signal
    plt.grid(True)
    plt.plot(time,signal)
    
    plt.text(0.5, -0.23,f"PVC Percentage:{pvc_percentage:.2f} \nPAC Percentage:{pac_percentage:.2f}",
             transform=plt.gca().transAxes, ha='center',fontsize=12)
    
    for idx,symbol in zip(annotation_indices,annotation_symbols):
        plt.plot(idx/sampling_freq,signal[idx],ann_style)
        plt.annotate(symbol,(idx/sampling_freq,signal[idx]),xytext=(4,5),textcoords='offset pixels')
        
    #set plot title and label
    #plt.title("Signal with annotation")
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitube (mV)")
    plt.xlim(0,time[-1])