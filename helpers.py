import time
import queue as std_queue  # to catch std_queue.Empty safely across modules
from scipy.signal import butter, filtfilt
import numpy as np

def clear_queue(q, block=False, settle_time=0.1) -> int:
    """
    Empties all items from a queue (threading or multiprocessing).

    Parameters:
        q : queue.Queue or multiprocessing.Queue
        block : bool, optional
            If True, waits until queue stops receiving new items.
        settle_time : float, optional
            Time (seconds) to wait to confirm no new items arrive.

    Returns:
        int: number of items removed
    """
    count = 0
    while True:
        try:
            q.get_nowait()
            count += 1
        except std_queue.Empty:
            if not block:
                break
            # Wait briefly and see if new items arrive
            time.sleep(settle_time)
            try:
                q.get_nowait()
                count += 1
                continue
            except std_queue.Empty:
                break
    return count


def butter_lowpass_filter(data, cutoff, fs, order=4) -> np.ndarray:
    '''Lowpass Butterworth filter'''

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def butter_highpass_filter(data, cutoff, fs, order=4) -> np.ndarray:
    '''Highpass Butterworth filter'''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)
