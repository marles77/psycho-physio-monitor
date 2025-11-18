# ==========================================================
# Helper functions
# ==========================================================

import os
import io
import sys
import time
from scipy.signal import butter, filtfilt
import numpy as np
from constants import Constants
from logger import logger
from contextlib import contextmanager
from matplotlib.figure import Figure
from queue import Queue, Empty
from threading import Event
import heartpy as hp
from pyEDA.main import process_statistical


# Processing worker (process)

def processing_worker_analysis(in_q: Queue, out_q: Queue, stop_event: Event, pack_rate: int, time_window: int, sample_count: int, mode: str) -> None:
    """Process data"""
    
    batch_size: int = pack_rate * time_window  # e.g., 20 packets/sec * 5 sec = 100 packets
    fs: float = float(pack_rate * sample_count)  # Sampling frequency
    buffer = np.zeros((batch_size, sample_count), dtype=np.uint16) if mode in ('ecg', 'synth_ecg') else np.zeros((batch_size, sample_count), dtype=np.float32)
    #idx = 0
    try:
        while not stop_event.is_set():
            # Wait until all packets are available
            if in_q.qsize() < batch_size:
                #time.sleep(0.05)
                continue

            # Retrieve exactly batch-size packets
            packets = []
            for _ in range(batch_size):
                try:
                    pkt = in_q.get(block=True, timeout=0.5) # wait for data - important! Otherwise we may get not enough packets
                    packets.append(pkt)
                except Exception:
                    break

            # Check if we got the full batch
            if len(packets) < batch_size:
                print(f"Not enough packets retrieved: {len(packets)} < {batch_size}")
                #continue

            # Fill the buffer
            buffer[:] = np.vstack(packets)

            # --- Processing step ---
            
            filtered = buffer.flatten()
            
            if mode in ('ecg', 'synth_ecg'):
                # Apply bandpass filter for ECG
                cutoff_low = Constants.FILT_CUTOFF_LOW
                filtered_ecg = butter_lowpass_filter(filtered, cutoff_low, fs)
                filtered_ecg = hp.remove_baseline_wander(filtered_ecg, fs)
                working_data, measures = hp.process(filtered_ecg, fs)
                fig = hp.plotter(working_data, measures, show=False, title='')
                fig.tight_layout()
                fig.set_size_inches(17, 3, forward=True)

            elif mode in ('gsr', 'synth_gsr'):
                with suppress_print():
                    # needed because scipy prints dataframes automatically to console
                    measures, wd, eda_clean = process_statistical(filtered, use_scipy=True, sample_rate=fs, new_sample_rate=fs, segment_overlap=0)
                fig = Figure(figsize=(16, 3), dpi=100, tight_layout=True)
                ax = fig.add_subplot(111)
                print(f"Shape of data: {eda_clean[0].shape[0]}")
                ax.set_xlabel("time")
                ax.plot(eda_clean[0])
                ax.vlines(wd['indexlist'][0], ax.get_ylim()[0], ax.get_ylim()[1], colors='red')
                ax.set_xticks(np.linspace(0, eda_clean[0].shape[0], time_window+1))
                ax.set_xticklabels([f"{t:.0f}" for t in np.linspace(0, time_window, time_window+1)])
                 
            else:
                pass

            if fig:       
                fig.set_dpi(100)
                fig.canvas.draw()
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
                buf.seek(0)
                
            if buf and measures:
                out_q.put([buf.getvalue(), measures]) #, filtered])
            else:
                print("Worker: brak danych")

            buffer.fill(0)
            time.sleep(0.5)
    except (Warning, Exception) as e:
        logger.info(f"Processing worker error: {e}")

    logger.info("Processing worker stopped.")
    print("Processing worker stopped.")


# Signal filters

def butter_lowpass_filter(data: np.ndarray, cutoff: int, fs: float, order=4) -> np.ndarray:
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


# Other helper functions

@contextmanager
def suppress_print():
    '''This will suppress printing to console'''

    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


def clear_queue(q: Queue, block=False, settle_time: float=0.1) -> int:
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
        except Empty:
            if not block:
                break
            # Wait briefly and see if new items arrive
            time.sleep(settle_time)
            try:
                q.get_nowait()
                count += 1
                continue
            except Empty:
                break
    return count


