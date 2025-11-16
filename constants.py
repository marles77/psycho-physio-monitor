# ==========================================================
# Constants
# ==========================================================

import os

class Constants:
    APP_TITLE = "PsychoPhysio Monitor"
    APP_THEME = "litera"
    CURR_DIR = os.getcwd()
    BG_COLOR = "#CFD8DC"
    PLOT_COLOR = "#002699"
    PORTS = ['COM1', 'COM2', 'COM3', 'COM4', 'COM5']
    BAUDRATES = [9600, 19200, 38400, 57600, 115200]
    #BATCH_SIZE = 100 # 20 packets (10 data points for ECG) * 5 seconds
    PACK_RATE = 20  # packets per second
    SAMPLE_COUNT_ECG = 10 # samples in a packet
    SAMPLE_COUNT_GSR = 1  # samples in a packet
    NBYTES = 2 # bytes per data point (uint16)
    WINDOW_SIZE = (1920, 1024)
    ROW_MIN_SIZE = 1000
    BUTTON_WIDTH = 30
    EPOCH_TIME = 5  # seconds
    EPOCH_NUM = 100 # max number of epochs
    DIVIDER_ECG = 1000 # divider for raw ecg data (raw plot)
    DIVIDER_GSR = 100  # divider for raw gsr data (raw plot)
    MULTIPLIER_HR = 200.  # multiplier for hr data (agg plot)
    MULTIPLIER_GSR = 100. # multiplier for gsr data (agg plot)
    FILT_CUTOFF_LOW = 45 # butter filter
    FILT_CUTOFF_HIGH = 1 # butter filter
    RAW_PLOT_RATE = 50    # milliseconds
    FILT_PLOT_RATE = 500  # milliseconds
    MODES_COMMANDS = {'gsr': 'a',
                      'ecg': 'b',
                      'synth_ecg': 'c',
                      'synth_gsr': 'd'}
    
