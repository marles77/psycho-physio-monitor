#
import os
import sys
#import logging
from logger import logger
import time
import tkinter as tk
from tkinter import filedialog, Menu
from tkinter.messagebox import showinfo
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.style import Bootstyle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseButton
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import serial
from threading import Thread, Event
from queue import LifoQueue, Queue, Empty
import multiprocessing as mp
from typing import Union, Any
from dataclasses import dataclass
import struct
from collections import deque
from helpers import clear_queue, butter_lowpass_filter, butter_highpass_filter
import heartpy as hp
from pyEDA.main import process_statistical

# ==========================================================
# Settings and Constants
# ==========================================================



class Constants:
    CURR_DIR = os.getcwd()
    BG_COLOR = "#CFD8DC"
    PLOT_COLOR = "#002699"
    PORTS = ['COM1', 'COM2', 'COM3', 'COM4', 'COM5']
    BAUDRATES = [9600, 19200, 38400, 57600, 115200]
    BATCH_SIZE = 100 # 20 packets (10 data points for ECG) * 5 seconds
    PACK_RATE = 20  # packets per second
    SAMPLE_COUNT_ECG = 10
    SAMPLE_COUNT_GSR = 1
    NBYTES = 2 # bytes per data point (uint16)
    WINDOW_SIZE = (1800, 900)
    ROW_MIN_SIZE = 1000
    BUTTON_WIDTH = 30
    TIME_WINDOW = 5  # seconds
    DIVIDER_ECG = 1024
    FILT_CUTOFF_LOW = 45
    FILT_CUTOFF_HIGH = 1

@dataclass
class SpinValue:
    plot1_low: int = 100
    plot1_up: int = 500
    plot2_low: int = 45
    plot2_up: int = 55
    plot3_low: int = 45
    plot3_up: int = 55


# ==========================================================
# Serial Reader (thread)
# ==========================================================

class SerialReader:
    def __init__(self, 
                 port: str = Constants.PORTS[2], 
                 baudrate: int = Constants.BAUDRATES[4], 
                 sample_count: int = Constants.SAMPLE_COUNT_ECG,
                 packet_size: int = Constants.SAMPLE_COUNT_ECG * Constants.NBYTES, 
                 bytes=Constants.NBYTES, 
                 raw_queue=None,
                 proc_queue=None,
                 run_event=None,
                 pause_event=None) -> None:
        self.port = port
        self.baudrate = baudrate
        self.sample_count = sample_count
        self.packet_size = packet_size #sample_count * bytes
        self.ser = None
        self.is_running = False
        self.raw_queue = raw_queue # queue to store raw data packets
        self.proc_queue = proc_queue # queue to send data to processing worker
        self.run_event = run_event if run_event else Event()
        self.pause_event = pause_event or Event()
        self.thread = None


    def set_port_baud(self, port: str, baudrate: int) -> None:
        self.port = port
        self.baudrate = baudrate


    def start(self) -> None:
        #print(f"Running: {self.is_running}")
        if not self.is_running:
            res = self._open_connection()
            return res
        return False
            


    def stop(self) -> None:
        self._close_connection()


    def pause(self) -> None:
        #self.stop_event.set()
        self.run_event.clear()
        self.is_running = False

    
    def _open_connection(self) -> None:
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.thread = Thread(target=self._read_data, daemon=True)
        except serial.SerialException as e:
            print(f"Serial error: {e}")
            logger.info(f"Failed to open serial port {self.port}: {e}")
            return False
        if self.ser:
            logger.info(f"Opened serial port {self.port} at {self.baudrate} baudrate.")
            return self.ser.is_open
        return False
    

    def _close_connection(self):
        #self.stop_event.set()
        self.run_event.clear()
        self.is_running = False
        if self.ser and self.ser.is_open:
            self.ser.write("x".encode())
            time.sleep(0.5)
            self.ser.close()


    def _read_data(self) -> None:
        '''Read data from serial port in a separate thread.'''

        self.is_running = True
        if self.ser:
            self.ser.write("c".encode())
            time.sleep(0.5)

            try:
                while self.run_event.is_set():
                        #print("Reading data...")
                        #if self.ser.in_waiting:
                        if self.pause_event.is_set():
                            # paused — do not read
                            time.sleep(0.1)
                            continue
                        raw = self.ser.read(self.packet_size)
                        #print(f"Raw data length: {len(raw)}")
                        if len(raw) == self.packet_size:
                            vals = struct.unpack('<' + 'H'*self.sample_count, raw)
                            pkt = np.array(vals, dtype=np.uint16)
                            #print(f"Read packet: {pkt}")
                            if self.raw_queue:
                                self.raw_queue.put(pkt)
                            if self.proc_queue:
                                self.proc_queue.put(pkt)
                            #print(f"Raw: {self.raw_queue.qsize()}, Proc queue size: {self.proc_queue.qsize()}")
                    
            except (serial.SerialException, KeyboardInterrupt, Exception) as e:
                logger.info(f"Unexpected error: {e}")
                self.running = False
                self._close_connection()
            finally:
                logger.info("SerialReader stopped.")
        else:
            logger.info("Serial port not open.")

    
    def _clear_queue(self, queue) -> None:
        '''Clear all items in the queue.'''
        removed = clear_queue(queue, block=True)
        logger.info(f"Cleared {removed} items from the queue.")


# ==========================================================
# Processing worker (process)
# ==========================================================

def processing_worker_analysis(in_q: Queue, out_q: Queue, stop_event: Event, batch_size: int, sample_count: int, mode: str) -> None:
    """Process data"""
    
    buffer = np.zeros((batch_size, sample_count), dtype=np.uint16)
    #idx = 0

    while not stop_event.is_set():
        # Wait until at least 100 packets are available
        #print(f"Proc size: {in_q.qsize()}")
        if in_q.qsize() < batch_size:
            # small sleep to avoid busy waiting
            time.sleep(0.1)
            continue

        # Retrieve exactly 100 packets
        packets = []
        for _ in range(batch_size):
            try:
                pkt = in_q.get_nowait()
                packets.append(pkt)
            except Exception:
                break

        # Check if we got the full batch
        if len(packets) < batch_size:
            # not enough data (another thread took some?)
            continue

        # Fill the buffer
        buffer[:] = np.vstack(packets)

        # --- Processing step ---
        #print(f"Processing {len(packets)} packets...")
        filtered = buffer.flatten()
        if mode == 'ecg':
            # Apply bandpass filter for ECG
            fs = 100.0 # Constants.PACK_RATE * Constants.SAMPLE_COUNT_ECG  # Sampling frequency
            cutoff_low = Constants.FILT_CUTOFF_LOW  # Low cutoff frequency
            cutoff_high = Constants.FILT_CUTOFF_HIGH  # High cutoff frequency

            # Lowpass filter
            filtered_ecg = butter_lowpass_filter(filtered, cutoff_low, fs)
            filtered_ecg = hp.remove_baseline_wander(filtered_ecg, fs)
            working_data, measures = hp.process(filtered_ecg, fs)
            print(f"BPM: {measures['bpm']}") #returns BPM value
            print(f"RMSSD: {measures['rmssd']}") # returns RMSSD HRV measure
            #logger.info(f"Measures: {measures}")
            

        # Send to GUI or another consumer
        out_q.put(filtered)
        buffer.fill(0)
        time.sleep(0.5)

    logger.info("Processing worker stopped.")
    print("Processing worker stopped.")


# ==========================================================
# GUI Application
# ==========================================================            
class App(ttk.Window):
    def __init__(self) -> None:
        super().__init__(title="CogSci Monitor", themename="litera", size=Constants.WINDOW_SIZE, resizable=(1, 1))
        self.protocol('WM_DELETE_WINDOW', self.on_close)

        self.port_index = 2
        self.baud_index = 4

        # --- Thread and Process communication ---
        self.conn = None
        self.raw_queue = Queue()
        self.proc_in = mp.Queue()
        self.proc_out = mp.Queue()
        self.run_event = Event()
        self.pause_event = Event()
        self.stop_event = mp.Event()
        self.proc = None

        # --- Data buffers --- 
        self.sample_count = 0
        self.sample_rate = 0
        self.buffer_duration = Constants.TIME_WINDOW  # seconds visible on raw plot
        self.buffer = None # deque(maxlen=self.sample_rate * self.buffer_duration)
        self.filtered_data = np.array([], dtype=np.uint16)

        # configure window grid
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        #self.frame.pack(fill=BOTH, expand=YES)

        self.frame = ttk.Frame(master=self, padding=10)
        self.frame.grid(column=0, row=0, sticky=NSEW)

        # configure frame grid
        for i in range(6):
            self.frame.columnconfigure(i, weight=1, uniform="equal")
        
        self.frame.rowconfigure(9, minsize=Constants.ROW_MIN_SIZE)

        # set style for all ttk widgets
        self.frame.style = ttk.Style()
        self.frame.style.configure('.', font=('Consolas', 11))

        self.spin_value = SpinValue()

        self._create_widgets(master=self.frame)
        #self.create_menu(master=self.frame)

        self.serial_reader = SerialReader(
                                          raw_queue=self.raw_queue,
                                          proc_queue=self.proc_in, 
                                          run_event=self.run_event, 
                                          pause_event=self.pause_event,
                                          sample_count=0
                                          )

        self.after(50, self.update_raw_plot)
        self.after(500, self.update_filtered_plot)
        


    def _create_widgets(self, master) -> None:
        
        self.label_head = ttk.Label(master=master, text="Welcome to CogSci Monitor")
        self.label_head.grid(column=0, row=0, sticky=W, padx=1, pady=1, columnspan=6)
        #self.label.pack(pady=20)

        self.button_conn = ttk.Button(master=master, text="Connect", bootstyle="success", command=self.on_button_conn_click, width=Constants.BUTTON_WIDTH)
        self.button_conn.grid(column=0, row=1, sticky=NSEW, padx=1, pady=1)

        self.button_gsr = ttk.Button(master=master, text="Restart GSR", bootstyle="info", command=self.on_button_gsr_click, width=Constants.BUTTON_WIDTH, state=DISABLED)
        self.button_gsr.grid(column=1, row=1, sticky=NSEW, padx=1, pady=1)

        self.button_ecg = ttk.Button(master=master, text="Restart ECG", bootstyle="info", command=self.on_button_ecg_click, width=Constants.BUTTON_WIDTH, state=DISABLED)
        self.button_ecg.grid(column=2, row=1, sticky=NSEW, padx=1, pady=1)

        self.button_stop = ttk.Button(master=master, text="Stop", bootstyle="warning", command=self.on_button_stop_click, width=Constants.BUTTON_WIDTH)
        self.button_stop.grid(column=3, row=1, sticky=NSEW, padx=1, pady=1)

        self.cbo_port = ttk.Combobox(master=master, text="Select COM Port", values=Constants.PORTS, exportselection=False, width=Constants.BUTTON_WIDTH)
        self.cbo_port.grid(column=4, row=1, sticky=NSEW, padx=1, pady=1)
        self.cbo_port.current(self.port_index)

        self.cbo_baud = ttk.Combobox(master=master, text="Select Baudrate", values=Constants.BAUDRATES, exportselection=False, width=Constants.BUTTON_WIDTH)
        self.cbo_baud.grid(column=5, row=1, sticky=NSEW, padx=1, pady=1)
        self.cbo_baud.current(self.baud_index)

        # buttons to control axes limits

        self.spinbox_plot1_up = ttk.Spinbox(master=master, from_=0, to=1000)
        self.spinbox_plot1_up.grid(column=0, row=2, sticky=W, padx=1, pady=1)
        self.spinbox_plot1_low = ttk.Spinbox(master=master, from_=0, to=1000)
        self.spinbox_plot1_low.grid(column=0, row=3, sticky=W, padx=1, pady=1)
        self.spinbox_plot1_up.set(self.spin_value.plot1_up)
        self.spinbox_plot1_low.set(self.spin_value.plot1_low)

        self.trim_low = 0
        self.trim_up = 0

        self.spinbox_plot2_up = ttk.Spinbox(master=master, from_=0, to=1000)
        self.spinbox_plot2_up.grid(column=0, row=4, sticky=W, padx=1, pady=1)
        self.spinbox_plot2_low = ttk.Spinbox(master=master, from_=0, to=1000)
        self.spinbox_plot2_low.grid(column=0, row=5, sticky=W, padx=1, pady=1)
        self.spinbox_plot2_up.set(self.spin_value.plot2_up)
        self.spinbox_plot2_low.set(self.spin_value.plot2_low)

        self.spinbox_plot3_up = ttk.Spinbox(master=master, from_=0, to=1000)
        self.spinbox_plot3_up.grid(column=0, row=6, sticky=W, padx=1, pady=1)
        self.spinbox_plot3_low = ttk.Spinbox(master=master, from_=0, to=1000)
        self.spinbox_plot3_low.grid(column=0, row=7, sticky=W, padx=1, pady=1)
        self.spinbox_plot3_up.set(self.spin_value.plot3_up)
        self.spinbox_plot3_low.set(self.spin_value.plot3_low)

        self.label_bottom = ttk.Label(master=master, text=f"Status: {'Disconnected'}")
        self.label_bottom.grid(column=0, row=8, sticky=W, padx=1, pady=1, columnspan=4)

        self.label_val1 = ttk.Label(master=master, text="N/A")
        self.label_val1.grid(column=4, row=8, sticky=W, padx=1, pady=1, columnspan=1)

        self.label_val2 = ttk.Label(master=master, text="N/A")
        self.label_val2.grid(column=5, row=8, sticky=W, padx=1, pady=1, columnspan=1)

        # --- Raw plot ---
        self.fig1 = Figure(figsize=(12, 2), dpi=100)
        self.ax1 = self.fig1.add_subplot()
        self.raw_line = self.ax1.plot([], [], lw=1, color = 'g')[0]
        #self.ax1.set_title("Live Raw Data")
        self.ax1.set_autoscaley_on(False) 
        #self.ax1.set_ylim(self.spinbox_plot1_low.get(), self.spinbox_plot1_up.get())
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=master)
        self.canvas1.get_tk_widget().grid(column=1, row=2, columnspan=5, rowspan=2, sticky=NSEW)

        # --- Filtered plot ---
        self.fig2 = Figure(figsize=(12, 2), dpi=100)
        self.ax2 = self.fig2.add_subplot()
        #self.filtered_line, = self.ax2.plot([], [], color='orange', lw=1.5)
        #self.ax2.set_title("Filtered Data (5 s updates)")
        self.ax2.set_ylim(self.spinbox_plot2_low.get(), self.spinbox_plot2_up.get())
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=master)
        self.canvas2.get_tk_widget().grid(column=1, row=4, columnspan=5, rowspan=2, sticky=NSEW)


    def _create_menu(self, master) -> None:
        menubar = ttk.Menu(master=master)
        #file_menu = ttk.Menu(master=menubar)
        #file_menu.add_command(label="Open", command=self.open_file)
        menubar.add_command(label="GSR")
        menubar.add_command(label="ECG")
        menubar.add_command(label="Exit", command=self.on_close)
        #menubar.add_cascade(label="File", menu=file_menu)

        mb = ttk.Menubutton(
            master=master,
            text="Menu",
            bootstyle=SECONDARY, 
            menu=menubar
        )
        mb.grid(column=3, row=1, sticky=NE, padx=1, pady=1)


    def on_button_conn_click(self) -> None:
        if not self.serial_reader.ser or not self.serial_reader.ser.is_open:
            selected_port = self.cbo_port.get()
            baudrate = int(self.cbo_baud.get())
            self.serial_reader.set_port_baud(selected_port, baudrate)
            #showinfo("Information", f"Set port to {selected_port} with baudrate {baudrate}.")
            self.conn = self.serial_reader.start()
            if self.conn:
                self.button_conn.config(bootstyle="danger", text="Disconnect")
                self.label_bottom.config(text=f"Status: Connected to {selected_port} at {baudrate} baudrate")
                #showinfo("Information", f"Serial port {self.serial_reader.port} opened successfully.")
                time.sleep(1)
                self.button_ecg.config(state=NORMAL)
            else:
                showinfo("Error", f"Failed to open serial port {self.serial_reader.port}.")
        else:
            if self.serial_reader.thread.is_alive():
                self.serial_reader.thread.join(timeout=1)
            if self.proc:
                self.proc.join(timeout=1)
                self.proc = None
            self.serial_reader.stop()
            clear_queue(self.raw_queue, block=True)
            clear_queue(self.proc_in, block=True)
            clear_queue(self.proc_out, block=True)
            self.button_conn.config(bootstyle="success", text="Connect")
            self.label_bottom.config(text="Status: Disconnected")
            print("Disconnected from serial port.")
            self.button_ecg.config(state=DISABLED)

    

    def on_button_gsr_click(self) -> None:
        showinfo("Information", "Button GSR was clicked!")


    def on_button_ecg_click(self) -> None:
        #showinfo("Information", "Button ECG was clicked!")
        if self.conn:
            self.sample_count = Constants.SAMPLE_COUNT_ECG
            self.sample_rate = Constants.SAMPLE_COUNT_ECG * Constants.PACK_RATE
            self.buffer = np.zeros((Constants.PACK_RATE * Constants.SAMPLE_COUNT_ECG * Constants.TIME_WINDOW)) #deque(maxlen=self.sample_rate * self.buffer_duration)
            self.buffer_idx = 0
            self.ax1.set_xlim(0, self.buffer.shape[0])
            low = int(self.spinbox_plot1_low.get())
            up = int(self.spinbox_plot1_up.get())
            self.trim_low = low / Constants.DIVIDER_ECG
            self.trim_up = (Constants.DIVIDER_ECG - up) / Constants.DIVIDER_ECG
            self.ax1.set_ylim(low, up)
            # y = np.random.randint(100, 500, size=(self.buffer.shape[0],))
            # x = np.arange(y.shape[0])
            # y_norm = self._normalize_to_range(y, 0, 1)
            # print(y_norm)
            self.ax1.vlines(200, 0, 0.5, color='black', lw=0.5)
            # self.raw_line.set_data(x, y_norm)
            
            # self.ax1.set_xlim(max(0, len(y) - len(self.buffer)), len(y))
            
            self.canvas1.draw_idle()
            self.canvas1.flush_events()
            #logger.info(f"Initial data x: {x}, y: {y}.")
            print("Starting acquisition of ECG data...")
            self.run_event.set()
            self.pause_event.clear()
            self.stop_event.clear()
            self.serial_reader.sample_count=Constants.SAMPLE_COUNT_ECG
            self.serial_reader.packet_size=Constants.SAMPLE_COUNT_ECG * Constants.NBYTES
            self.serial_reader.thread.start()
            self.proc = mp.Process(target=processing_worker_analysis, 
                                   args=(self.proc_in, self.proc_out, self.stop_event), 
                                   kwargs={'batch_size': Constants.BATCH_SIZE, 'sample_count': Constants.SAMPLE_COUNT_ECG, 'mode': 'ecg'}, 
                                   daemon=True)
            self.proc.start()
            
        

    def on_button_stop_click(self) -> None:
        print("Paused")
        self.run_event.clear()
        self.pause_event.clear()
        self.stop_event.set()
        self.serial_reader.pause()
        #self.pause_event.set()
        # clear_queue(self.raw_queue, block=True)
        # clear_queue(self.proc_in, block=True)
        # clear_queue(self.proc_out, block=True)
        if self.serial_reader.thread.is_alive():
            self.serial_reader.thread.join(timeout=1)
        if self.proc:
            self.proc.join(timeout=1)
            self.proc = None
        print("Stopped cleanly.")

    
    #TODO: not implemented yet
    def resume(self):
        print("Resuming reader…")
        self.pause_event.clear()



    def on_close(self) -> None:
        '''
        Called when close icon clicked
        Safely kill the app
        '''

        # update events
        # self.close_event.set()
        # self.stop_event.set()
        # self.pause_event.clear()

        # make sure that tkinter loop is stopped
        # try:
        #     self.after_cancel(self.after_id)
        # except (AttributeError, Exception) as e:
        #     print(f"Error: {e}. Quitting anyway.")

        self.serial_reader.stop()
        time.sleep(0.1)
        self.quit()
        self.destroy()
        
        print(f"App closed")


    # def check_queue(self) -> None:
    #     """Update label if new serial data arrived."""
    #     q = self.serial_reader.queue
    #     if not q.empty():
    #         data = q.get()
    #         self.label_val.configure(text=data)
    #     self.after(100, self.check_queue)

    # ======================================================
    # Plot update loops
    # ======================================================
    #
    def update_raw_plot(self):
        """Continuously updates live waveform"""
        new_data = False
        #print(self.buffer)
        if (self.buffer is not None) and (self.run_event.is_set()):
            while not self.raw_queue.empty():
                    pkt = self.raw_queue.get_nowait()
                    pkt_len = len(pkt)
                    self.buffer[self.buffer_idx : self.buffer_idx + pkt_len] = pkt
                    self.buffer_idx += pkt_len
                    #logger.info(f"Buffer idx: {self.buffer_idx}, Packet len: {pkt_len}, Packet data: {pkt}\n")
                    new_data = True

            if new_data:
                y = self.buffer[:self.buffer_idx]
                y_norm = y / Constants.DIVIDER_ECG #self._normalize_to_display(y, 0.2, 0.8) #_normalize_to_range(y, 0, 1)
                x = np.arange(y.shape[0])
                self.raw_line.set_data(x, y_norm)
                self.ax1.set_ylim(self.trim_low, 1 - self.trim_up)
                self.canvas1.draw_idle()
                self.canvas1.flush_events()
                self.label_val1.configure(text=f"Len: {x.shape[0]}")
                #logger.info(f"Min: {y_norm.min()}, Max: {y_norm.max()}")
            if self.buffer_idx >= self.buffer.shape[0] - self.sample_count:
                self.buffer.fill(0)
                self.buffer_idx = 0  # reset index when buffer is full

        self.after(50, self.update_raw_plot)


    def update_filtered_plot(self):
        """Updates filtered plot every 5 seconds"""
        new_data = False
        if self.run_event.is_set():
            while not self.proc_out.empty():
                self.filtered_data = self.proc_out.get_nowait()
                new_data = True
                

            if new_data:
                #x = np.arange(len(self.filtered_data))
                # self.filtered_line.set_data(x, self.filtered_data)
                # self.ax2.set_xlim(0, len(x))
                # self.ax2.set_ylim(self.filtered_data.min(), self.filtered_data.max())
                # self.canvas2.draw_idle()
                self.label_val2.configure(text=f"Filtered Len: {len(self.filtered_data)}")
                logger.info(f"Filtered data shape: {self.filtered_data.shape}\nData: {self.filtered_data}")
        

        self.after(500, self.update_filtered_plot)

    # ====================not used==================================
    def _normalize_to_range(self, y, lo=0.2, hi=0.5):
        #y = np.astype(float)
        ymin = y.min()
        ymax = y.max()
        #print(f"ymin: {ymin}, ymax: {ymax}")
        if ymax == ymin:
            # constant array — map to mid-range
            return np.full_like(y, (lo + hi) / 2.0)
        return lo + (y - ymin) * (hi - lo) / (ymax - ymin)
    
    
    # ====================not used==================================
    def _normalize_to_display(self, y, display_min, display_max):
        # Compute normalized 0–1 range for raw data plotting
        y_min = y.min()
        y_max = y.max()
        y_norm = (y - y_min) / (y_max - y_min + 1e-12)

        # Map it into display range, BUT rescale proportionally if data range < display range
        scale_ratio = (y_max - y_min) / (display_max - display_min)

        # Center the waveform within the display range
        y_center = (display_max + display_min) / 2
        y_scaled = y_center + (y_norm - 0.5) * (y_max - y_min) / scale_ratio

        # Clamp to display limits just in case
        return np.clip(y_scaled, display_min, display_max)
