# ==========================================================
# Psycho Physio Monitor
# Author: Marcin Leśniak, PhD
#
# Main classes
# ==========================================================

import io
from logger import logger
import time
from tkinter.messagebox import showinfo
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.style import Bootstyle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#from matplotlib.backend_bases import MouseButton
from matplotlib.ticker import FuncFormatter
#import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import serial
from threading import Thread, Event
from queue import Queue
import multiprocessing as mp
from typing import Union, Any
from dataclasses import dataclass
import struct
#from collections import deque
from helpers import clear_queue, processing_worker_analysis
from constants import Constants

# ==========================================================
# Settings and Constants
# ==========================================================

@dataclass
class SpinValue:
    plot1_ecg_low: int = 100
    plot1_ecg_up: int = 500
    plot1_gsr_low: int = 0
    plot1_gsr_up: int = 60
    plot3_ecg_low: int = 10
    plot3_ecg_up: int = 150
    plot3_gsr_low: int = 0
    plot3_gsr_up: int = 60


# ==========================================================
# Serial Reader (thread)
# ==========================================================

class SerialReader:
    def __init__(self, 
                 port: str = Constants.PORTS[2], 
                 baudrate: int = Constants.BAUDRATES[4], 
                 sample_count: int = Constants.SAMPLE_COUNT_ECG,
                 packet_size: int = Constants.SAMPLE_COUNT_ECG * Constants.NBYTES,  
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
        self.mode = ""


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
            self.ser.write(Constants.MODES_COMMANDS[self.mode].encode())
            time.sleep(1)

            try:
                while self.run_event.is_set():
                        #print("Reading data...")
                        if self.pause_event.is_set():
                            # paused — do not read
                            time.sleep(0.1)
                            continue
                        raw = self.ser.read(self.packet_size)
                        
                        if len(raw) == self.packet_size:
                            vals = struct.unpack('<' + 'H'*self.sample_count, raw)
                            pkt = np.array(vals, dtype=np.float32)/100 if self.mode in ('gsr', 'synth_gsr') else np.array(vals, dtype=np.uint16) 
                            
                            if self.raw_queue:
                                self.raw_queue.put(pkt)
                            if self.proc_queue:
                                self.proc_queue.put(pkt)
                            
                    
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
# GUI Application
# ==========================================================            
class App(ttk.Window):
    def __init__(self) -> None:
        super().__init__(title=Constants.APP_TITLE, themename=Constants.APP_THEME, size=Constants.WINDOW_SIZE, resizable=(1, 1))
        self.protocol('WM_DELETE_WINDOW', self.on_close)

        self.port_index = 2
        self.baud_index = 4

        self.counter = 0

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
        self.buffer_duration = Constants.EPOCH_TIME  # seconds visible on raw plot
        self.buffer = None
        self.filtered_data = np.array([], dtype=np.uint16)
        self.agg_measure1 = []
        self.agg_buffer = None
        self.multiplier = None

        # configure window grid
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.frame = ttk.Frame(master=self, padding=10)
        #self.frame.grid_rowconfigure(0, minsize=50)
        self.frame.grid_rowconfigure(4, minsize=160)
        self.frame.grid_rowconfigure(5, minsize=160)
        
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

        # First update of plots
        self.after(Constants.RAW_PLOT_RATE, self.update_raw_plot)
        self.after(Constants.FILT_PLOT_RATE, self.update_agg_plots)
        

    def _create_widgets(self, master) -> None:
        
        # --- 1st row ---

        self.label_head = ttk.Label(master=master, text=f"Welcome to {Constants.APP_TITLE}")
        self.label_head.grid(column=0, row=0, sticky=W, padx=1, pady=10, columnspan=6)

        # --- 2nd row buttons ---

        self.button_conn = ttk.Button(master=master, text="Connect", bootstyle="success", command=self.on_button_conn_click, width=Constants.BUTTON_WIDTH)
        self.button_conn.grid(column=0, row=1, sticky=NSEW, padx=1, pady=1)

        self.button_gsr = ttk.Button(master=master, text="Restart GSR", bootstyle="info", command=self.on_button_gsr_click, width=Constants.BUTTON_WIDTH, state=DISABLED)
        self.button_gsr.grid(column=1, row=1, sticky=NSEW, padx=1, pady=1)

        self.button_ecg = ttk.Button(master=master, text="Restart ECG", bootstyle="info", command=self.on_button_ecg_click, width=Constants.BUTTON_WIDTH, state=DISABLED)
        self.button_ecg.grid(column=2, row=1, sticky=NSEW, padx=1, pady=1)

        self.button_stop = ttk.Button(master=master, text="Stop", bootstyle="warning", command=self.on_button_stop_click, width=Constants.BUTTON_WIDTH, state=DISABLED)
        self.button_stop.grid(column=3, row=1, sticky=NSEW, padx=1, pady=1)

        self.cbo_port = ttk.Combobox(master=master, text="Select COM Port", values=Constants.PORTS, exportselection=False, width=Constants.BUTTON_WIDTH)
        self.cbo_port.grid(column=4, row=1, sticky=NSEW, padx=1, pady=1)
        self.cbo_port.current(self.port_index)

        self.cbo_baud = ttk.Combobox(master=master, text="Select Baudrate", values=Constants.BAUDRATES, exportselection=False, width=Constants.BUTTON_WIDTH)
        self.cbo_baud.grid(column=5, row=1, sticky=NSEW, padx=1, pady=1)
        self.cbo_baud.current(self.baud_index)

        # --- Spinboxes to control axes limits ---
        
        self.input_group1 = ttk.Labelframe(master=master, text="Raw plot limits", padding=1)
        self.input_group1.grid(column=0, row=2, rowspan=2, sticky=W, padx=1, pady=1)
        self.spinbox_plot1_up = ttk.Spinbox(master=self.input_group1, from_=0, to=1000)
        self.spinbox_plot1_up.pack(fill=X, padx=2, pady=2)
        self.spinbox_plot1_low = ttk.Spinbox(master=self.input_group1, from_=0, to=1000)
        self.spinbox_plot1_low.pack(fill=X, padx=2, pady=2)
        self.spinbox_plot1_up.set(self.spin_value.plot1_ecg_up)
        self.spinbox_plot1_low.set(self.spin_value.plot1_ecg_low)

        self.trim1_low = 0
        self.trim1_up = 0

        self.input_group2 = ttk.Labelframe(master=master, text="Epoch time (sec)", padding=1)
        self.input_group2.grid(column=0, row=4, rowspan=2, sticky=W, padx=1, pady=1)
        self.spinbox_time = ttk.Spinbox(master=self.input_group2, from_=5, to=30)
        self.spinbox_time.pack(fill=X, padx=2, pady=2)
        self.spinbox_time.set(Constants.EPOCH_TIME)
        # self.toggle_record = ttk.Checkbutton(master=master, text="Record", bootstyle=(DANGER, ROUND, TOGGLE))
        # self.toggle_record.invoke()
        # self.toggle_record.grid(column=0, row=5, sticky=NSEW, padx=1, pady=1)

        self.input_group3 = ttk.Labelframe(master=master, text="Epochs plot limits", padding=1)
        self.input_group3.grid(column=0, row=6, rowspan=2, sticky=W, padx=1, pady=1)

        self.spinbox_plot3_up = ttk.Spinbox(master=self.input_group3, from_=0, to=1000)
        self.spinbox_plot3_up.pack(fill=X, padx=2, pady=2)
        self.spinbox_plot3_low = ttk.Spinbox(master=self.input_group3, from_=0, to=1000)
        self.spinbox_plot3_low.pack(fill=X, padx=2, pady=2)
        self.spinbox_plot3_up.set(self.spin_value.plot3_ecg_up)
        self.spinbox_plot3_low.set(self.spin_value.plot3_ecg_low)

        self.trim3_low = 0
        self.trim3_up = 0

        # --- Raw plot (data from thread) ---

        self.fig1 = Figure(figsize=(12, 2), dpi=100, tight_layout=True)
        self.ax1 = self.fig1.add_subplot()
        self.ax1.tick_params(labelsize=8)
        self.ax1.set_xlabel("samples", fontsize=8)
        self.raw_line = self.ax1.plot([], [], lw=1, color = 'g')[0]
        self.ax1.set_autoscaley_on(False) 
        #self.ax1.set_ylim(self.spinbox_plot1_low.get(), self.spinbox_plot1_up.get())
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=master)
        self.canvas1.get_tk_widget().grid(column=1, row=2, columnspan=5, rowspan=2, sticky=NSEW)

        # --- Filtered plot (data from process: heartPy / pyEDA image) ---

        self.label_fig2 = ttk.Label(master=master, text="No data", image="", anchor="center", justify="center", bootstyle="danger")
        self.label_fig2.grid(column=1, row=4, columnspan=5, rowspan=2, sticky=NSEW, padx=(1, 0), pady=1)
        
        # --- Aggregation plot (data from process: heartPy / pyEDA measures)---

        self.fig3 = Figure(figsize=(12, 2), dpi=100, tight_layout=True)
        self.ax3 = self.fig3.add_subplot()
        self.ax3.tick_params(labelsize=8)
        self.ax3.set_xlabel("epochs", fontsize=8)
        self.agg_line = self.ax3.plot([], [], color='orange', lw=1.5)[0]
        #self.ax3.grid(axis='y')
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=master)
        self.canvas3.get_tk_widget().grid(column=1, row=6, columnspan=5, rowspan=2, sticky=NSEW)

        # --- Bottom labels ---

        self.label_bottom1 = ttk.Label(master=master, text=f"Status: {'Disconnected'}")
        self.label_bottom1.grid(column=0, row=8, sticky=SW, padx=1, pady=5, columnspan=2)

        self.label_bottom2 = ttk.Label(master=master, text="Streaming not started")
        self.label_bottom2.grid(column=2, row=8, sticky=SW, padx=1, pady=5, columnspan=2)

        self.label_val1 = ttk.Label(master=master, text="N/A")
        self.label_val1.grid(column=4, row=8, sticky=SW, padx=1, pady=5, columnspan=1)

        self.label_val2 = ttk.Label(master=master, text="N/A")
        self.label_val2.grid(column=5, row=8, sticky=SW, padx=1, pady=5, columnspan=1)
        

    # ============ TODO: consider if it is needed at all ============
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
# ====================================================================

    def on_button_conn_click(self) -> None:
        if not self.serial_reader.ser or not self.serial_reader.ser.is_open:
            selected_port = self.cbo_port.get()
            baudrate = int(self.cbo_baud.get())
            self.serial_reader.set_port_baud(selected_port, baudrate)
            #showinfo("Information", f"Set port to {selected_port} with baudrate {baudrate}.")
            self.conn = self.serial_reader.start()
            if self.conn:
                self.button_conn.config(bootstyle="danger", text="Disconnect")
                self.label_bottom1.config(text=f"Status: Connected to {selected_port} at {baudrate} baudrate")
                #showinfo("Information", f"Serial port {self.serial_reader.port} opened successfully.")
                time.sleep(1)
                self.button_ecg.config(state=NORMAL)
                self.button_gsr.config(state=NORMAL)
                self.button_stop.config(state=NORMAL)

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
            print("Disconnected from serial port.")
            self.button_ecg.config(state=DISABLED)
            self.button_gsr.config(state=DISABLED)
            self.button_stop.config(state=DISABLED)
            self.button_conn.config(bootstyle="success", text="Connect")
            self.label_bottom1.config(text="Status: Disconnected")
            self.label_bottom2.config(text="Streaming not started")
            self.label_val1.config(text="N/A")
            self.label_val2.config(text="N/A")

            # Clear plots
            self.raw_line.set_data([], [])
            self.canvas1.draw_idle()
            self.canvas1.flush_events()

            self.label_fig2.config(text="No data", image="")

            self.agg_line.set_data([], [])
            for txt in list(self.ax3.texts):
                try:
                    txt.remove()
                except Exception:
                    pass
            self.canvas3.draw_idle()
            self.canvas3.flush_events()
        

    def on_button_gsr_click(self) -> None:
        #showinfo("Information", "Button GSR was clicked!")
        if self.conn:
            mode = Constants.MODES_USED[0]
            epoch_time = int(self.spinbox_time.get())

            # Buttons etc.
            self.button_gsr.config(state=DISABLED)
            self.button_ecg.config(state=DISABLED)
            self.button_stop.config(state=NORMAL)
            self.spinbox_plot1_up.set(self.spin_value.plot1_gsr_up)
            self.spinbox_plot1_low.set(self.spin_value.plot1_gsr_low)
            self.spinbox_plot3_up.set(self.spin_value.plot3_gsr_up) 
            self.spinbox_plot3_low.set(self.spin_value.plot3_gsr_low)
            self.label_bottom2.config(text=f"Streaming {mode.upper()} at {Constants.PACK_RATE*Constants.SAMPLE_COUNT_GSR} Hz")
            
            # Prepare GSR buffers etc.
            self.multiplier = Constants.MULTIPLIER_GSR
            self.sample_count = Constants.SAMPLE_COUNT_GSR
            self.sample_rate = Constants.SAMPLE_COUNT_GSR * Constants.PACK_RATE
            self.buffer = np.zeros((Constants.PACK_RATE * Constants.SAMPLE_COUNT_GSR * epoch_time))
            self.buffer_idx = 0
            self.agg_buffer = np.zeros(100)
            self.agg_buffer_idx = 1

            logger.info(f"Buffer shape: {self.buffer.shape[0]}")

            # Fig1 GSR setup
            self.ax1.set_xlim(0, self.buffer.shape[0])
            low1 = int(self.spinbox_plot1_low.get())
            up1 = int(self.spinbox_plot1_up.get())
            self.trim1_low = low1 / Constants.DIVIDER_GSR
            self.trim1_up = (Constants.DIVIDER_GSR - up1) / Constants.DIVIDER_GSR
            self.ax1.set_ylim(low1, up1)
            #self.ax1.vlines(200, 0, 0.5, color='black', lw=0.5)
            self.canvas1.draw_idle()
            self.canvas1.flush_events()

            # Fig3 GSR setup
            self.ax3.set_xlim(0, self.agg_buffer.shape[0])
            low3 = int(self.spinbox_plot3_low.get())
            up3 = int(self.spinbox_plot3_up.get())
            self.trim3_low = low3 / Constants.MULTIPLIER_GSR
            self.trim3_up = (Constants.MULTIPLIER_GSR - up3) / Constants.MULTIPLIER_GSR
            self.ax3.set_ylim(self.trim3_low, 1 - self.trim3_up)
            self.ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y*self.multiplier:.0f}'))
            self.ax3.vlines(0, 0, 0.5, color='black', lw=0.5)
            self.canvas3.draw_idle()
            self.canvas3.flush_events()

            # Set GSR events and starting thread + process
            print("Starting acquisition of GSR data...")
            self.run_event.set()
            self.pause_event.clear()
            self.stop_event.clear()
            self.serial_reader.sample_count=Constants.SAMPLE_COUNT_GSR
            self.serial_reader.packet_size=Constants.SAMPLE_COUNT_GSR * Constants.NBYTES
            self.serial_reader.mode = mode
            self.serial_reader.thread.start()
            self.proc = mp.Process(target=processing_worker_analysis, 
                                   args=(self.proc_in, self.proc_out, self.stop_event), 
                                   kwargs={'pack_rate': Constants.PACK_RATE, 'time_window': epoch_time, 'sample_count': Constants.SAMPLE_COUNT_GSR, 'mode': mode}, 
                                   daemon=True)
            self.proc.start()



    def on_button_ecg_click(self) -> None:
        #showinfo("Information", "Button ECG was clicked!")
        if self.conn:
            mode = Constants.MODES_USED[1]
            epoch_time = int(self.spinbox_time.get())

            # Buttons etc.
            self.button_ecg.config(state=DISABLED)
            self.button_gsr.config(state=DISABLED)
            self.button_stop.config(state=NORMAL)
            self.spinbox_plot1_up.set(self.spin_value.plot1_ecg_up)
            self.spinbox_plot1_low.set(self.spin_value.plot1_ecg_low)
            self.spinbox_plot3_up.set(self.spin_value.plot3_ecg_up) 
            self.spinbox_plot3_low.set(self.spin_value.plot3_ecg_low)
            self.label_bottom2.config(text=f"Streaming {mode.upper()} at {Constants.PACK_RATE*Constants.SAMPLE_COUNT_ECG} Hz")

            # Prepare ECG buffers etc.
            self.multiplier = Constants.MULTIPLIER_HR
            self.sample_count = Constants.SAMPLE_COUNT_ECG
            self.sample_rate = Constants.SAMPLE_COUNT_ECG * Constants.PACK_RATE
            self.buffer = np.zeros((Constants.PACK_RATE * Constants.SAMPLE_COUNT_ECG * epoch_time))
            self.buffer_idx = 0
            self.agg_buffer = np.zeros(100)
            self.agg_buffer_idx = 1

            # Fig1 ECG setup
            self.ax1.set_xlim(0, self.buffer.shape[0])
            low1 = int(self.spinbox_plot1_low.get())
            up1 = int(self.spinbox_plot1_up.get())
            self.trim1_low = low1 / Constants.DIVIDER_ECG
            self.trim1_up = (Constants.DIVIDER_ECG - up1) / Constants.DIVIDER_ECG
            self.ax1.set_ylim(low1, up1)
            #self.ax1.vlines(200, 0, 0.5, color='black', lw=0.5)
            self.canvas1.draw_idle()
            self.canvas1.flush_events()

            # Fig3 ECG setup
            self.ax3.set_xlim(0, self.agg_buffer.shape[0])
            low3 = int(self.spinbox_plot3_low.get())
            up3 = int(self.spinbox_plot3_up.get())
            self.trim3_low = low3 / Constants.MULTIPLIER_HR
            self.trim3_up = (Constants.MULTIPLIER_HR - up3) / Constants.MULTIPLIER_HR
            self.ax3.set_ylim(self.trim3_low, 1 - self.trim3_up)
            self.ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y*self.multiplier:.0f}'))
            self.ax3.vlines(0, 0, 0.5, color='black', lw=0.5)
            self.canvas3.draw_idle()
            self.canvas3.flush_events()
            #logger.info(f"Initial data x: {x}, y: {y}.")

            # Set ECG events and starting thread + process
            print("Starting acquisition of ECG data...")
            self.run_event.set()
            self.pause_event.clear()
            self.stop_event.clear()
            self.serial_reader.sample_count=Constants.SAMPLE_COUNT_ECG
            self.serial_reader.packet_size=Constants.SAMPLE_COUNT_ECG * Constants.NBYTES
            self.serial_reader.mode = mode
            self.serial_reader.thread.start()
            self.proc = mp.Process(target=processing_worker_analysis, 
                                   args=(self.proc_in, self.proc_out, self.stop_event), 
                                   kwargs={'pack_rate': Constants.PACK_RATE, 'time_window': epoch_time, 'sample_count': Constants.SAMPLE_COUNT_ECG, 'mode': mode}, 
                                   daemon=True)
            self.proc.start()
            
        
    def on_button_stop_click(self) -> None:
        print("Paused")
        self.run_event.clear()
        self.pause_event.clear()
        self.stop_event.set()
        self.serial_reader.pause()
        #self.pause_event.set()
        
        if self.serial_reader.thread.is_alive():
            self.serial_reader.thread.join(timeout=1)
        if self.proc:
            self.proc.join(timeout=1)
            self.proc = None
        self.button_stop.config(state=DISABLED)
        print("Stopped cleanly.")

    
    # ======= TODO: not implemented yet ======
    def resume(self):
        print("Resuming reader…")
        self.pause_event.clear()
    # ========================================


    def on_close(self) -> None:
        '''
        Called when close icon clicked
        Safely kill the app
        '''

        self.serial_reader.pause()
        self.stop_event.set()
        
        if self.serial_reader.thread:
            if self.serial_reader.thread.is_alive():
                    self.serial_reader.thread.join(timeout=1)
        if self.proc:
            self.proc.join(timeout=1)
            self.proc = None
        
        self.serial_reader.stop()
        time.sleep(0.1)
        clear_queue(self.raw_queue, block=True)
        clear_queue(self.proc_in, block=True)
        clear_queue(self.proc_out, block=True)
        self.quit()
        self.destroy()
        
        print(f"App closed")


    # ======================================================
    # Plot update loops
    # ======================================================
    
    def update_raw_plot(self):
        """Continuously updates raw data plot"""
        new_data = False
        divider = Constants.DIVIDER_ECG if self.serial_reader.mode in ('ecg', 'synth_ecg') else Constants.DIVIDER_GSR
        
        if (self.buffer is not None) and (self.run_event.is_set()):
            try:
                while not self.raw_queue.empty():
                        pkt = self.raw_queue.get_nowait()
                        pkt_len = len(pkt)
                        if (pkt_len > 0) and (self.buffer_idx + pkt_len <= self.buffer.shape[0]):
                            #shape_to_report = self.buffer[self.buffer_idx : self.buffer_idx + pkt_len].shape
                            idx = self.buffer_idx
                            self.buffer[idx : idx + pkt_len] = pkt
                            self.buffer_idx += pkt_len
                            new_data = True
            except (ValueError, Exception) as e:
                logger.info(f"Error in update_raw_plot: {e}")
                #logger.info(f"Buffer size: {self.buffer.shape}, Buffer index = {idx}, Broadcast shape: {shape_to_report}, Packet size: {pkt.shape}, Packet len: {pkt_len}")
                
            if new_data:
                y = self.buffer[:self.buffer_idx]
                y_norm = y / divider
                x = np.arange(y.shape[0])
                #logger.info(f"x: {x}, y: {y_norm}")
                self.raw_line.set_data(x, y_norm)
                self.ax1.set_ylim(self.trim1_low, 1 - self.trim1_up)
                self.ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y*divider:.0f}'))
                self.canvas1.draw_idle()
                self.canvas1.flush_events()
                self.label_val1.configure(text=f"Len: {x.shape[0]}")
                #logger.info(f"Min: {y_norm.min()}, Max: {y_norm.max()}")
            if self.buffer_idx > (self.buffer.shape[0] - self.sample_count):
                self.buffer.fill(0)
                self.buffer_idx = 0  # reset index when buffer is full

        self.after(Constants.RAW_PLOT_RATE, self.update_raw_plot)
        

    def update_agg_plots(self):
        """Updates filtered and aggregated epoch data plots"""
        new_data = False
        img_data = False
        if self.run_event.is_set():
            while not self.proc_out.empty():
                data = self.proc_out.get_nowait()
                img_data = data[0]
                measures = data[1]
                #filtered = data[2]
                new_data = True
                
            if new_data and img_data: 
                self.counter += 1
                image = Image.open(io.BytesIO(img_data))
                self.photo = ImageTk.PhotoImage(image)
                self.label_fig2.config(text="", image=self.photo)
                self.agg_buffer[self.agg_buffer_idx] = measures['bpm']/self.multiplier if self.serial_reader.mode in ('ecg', 'synth_ecg') else measures['mean_gsr'][0]/self.multiplier
                y = self.agg_buffer[:self.agg_buffer_idx+1]
                x = np.arange(y.shape[0])
                self.agg_line.set_data(x, y)

                for xy in zip(x, y):                                       
                    self.ax3.annotate('%.1f' % float(xy[1]*self.multiplier), xy=xy, textcoords='data', fontsize=6)
                self.ax3.set_ylim(self.trim3_low, 1 - self.trim3_up)
                self.ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{y*self.multiplier:.0f}'))
                
                self.canvas3.draw_idle()
                self.canvas3.flush_events()
                self.agg_buffer_idx += 1

                if self.serial_reader.mode in ('ecg', 'synth_ecg'):
                    self.label_val2.configure(text=f"HR: {measures['bpm']:.1f}, RMSSD: {measures['rmssd']:.1f}")
                else:
                    self.label_val2.configure(text=f"Mean GSR: {measures['mean_gsr'][0]:.1f}, N peaks: {measures['number_of_peaks'][0]}")
        

        self.after(Constants.FILT_PLOT_RATE, self.update_agg_plots)


