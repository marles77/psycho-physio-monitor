# PsychoPhysio monitor

This Python/Tkinter app uses Arduino and connected GSR/ECG sensors to monitor physiological activity.

## Python setup
Use `requirements.txt` to install packages in your environment. The script also uses **pyEDA** library which must be downloaded from https://github.com/HealthSciTech/pyEDA (WARNING: there is another pyEDA package provided by pip, but it is a completetly different library!).

Important!: in main.py of pyEDA module comment out all functions except `process_statistical()`. Do the same with the line `from pyEDA.pyEDA.autoencoder import *`. Otherwise, the interpreter will keep asking you to install unnecessary (and huge) packages such as `torch`.

ECG is analyzed using [HeartPy library](https://python-heart-rate-analysis-toolkit.readthedocs.io/).

The script uses one thread to collect data from Arduino (via pyserial), and one multiprocessing worker to analyze data from each epoch (results are returned as a variable `measures`) and generate plots.

`MODES_USED` constant is used to enable either synthetic data `('synth_gsr', 'synth_ecg')` or real data from GSR and ECG sensors `('gsr', 'ecg')`. Do not change the order of elements in this tuple.

## Arduino setup
The sketch for Arduino can be found in the file `sketch_physio_sensor.ino`. Arduino collects data at a frequency of 200Hz but sends data to serial at a frequency of 20Hz. A packet (n=10) of GSR data is avaraged before sending out and converted to $\mu S$. A packet (n=10) of ECG data is sent as it is (i.e ADC values 0-1023). 

### GSR
GSR sensor used in this project: https://wiki.seeedstudio.com/Grove-GSR_Sensor/

#### Connections
|Arduino|Grove-GSR Sensor|
|-:|-:|
|GND|Black|
|5V|Red|
|NC|White|
|A0|Yellow|
|||

### ECG
ECG sensor used in this project: https://wiki.dfrobot.com/Heart_Rate_Monitor_Sensor_SKU__SEN0213

#### Connections
|Arduino|HR Monitor Sensor|
|-:|-:|
|GND|Black|
|5V|RED|
|A1|Blue|
|||

### Commands

To communicate with Arduino board, and start data streaming, Python script uses following commands:
* 'a': GSR
* 'b': ECG
* 'c': synth ECG
* 'd': synth GSR
* 'x': stop streaming

## Known issues and missing features

The script was tested on Python 3.13.
There is currently no option to record raw data.