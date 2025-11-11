#
from serial import Serial, SerialException

def connect_serial(port, baudrate=9600, timeout=1):
    '''
    Connect to a serial port
    '''
    try:
        ser = Serial(port, baudrate=baudrate, timeout=timeout)
        return ser
    except SerialException as e:
        print(f"Error connecting to serial port {port}: {e}")
        return None
    
def read_serial(ser):
    '''
    Read a line from the serial port
    '''
    try:
        if ser and ser.is_open:
            line = ser.readline().decode('utf-8').rstrip()
            return line
        else:
            print("Serial port is not open")
            return None
    except SerialException as e:
        print(f"Error reading from serial port: {e}")
        return None