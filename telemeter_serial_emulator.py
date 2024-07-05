# Emulator for telemeter serial communication
#
# It send periodically measure:
#======================================
# d111.111x111.111y111.111z111.111
# delay(MEASURE_DELAY_1)
# D111.111X111.111Y111.111Z111.111
# delay(MEASURE_DELAY_2)
#======================================
# It can also display the received data

import serial
import threading
import time

# Serial port configuration
SERIAL_PORT = 'COM7'
BAUD = 115200
TIMEOUT = 1

# Emulator configuration
# This format is used to send data
# d_d1_x_x1_y_y1_z_z1
# D_d2_X_x2_Y_y2_Z_z2
# _ is used as separator but its not sent
# Value must always be 6 digits 3 before and 3 after the dot

d1 = 111.111
d2 = 111.222
x1 = 222.111
x2 = 222.222
y1 = 333.111
y2 = 333.222
z1 = 444.111
z2 = 444.222

MEASURE_DELAY_1 = 2
MEASURE_DELAY_2 = 4

# Function to read from the serial port
def read_from_serial(serial_port):
    while True:
        if serial_port.in_waiting > 0:
            data = serial_port.readline().decode('utf-8').rstrip()
            print(f"Received: {data}")
            time.sleep(1)

# Function to write to the serial port
def write_to_serial(serial_port, data, delay_measure):
    while True:
        serial_port.write(data.encode('utf-8'))
        print(f"Sent: {data}")
        time.sleep(delay_measure)

def main():
    serial_port = serial.Serial(
        port=SERIAL_PORT, 
        baudrate=BAUD,
        timeout=TIMEOUT
    )
    print(serial)

    # Data to send
    data_1 = f"d{d1:.3f}x{x1:.3f}y{y1:.3f}z{z1:.3f}\n"
    data_2 = f"D{d2:.3f}X{x2:.3f}Y{y2:.3f}Z{z2:.3f}\n"

    # Create and start the reading thread
    read_thread = threading.Thread(target=read_from_serial, args=(serial_port,))
    read_thread.daemon = True
    read_thread.start()

    # Create and start the writing threads
    write_thread_1 = threading.Thread(target=write_to_serial, args=(serial_port, data_1, MEASURE_DELAY_1, ))
    write_thread_1.daemon = True
    write_thread_1.start()

    write_thread_2 = threading.Thread(target=write_to_serial, args=(serial_port, data_2, MEASURE_DELAY_2, ))
    write_thread_2.daemon = True
    write_thread_2.start()

    # Keep the main thread alive to keep the program running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
        serial_port.close()

if __name__ == "__main__":
    main()