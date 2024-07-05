import re
import time

class laser:
    """
    Classe permettant d'interragir avec le telemetre laser
    Récupère les données de distance mesurée par le telemetre
    retourne la distance calculée au telemetre
    """
    
    def __init__(self, port: str = None,
                baudrate: int = 115200,
                timeout: int = 1,
                measure0: dict = None,
                measure1: dict = None):
        # Verification des parametres
        if port is None:
            raise ValueError("Serial port is missing")
        if measure0 is None:
            raise ValueError("No measure0 data")
        if measure1 is None:
            raise ValueError("No measure1 data")
        
        # Initiatilation port série
        import serial
        self.__ser = serial.Serial(port, baudrate, timeout=timeout)
        
        self.__measure0 = measure0
        self.__measure1 = measure1
    
    def extract_values(buffer: str = None):
        if buffer is None:
            return None
        data = None
        pattern = re.compile(r'([dD])(\d+\.\d+)[xX](\d+\.\d+)[yY](\d+\.\d+)[zZ](\d+\.\d+)')
        match = pattern.search(buffer)
        if match:
            num = 0
            if match.group(1) == 'D':
                num = 1
            x = match.group(2)
            y = match.group(3)
            z = match.group(4)
            data = {
                "num": num,
                "ts": time.time(),
                "x": float(x),
                "y": float(y),
                "z": float(z)
            }
        return data
    
    def read(self):
        buffer = self.ser.readline()
        measure = self.extract_values(buffer.decode('utf-8').rstrip())
        
        if measure is not None:
            if measure["num"] == 0: # measure 0
                if self.__measure0 is not None:
                    if self.__measure0["ts"] < measure["ts"]:
                        self.__measure0 = measure
                else:
                    self.__measure0 = measure
            else: # measure 1
                if self.__measure1 is not None:
                    if self.__measure1["ts"] < measure["ts"]:
                        self.__measure1 = measure
                else :
                    self.__measure1 = measure
    
    def write(self, distance: float = 0.0):
        if not isinstance(distance, float):
            raise ValueError("Distance must be a float")
        self.ser.write(f"{distance}\n".encode())
        
    # Getters
    def get_x_0(self):
        return self.__measure0["x"]
    
    def get_y_0(self):
        return self.__measure0["y"]
    
    def get_z_0(self):
        return self.__measure0["z"]
    
    def get_x_1(self):
        return self.__measure1["x"]
    
    def get_y_1(self):
        return self.__measure1["y"]
    
    def get_z_1(self):
        return self.__measure1["z"]
    
    def get_ts_0(self):
        return self.__measure0["ts"]
    
    def get_ts_1(self):
        return self.__measure1["ts"]
    
    def get_xyz_0(self):
        return self.__measure0["x"], self.__measure0["y"], self.__measure0["z"]
    
    def get_xyz_1(self):
        return self.__measure1["x"], self.__measure1["y"], self.__measure1["z"]