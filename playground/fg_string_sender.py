import socket
import numpy as np

class FGStringSender:
    """JSBSim verilerini FlightGear'a düz metin (ASCII) olarak gönderen kararlı sınıf"""
    def __init__(self, dest_ip="127.0.0.1", dest_port=5555):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest = (dest_ip, dest_port)

    def send_state(self, fdm):
        # JSBSim'den değerleri alıyoruz
        lat = fdm['position/lat-geod-deg']
        lon = fdm['position/long-gc-deg']
        alt = fdm['position/h-sl-ft']
        
        # Radyanları dereceye çeviriyoruz
        roll = np.degrees(fdm['attitude/roll-rad'])
        pitch = np.degrees(fdm['attitude/pitch-rad'])
        heading = np.degrees(fdm['attitude/psi-rad'])

        # XML şablonumuzun beklediği formatta düz bir metin oluşturuyoruz
        # Format: lat,lon,alt,roll,pitch,heading\n
        packet_string = f"{lat},{lon},{alt},{roll},{pitch},{heading}\n"
        
        try:
            self.sock.sendto(packet_string.encode('utf-8'), self.dest)
        except:
            pass