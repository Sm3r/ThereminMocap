# ESEMPIO: 
# manda 3 messaggi OSCtemporizzati a SuperCollider per controllare il synth. 


from pythonosc.udp_client import SimpleUDPClient
import time

SC_IP = "127.0.0.1"
SC_PORT = 57120

client = SimpleUDPClient(SC_IP, SC_PORT)

client.send_message("/control", [440, 0.2])
time.sleep(1)
client.send_message("/control", [660, 0.15])
time.sleep(1)
client.send_message("/control", [220, 0.4])