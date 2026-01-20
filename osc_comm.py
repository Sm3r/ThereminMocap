from pythonosc import udp_client
from pythonosc import osc_message_builder
import time
import pandas as pd
import numpy as np


oscSender = udp_client.SimpleUDPClient("10.196.223.96", 57120)

dataframe = pd.read_csv("/home/ardan/ARDAN/ThereminMocap/ThereminSong_audio.csv")

for i in range(len(dataframe)):
    pitch = dataframe["Pitch"][i]
    volume = dataframe["Volume"][i]
    msg = osc_message_builder.OscMessageBuilder(address="/theremin")
    msg.add_arg(pitch)
    msg.add_arg(volume)
    msg = msg.build()
    oscSender.send(msg)
    print(msg)
    time.sleep(1/441)

