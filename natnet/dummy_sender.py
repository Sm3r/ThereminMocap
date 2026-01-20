from pythonosc import udp_client
from pythonosc import osc_message_builder
import time
import random

def main():
  oscSender = udp_client.UDPClient("10.196.223.96", 57120)
  while True:
    random_pitch = random.uniform(30, 3000)
    random_volume = random.uniform(0, 1)

    print(random_pitch, random_volume)

    msg = osc_message_builder.OscMessageBuilder(address = "/theremin")
    msg.add_arg(random_pitch)
    msg.add_arg(random_volume)
    oscSender.send(msg.build())

    time.sleep(1/441)  

if __name__ == "__main__":
  main()