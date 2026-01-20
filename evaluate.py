import torch
from torch.utils.data import DataLoader
from data_loader import ThereminDataset
from network import ThereminMLP
from einops import rearrange
from tqdm import tqdm
import time
from pythonosc import udp_client
from pythonosc import osc_message_builder
import time

from time import perf_counter_ns

def spinwait_us(delay):
    target = perf_counter_ns() + delay #* 1000
    while perf_counter_ns() < target:
        pass

BATCH_SIZE = 4096
NUM_EPOCHS = 1000


oscSender = udp_client.UDPClient("10.196.223.96", 57120)

test_dataset = ThereminDataset(training=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = ThereminMLP()

model.load_state_dict(torch.load("best.pth"))

criterion = torch.nn.MSELoss()

model.eval()
with torch.no_grad():
    total_loss = 0
    for i, (audio, mocap, loading_time) in tqdm(enumerate(test_dataloader)):
        start_time = perf_counter_ns()
        mocap_antennas = mocap[:, :6, :]
        mocap_hands = mocap[:, 6:, :]

        # Flatten the last two dimensions
        mocap_antennas = rearrange(mocap_antennas, 'b c f -> b (c f)')
        mocap_hands = rearrange(mocap_hands, 'b c f -> b (c f)')

        output = model(mocap_hands.float(), mocap_antennas.float())

        print(output.shape)

        pitch = output[0][0].item() * 3000.0
        volume = output[0][1].item()

        # pitch = audio[0][0].item() * 3000.0
        # volume = audio[0][1].item()
        # print(audio.shape)
        # print(audio)

        print(f"GT pitch: {audio[0][0].item() * 3000.0}, Est pitch: {output[0][0].item() * 3000.0}")
        print(f"GT volume: {audio[0][1].item()}, Est volume: {output[0][1].item()}")

        
        msg = osc_message_builder.OscMessageBuilder(address = "/theremin")
        msg.add_arg(pitch)
        msg.add_arg(volume)
        oscSender.send(msg.build())

        processing_time = perf_counter_ns() - start_time

        total = 1/441 * 1000000000
        waiting_time = total - (processing_time + loading_time.item())
        spinwait_us(waiting_time)

