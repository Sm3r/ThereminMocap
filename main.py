import torch
from torch.utils.data import DataLoader
from data_loader import ThereminDataset
from network import *
from einops import rearrange, repeat
from tqdm import tqdm
import wandb

# from pythonosc import udp_client
# from pythonosc import osc_message_builder
import time

from time import perf_counter_ns


LOGS = True

def spinwait_us(delay):
    target = perf_counter_ns() + delay #* 1000
    while perf_counter_ns() < target:
        pass

# oscSender = udp_client.UDPClient("10.196.223.96", 57120)


if LOGS:
    wandb.init(project="ThereminMocap")

BATCH_SIZE = 1024 #4096
NUM_EPOCHS = 1000

train_dataset = ThereminDataset(training=True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = ThereminDataset(training=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


criterion = torch.nn.MSELoss()


    

def test(model, test_dataloader, criterion):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for i, (audio, mocap, loading_time) in tqdm(enumerate(test_dataloader)):
            start_time = perf_counter_ns()
            mocap_antennas = mocap[:, :6, :]
            mocap_hands = mocap[:, 6:, :]

            audio[:, 0] = audio[:, 0] #/ 4000 #1650.17

            mocap_antennas = rearrange(mocap_antennas, 'b c f -> b (c f)')
            mocap_hands = rearrange(mocap_hands, 'b c f -> b (c f)')

            output = model(mocap_hands.float(), mocap_antennas.float())

            # Create a mask that filters out the -1 mocap values and values < 0.1 in the audio
            mask = (mocap != -1) #& repeat((audio[:, 0] != 0.0), 'b -> b m d', m = mocap.shape[1], d = mocap.shape[2])

            loss_all = criterion(output.float(), audio.float())
            sum = torch.sum(mask)
            if sum != 0:
                loss = torch.sum(loss_all * mask) / sum
                total_loss += loss.item()

            '''
            pitch = output[0][0].item()
            volume = output[0][1].item()
            
            msg = osc_message_builder.OscMessageBuilder(address="/theremin")
            msg.add_arg(pitch)
            msg.add_arg(volume)
            msg = msg.build()
            oscSender.send(msg)
            '''

            processing_time = perf_counter_ns() - start_time
            total = 1/441 * 1000000000
            waiting_time = total - (processing_time + loading_time.item())
            
        print(f"Test Loss: {total_loss / len(test_dataloader)}")

        if LOGS:
            wandb.log({"test_loss": total_loss / len(test_dataloader)})



def train(model, train_dataloader, criterion, optimizer, num_epochs, save_model_path):
    min_loss = float('inf')
    for epoch in (pbar := tqdm(range(num_epochs))):
        model.train()
        mean_loss = 0
        max_loss = 0
        for i, (audio, mocap, _) in enumerate(train_dataloader):
            mocap_antennas = mocap[:, :6, :]
            mocap_hands = mocap[:, 6:, :]

            audio[:, 0] = audio[:, 0] #/ 4000 #1650.17
            # print(max(audio[:,0]), max(audio[:,1]))

            mocap_antennas = rearrange(mocap_antennas, 'b c f -> b (c f)')
            mocap_hands = rearrange(mocap_hands, 'b c f -> b (c f)')

            optimizer.zero_grad()
            output = model(mocap_hands.float(), mocap_antennas.float())

            # Create a mask that filters out the -1 mocap values and values < 0.1 in the audio
            mask = (mocap != -1) #& repeat((audio[:, 0] != 0.0), 'b -> b m d', m = mocap.shape[1], d = mocap.shape[2])

            if not i % 1000:
                print(f"GT pitch: {audio[0][0].item():.4f}, Est pitch: {output[0][0].item():.4f}")
                print(f"GT volume: {audio[0][1].item():.4f}, Est volume: {output[0][1].item():.4f}")

            loss_all = criterion(output.float(), audio.float())
            sum = torch.sum(mask)
            if sum != 0:
                loss = torch.sum(loss_all * mask) / sum
                loss.backward()
                optimizer.step()

                pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.3e}")
                mean_loss += loss.item()
                max_loss = max(max_loss, loss.item())

        mean_loss /= len(train_dataloader)
        if LOGS:
            wandb.log({"train_loss": mean_loss}, step=epoch)

        if mean_loss < min_loss:
            min_loss = mean_loss
            print(f"Saving best model at epoch {epoch}")
            torch.save(model.state_dict(), save_model_path)
        
        if epoch % 10 == 0:
            test(model, test_dataloader, criterion)


if __name__ == "__main__":

    # model = ThereminMLPBig()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)
    # min_loss = 10000
    # criterion = torch.nn.MSELoss()

    # save_model_path = "theremin_best_big.pth"

    # if LOGS:
    #     wandb.init(project="ThereminMocap", entity='francesco-dalri-2', name="ThereminMLPBig_Run")

    # train(model, train_dataloader, criterion, optimizer, NUM_EPOCHS, save_model_path)
  
    # if LOGS:
    #     wandb.finish()


    model = ThereminMLPTiny()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)
    min_loss = 10000
    criterion = torch.nn.MSELoss()

    save_model_path = "theremin_best_small.pth"

    if LOGS:
        wandb.init(project="ThereminMocap", name="ThereminMLP_Run")

    train(model, train_dataloader, criterion, optimizer, NUM_EPOCHS, save_model_path)

    if LOGS:
        wandb.finish() 