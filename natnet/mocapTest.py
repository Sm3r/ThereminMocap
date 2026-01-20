import time
from threading import Thread
from NatNetClient import NatNetClient  # Import the provided NatNetClient


# Initialize the NatNet client
client = NatNetClient()

# Set server address (if different)
client.set_server_address('127.0.0.1')

# Optionally set client address if needed
client.set_client_address('127.0.0.1')

client.set_use_multicast(True)

is_running = client.run()

print(f"Is the client running? {is_running}")


# Specify a recording name
tak_filename = "MyCustomRecording"
print(f"Setting recording name to {tak_filename}...")
client.send_command(f"SetRecordTakeName,{tak_filename}")

# Start Recording in Motive
print("Sending start recording command...")
client.send_command("StartRecording")

print(f"Recording '{tak_filename}.tak' started...")
time.sleep(10)  # Record for 10 seconds

# Stop Recording
print("Sending stop recording command...")
client.send_command("StopRecording")

client.shutdown()
