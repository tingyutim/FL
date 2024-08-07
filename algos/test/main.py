import torch

from flwr.client import ClientApp
from flwr.server import ServerApp
from flwr.simulation import run_simulation

import client
import server
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_PARTITIONS = 10

client = ClientApp(client_fn=client.client_fn)
server = ServerApp(server_fn=server.server_fn)

# Specify the resources each of your clients need
# If set to none, by default, each client will be allocated 2x CPU and 0x GPUs
backend_config = {"client_resources": None}
if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_gpus": 1}}

# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_PARTITIONS,
    backend_config=backend_config,
)
