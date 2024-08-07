import torch
from flwr.server import ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import Context

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_PARTITIONS = 10

def server_fn(context: Context) -> ServerAppComponents:
    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.3,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=NUM_PARTITIONS,
    )

    # Configure the server for 3 rounds of training
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(strategy=strategy, config=config)