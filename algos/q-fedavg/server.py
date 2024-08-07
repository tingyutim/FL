import torch
from flwr.server import ServerConfig, ServerAppComponents
from flwr.server.strategy import QFedAvg
from flwr.common import Context

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_PARTITIONS = 10

def server_fn(context: Context) -> ServerAppComponents:
    # Create FedAvg strategy
    strategy = QFedAvg(
        q_param = 0.2,
        qffl_learning_rate = 0.1,
        fraction_fit = 1.0,
        fraction_evaluate = 1.0,
        min_fit_clients = 1,
        min_evaluate_clients = 1,
        min_available_clients = 1,
    )

    # Configure the server for 3 rounds of training
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(strategy=strategy, config=config)