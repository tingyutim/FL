import yaml
import flwr as fl

# Load configuration
with open("algos/test/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def fit_config(rnd: int):
    return {
        "batch_size": config["client"]["batch_size"],
        "local_epochs": config["client"]["epochs"],
    }

def main():
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        min_fit_clients=10,
        min_eval_clients=5,
        min_available_clients=10,
        on_fit_config_fn=fit_config,
    )
    # Start Flower server
    fl.server.start_server(config={"num_rounds": config["server"]["num_rounds"]}, strategy=strategy)

if __name__ == "__main__":
    main()
