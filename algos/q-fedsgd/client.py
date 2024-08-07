import numpy as np
import torch
from flwr.client import Client, NumPyClient
from flwr.common import Context
from data.data_loader import load_datasets
from models.cifar import Net
from collections import OrderedDict
from typing import List

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, valloader, learning_rate):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.learning_rate = learning_rate

    def get_parameters(self, config):
        print(f"[Client {self.partition_id}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] fit, config: {config}")
        set_parameters(self.net, parameters)
        q = config.get("q", 1.0)  # Get q from config
        train(self.net, self.trainloader, epochs=1)
        delta_k, h_k = self.compute_delta_h(self.net, self.trainloader, q)
        return get_parameters(self.net), len(self.trainloader), {"delta": delta_k, "h": h_k}

    def evaluate(self, parameters, config):
        print(f"[Client {self.partition_id}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

    def compute_delta_h(self, net, dataloader, q):
        """Compute delta and h for q-FedSGD."""
        criterion = torch.nn.CrossEntropyLoss()
        net.eval()
        delta_k = None
        h_k = 0
        num_examples = 0
        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch["img"], batch["label"]
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = net(images)
                loss = criterion(outputs, labels)
                grads = torch.autograd.grad(loss, net.parameters(), retain_graph=True)
                loss_value = loss.item()
                if delta_k is None:
                    delta_k = [(loss_value ** q) * g.cpu().numpy() for g in grads]
                else:
                    delta_k = [dk + (loss_value ** q) * g.cpu().numpy() for dk, g in zip(delta_k, grads)]
                norm = sum([torch.sum(g ** 2).item() for g in grads])
                h_k += q * (loss_value ** (q - 1)) * norm + (1.0 / self.learning_rate) * (loss_value ** q)
                num_examples += len(labels)
        delta_k = [dk / num_examples for dk in delta_k]  # Average over all examples
        h_k /= num_examples  # Average over all examples
        return delta_k, h_k

def client_fn(context: Context) -> Client:
    net = Net().to(DEVICE)
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader, _ = load_datasets(partition_id, num_partitions)
    learning_rate = 0.001  # Set a default learning rate or retrieve from context/config
    return FlowerClient(partition_id, net, trainloader, valloader, learning_rate).to_client()
