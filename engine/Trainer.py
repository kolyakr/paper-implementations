import torch
from .DatasetProvider import DatasetProvider
from utils import get_loss_function, get_optimizer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from tqdm import tqdm
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, dataset, loss_name, optimizer_name, device, lr, optmizer_params=None, lerarning_schedule_params=None):
        self.model = model.to(device)
        self.data = dataset
        self.loss = get_loss_function(loss_name)
        self.optimizer = get_optimizer(optimizer_name, model.parameters(), lr, **optmizer_params)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            **lerarning_schedule_params
        )

        self.device = device

        self.history = {
            "train_loss": [],
            "val_loss": [], 
            "val_err": []
        }

        self.console = Console()
        self.console.print(Panel.fit(
            f"[bold green]Starting Training[/]\n"
            f"Model: [cyan]{model.__class__.__name__}[/]\n"
            f"Dataset: [red]{self.data.dataset_name.value}[/]\n"
            f"Device: [yellow]{device}[/]",
            title="Experiment Config"
        ))

    def train_epoch(self):
        self.model.train() # switch to train mode

        running_loss = 0.0

        for X_train, y_train in self.data.trainloader:
            X_train, y_train = X_train.to(self.device), y_train.to(self.device)

            self.optimizer.zero_grad()
            y_hat = self.model(X_train)
            loss = self.loss(y_hat, y_train)


            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        
        return running_loss / len(self.data.trainloader)
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        total_samples = 0
        correct = 0

        for X_test, y_test in self.data.validloader:
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)

            y_hat = self.model(X_test)
            loss = self.loss(y_hat, y_test)

            running_loss += loss.item()
            y_hat = torch.argmax(y_hat, dim=1)

            correct += (y_hat == y_test).sum().item()
            total_samples += y_hat.size(0)

        loss = running_loss / len(self.data.validloader)
        self.scheduler.step(loss)

        val_err = 1 - (correct / total_samples) 

        return loss, val_err
    
    def train(self, max_epochs=10):
        summary_table = Table(show_header=True, header_style="bold magenta")
        summary_table.add_column("Epoch", justify="center")
        summary_table.add_column("Train Loss", justify="right")
        summary_table.add_column("Test Loss", justify="right")
        summary_table.add_column("Test Err", justify="right")

        with Live(summary_table, console=self.console, refresh_per_second=4):
            for epoch in range(1, max_epochs + 1):
                train_loss = self.train_epoch()
                test_loss, val_err = self.evaluate()

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(test_loss)
                self.history["val_err"].append(val_err)

                summary_table.add_row(
                    str(epoch),
                    f"{train_loss:.4f}",
                    f"{test_loss:.4f}",
                    f"{val_err:.2%}" 
                )

        self.console.print("[bold green]✔ Training Complete![/]")
    
    @torch.no_grad()
    def make_predictions(self):
        self.model.eval()
        total_samples = 0
        correct = 0
        total_y_hat = []
        total_y_true = []
        total_probs = []

        for X_test, y_test in self.data.testloader:
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)

            logits = self.model(X_test)
            softmax_probs = F.softmax(logits, dim=1)

            probs, y_hat = torch.max(softmax_probs, dim=1)

            total_y_hat.append(y_hat)
            total_y_true.append(y_test)
            total_probs.append(probs)

        total_y_hat = torch.cat(total_y_hat)
        total_y_true = torch.cat(total_y_true)
        total_probs = torch.cat(total_probs)

        correct += (total_y_hat == total_y_true).sum().item()
        total_samples = total_y_hat.shape[0]
        val_err = 1 - (correct / total_samples) 

        return total_y_hat, total_y_true, val_err, total_probs