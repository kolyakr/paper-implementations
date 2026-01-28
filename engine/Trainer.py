import torch
from .DatasetProvider import DatasetProvider
from utils import get_loss_function, get_optimizer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from tqdm import tqdm

class Trainer:
    def __init__(self, model, dataset_name, loss_name, optimizer_name, batch_size, device, lr):
        self.model = model.to(device)
        self.data = DatasetProvider(dataset_name, batch_size)
        self.loss = get_loss_function(loss_name)
        self.optimizer = get_optimizer(optimizer_name, model.parameters(), lr)
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
            f"Dataset: [magenta]{dataset_name.value}[/]\n"
            f"Device: [yellow]{device}[/]",
            title="Experiment Config"
        ))

    def train_epoch(self, epoch_idx):
        self.model.train() # switch to train mode

        running_loss = 0.0

        pbar = tqdm(self.data.trainloader, 
                    desc=f"Epoch {epoch_idx}", 
                    unit="batch", 
                    leave=False) 
        
        for X_train, y_train in pbar:
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

        for X_test, y_test in self.data.testloader:
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)

            y_hat = self.model(X_test)
            loss = self.loss(y_hat, y_test)

            running_loss += loss.item()
            y_hat = torch.argmax(y_hat, dim=1)

            correct += (y_hat == y_test).sum().item()
            total_samples += y_hat.size(0)

        loss = running_loss / len(self.data.testloader)
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
                train_loss = self.train_epoch(epoch)
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










    