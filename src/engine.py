from pathlib import Path
import torch
from tqdm.auto import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        checkpoint_path=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_path = checkpoint_path

        if self.checkpoint_path is not None:
            Path(self.checkpoint_path).parent.mkdir(
                parents=True,
                exist_ok=True
            )
    def train_one_epoch(self, epoch, total_epochs):
        self.model.train()
        running_loss = 0.0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{total_epochs} Training",
            unit="batch"
        )

        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(images)

            loss = self.criterion(logits, labels)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        epoch_loss = running_loss / len(self.train_loader)

        return epoch_loss

    def validate(self, epoch, total_epochs):
        self.model.eval()
        running_loss = 0.0

        progress_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch}/{total_epochs} Validation",
            unit="batch"
        )

        with torch.no_grad():
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)

                loss = self.criterion(logits, labels)

                running_loss += loss.item()

                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

        epoch_loss = running_loss / len(self.val_loader)

        return epoch_loss

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.history = checkpoint["history"]
        self.best_val_loss = checkpoint["best_val_loss"]

        return checkpoint

    def fit(self, num_epochs):
        if not hasattr(self, "history"):
            self.history = {
                "train_loss": [],
                "val_loss": []
            }

        if not hasattr(self, "best_val_loss"):
            self.best_val_loss = float("inf")

        start_epoch = len(self.history["train_loss"])
        total_epochs = start_epoch + num_epochs

        progress_bar = tqdm(
            total=total_epochs,
            initial=start_epoch,
            desc="Training Progress",
            unit="epoch"
        )

        for epoch in range(num_epochs):
            current_epoch = start_epoch + epoch + 1

            train_loss = self.train_one_epoch(
                current_epoch,
                total_epochs
            )

            val_loss = self.validate(
                current_epoch,
                total_epochs
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if (
                self.checkpoint_path is not None
                and val_loss < self.best_val_loss
            ):
                self.best_val_loss = val_loss

                checkpoint = {
                    "epoch": current_epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "history": self.history,
                    "best_val_loss": self.best_val_loss
                }

                torch.save(
                    checkpoint,
                    self.checkpoint_path
                )

            progress_bar.update(1)

            progress_bar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}"
            )

        progress_bar.close()

        return self.history