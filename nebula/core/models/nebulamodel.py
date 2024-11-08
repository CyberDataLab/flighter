from abc import ABC, abstractmethod
import torch
from nebula.addons.functions import print_msg_box
import lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassRecall,
    MulticlassPrecision,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)
from torchmetrics import MetricCollection
import seaborn as sns
import matplotlib.pyplot as plt
from torch import autograd


class NebulaModel(pl.LightningModule, ABC):
    """
    Abstract class for the NEBULA model.

    This class is an abstract class that defines the interface for the NEBULA model.
    """

    def process_metrics(self, phase, y_pred, y, loss=None):
        """
        Calculate and log metrics for the given phase.
        The metrics are calculated in each batch.
        Args:
            phase (str): One of 'Train', 'Validation', or 'Test'
            y_pred (torch.Tensor): Model predictions
            y (torch.Tensor): Ground truth labels
            loss (torch.Tensor, optional): Loss value
        """

        y_pred_classes = torch.argmax(y_pred, dim=1)
        if phase == "Train":
            # self.log(name=f"{phase}/Loss", value=loss, add_dataloader_idx=False)
            self.logger.log_data({f"{phase}/Loss": loss.item()}, step=self.global_step)
            self.train_metrics(y_pred_classes, y)
        elif phase == "Validation":
            self.val_metrics(y_pred_classes, y)
        elif phase == "Test (Local)":
            self.test_metrics(y_pred_classes, y)
            self.cm.update(y_pred_classes, y) if self.cm is not None else None
        elif phase == "Test (Global)":
            self.test_metrics_global(y_pred_classes, y)
            self.cm_global.update(y_pred_classes, y) if self.cm_global is not None else None
        else:
            raise NotImplementedError

    def log_metrics_end(self, phase):
        """
        Log metrics for the given phase.
        Args:
            phase (str): One of 'Train', 'Validation', 'Test (Local)', or 'Test (Global)'
            print_cm (bool): Print confusion matrix
            plot_cm (bool): Plot confusion matrix
        """
        if phase == "Train":
            output = self.train_metrics.compute()
        elif phase == "Validation":
            output = self.val_metrics.compute()
        elif phase == "Test (Local)":
            output = self.test_metrics.compute()
        elif phase == "Test (Global)":
            output = self.test_metrics_global.compute()
        else:
            raise NotImplementedError

        output = {f"{phase}/{key.replace('Multiclass', '').split('/')[-1]}": value for key, value in output.items()}

        self.logger.log_data(output, step=self.global_number[phase])

        # Print test metrics in the log file
        if phase == "Test (Local)" or phase == "Test (Global)":
            metrics_str = ""
            for key, value in output.items():
                metrics_str += f"{key}: {value:.4f}\n"
            print_msg_box(metrics_str, indent=2, title=f"{phase} Metrics | Step: {self.global_number[phase]}")

    def generate_confusion_matrix(self, phase, print_cm=False, plot_cm=False):
        """
        Generate and plot the confusion matrix for the given phase.
        Args:
            phase (str): One of 'Train', 'Validation', 'Test (Local)', or 'Test (Global)'
            :param phase:
            :param print:
            :param plot:
        """
        if phase == "Test (Local)":
            if self.cm is None:
                raise ValueError(f"Confusion matrix not available for {phase} phase.")
            cm = self.cm.compute().cpu()
        elif phase == "Test (Global)":
            if self.cm_global is None:
                raise ValueError(f"Confusion matrix not available for {phase} phase.")
            cm = self.cm_global.compute().cpu()
        else:
            raise NotImplementedError

        print(f"\n{phase}/ConfusionMatrix\n", cm) if print_cm else None
        if plot_cm:
            # TODO: Improve with strings for class names
            cm_numpy = cm.numpy()
            cm_numpy = cm_numpy.astype(int)
            classes = [i for i in range(self.num_classes)]
            fig, ax = plt.subplots(figsize=(10, 10))
            ax = plt.subplot()
            sns.heatmap(cm_numpy, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted labels")
            ax.set_ylabel("True labels")
            ax.set_title("Confusion Matrix")
            ax.xaxis.set_ticklabels(classes, rotation=90)
            ax.yaxis.set_ticklabels(classes, rotation=0)
            self.logger.log_figure(fig, step=self.global_number[phase], name=f"{phase}/CM")
            plt.close()
        self.cm.reset() if phase == "Test (Local)" else self.cm_global.reset()

    def __init__(
        self,
        input_channels=1,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        fisher_threshold=0.4
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        if metrics is None:
            metrics = MetricCollection(
                [
                    MulticlassAccuracy(num_classes=num_classes),
                    MulticlassPrecision(num_classes=num_classes),
                    MulticlassRecall(num_classes=num_classes),
                    MulticlassF1Score(num_classes=num_classes),
                ]
            )
        self.train_metrics = metrics.clone(prefix="Train/")
        self.val_metrics = metrics.clone(prefix="Validation/")
        self.test_metrics = metrics.clone(prefix="Test (Local)/")
        self.test_metrics_global = metrics.clone(prefix="Test (Global)/")
        del metrics
        if confusion_matrix is None:
            self.cm = MulticlassConfusionMatrix(num_classes=num_classes)
            self.cm_global = MulticlassConfusionMatrix(num_classes=num_classes)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.global_number = {"Train": 0, "Validation": 0, "Test (Local)": 0, "Test (Global)": 0}
        self.fisher_threshold = fisher_threshold

    @abstractmethod
    def forward(self, x):
        """Forward pass of the model."""
        pass

    @abstractmethod
    def configure_optimizers(self):
        """Optimizer configuration."""
        pass

    def step(self, batch, batch_idx, phase):
        """Training/validation/test step."""
        x, y = batch
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        self.process_metrics(phase, y_pred, y, loss)

        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        Args:
            batch:
            batch_id:

        Returns:
        """
        return self.step(batch, batch_idx=batch_idx, phase="Train")

    def on_train_end(self):
        self.log_metrics_end("Train")
        self.train_metrics.reset()
        self.global_number["Train"] += 1
        
        fisher_diag = self.compute_fisher_diag(self, self.train_dataloader)
        
        for param in self.parameters():
            u_param = (param * (fisher_diag > self.fisher_threshold)).clone().detach()
            param.data = u_param
    
        import torch
    from torch import autograd
    
    def compute_fisher_information(model, dataloader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        
        # Initialize Fisher Information matrix with zeros
        fisher_information = [torch.zeros_like(param) for param in model.parameters()]
    
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
    
            # Calculate output log probabilities
            log_probs = torch.nn.functional.log_softmax(model(data), dim=1)
    
            for i, label in enumerate(labels):
                log_prob = log_probs[i, label]
    
                # Calculate first-order derivatives (gradients)
                model.zero_grad()
                gradients = autograd.grad(log_prob, model.parameters(), create_graph=True, retain_graph=True)
    
                # Update Fisher Information matrix
                for fisher_value, grad in zip(fisher_information, gradients):
                    fisher_value.add_(grad.detach() ** 2)
                    
                # Free up memory by removing computation graph
                del log_prob, gradients
    
        # Calculate the mean value
        num_samples = len(dataloader.dataset)
        fisher_information = [fisher_value / num_samples for fisher_value in fisher_information]
    
        # Normalize Fisher values layer-wise
        normalized_fisher_information = []
        for fisher_value in fisher_information:
            x_min = torch.min(fisher_value)
            x_max = torch.max(fisher_value)
            normalized_fisher_value = (fisher_value - x_min) / (x_max - x_min)
            normalized_fisher_information.append(normalized_fisher_value)
    
        return normalized_fisher_information

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        Args:
            batch:
            batch_idx:

        Returns:
        """
        return self.step(batch, batch_idx=batch_idx, phase="Validation")

    def on_validation_end(self):
        self.log_metrics_end("Validation")
        self.val_metrics.reset()
        self.global_number["Validation"] += 1

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        """
        Test step for the model.
        Args:
            batch:
            batch_idx:

        Returns:
        """
        if dataloader_idx == 0:
            return self.step(batch, batch_idx=batch_idx, phase="Test (Local)")
        else:
            return self.step(batch, batch_idx=batch_idx, phase="Test (Global)")

    def on_test_end(self):
        self.log_metrics_end("Test (Local)")
        self.log_metrics_end("Test (Global)")
        self.generate_confusion_matrix("Test (Local)", print_cm=True, plot_cm=True)
        self.generate_confusion_matrix("Test (Global)", print_cm=True, plot_cm=True)
        self.test_metrics.reset()
        self.test_metrics_global.reset()
        self.global_number["Test (Local)"] += 1
        self.global_number["Test (Global)"] += 1

    def on_test_epoch_end(self):
        pass
