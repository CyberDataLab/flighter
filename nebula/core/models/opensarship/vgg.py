import torch
import torch.nn as nn
from torchvision import models
from nebula.core.models.nebulamodel import NebulaModel


class OpenSarShipModelVGG(NebulaModel):
    def __init__(
        self,
        input_channels=64,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)

        self.example_input_array = torch.zeros((1, input_channels, 128, 128))
        self.learning_rate = learning_rate
        self.momentum = 0.7
        self.weight_decay = 4e-3
        self.criterion = torch.nn.CrossEntropyLoss()

        self.features = models.vgg16(pretrained=False).features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Global Average Pooling
        self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 1)  # 1 output
            )


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1)
        return [optimizer], [lr_scheduler]
