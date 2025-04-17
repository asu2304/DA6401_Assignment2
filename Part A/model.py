# importing just relevant libs
import torch
import torch.nn as nn




class FlexibleCNN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        conv_filters=[32, 64, 128, 256, 512],
        kernel_sizes=[3, 3, 3, 3, 3],
        activation_fn_cnn='relu',
        activation_fn_dense='relu',
        dense_neurons=256,
        dropout_p=0.3,
        use_batchnorm=True,
        input_size=224
    ):
        super(FlexibleCNN, self).__init__()
        
        activation_map = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU
        }

        self.activation_cnn = activation_map[activation_fn_cnn]()
        self.activation_dense = activation_map[activation_fn_dense]()
        
        layers = []
        prev_channels = in_channels
        for i in range(5):
            padding = kernel_sizes[i] // 2
            layers.extend([
                nn.Conv2d(prev_channels, conv_filters[i], kernel_size=kernel_sizes[i], padding=padding),
                nn.BatchNorm2d(conv_filters[i]) if use_batchnorm else nn.Identity(),
                self.activation_cnn,
                nn.MaxPool2d(2)
            ])
            prev_channels = conv_filters[i]

        self.conv = nn.Sequential(*layers)
        
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            dummy = self.conv(dummy)
            self.flatten_size = dummy.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, dense_neurons),
            self.activation_dense,
            nn.Dropout(dropout_p),
            nn.Linear(dense_neurons, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.classifier(x)
