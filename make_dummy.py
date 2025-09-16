import torch
import torch.nn as nn

# Fake model: just outputs random scores for 2 classes
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3*224*224, 2)  # fake classifier

    def forward(self, x):
        # Flatten and run through linear layer
        x = x.view(x.size(0), -1)
        return self.fc(x)

dummy = DummyModel()
scripted = torch.jit.script(dummy)
scripted.save("model_scripted.pt")
