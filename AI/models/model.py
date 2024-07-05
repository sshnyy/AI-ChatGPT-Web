from torch import nn
import timm


class SkinlesionModel(nn.Module):
    def __init__(self, **kwargs):
        super(SkinlesionModel, self).__init__()
        backbone = kwargs["backbone"]
        num_classes = kwargs["num_classes"]
        self.model = timm.create_model(
            model_name=backbone, pretrained=True, num_classes=num_classes
        )

    def forward(self, inp):
        x = self.model.forward(inp)
        return x