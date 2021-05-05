import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class ResNetBertSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetBertSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.visual_backbone = self._get_basemodel(base_model)
        
        dim_mlp_visual = self.visual_backbone.fc.in_features

        dim_mlp_language = 768

        # add mlp projection head for vision
        self.visual_backbone.fc = nn.Sequential(nn.Linear(dim_mlp_visual, dim_mlp_visual), nn.ReLU(), self.visual_backbone.fc)

        # add mlp projection head for language
        self.language_backbone = nn.Sequential(nn.Linear(dim_mlp_language, dim_mlp_language), nn.ReLU(), nn.Linear(dim_mlp_language, out_dim))

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, images1, images2, caption_encoding):
        visual_embedding1 = self.visual_backbone(images1)
        visual_embedding2 = self.visual_backbone(images2)
        sentence_embedding = self.language_backbone(caption_encoding)

        return visual_embedding1, visual_embedding2, sentence_embedding

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
