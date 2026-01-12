import numpy as np

import torch
from torch import nn
from torchvision import models
from transformers import SamModel, SamProcessor

from .isomaxplus import IsoMaxPlusLossFirstPart

EFFICIENTNET_VERSIONS = {
    'b0': (models.efficientnet_b0, 32, 1280, models.EfficientNet_B0_Weights.DEFAULT.transforms),
    'b1': (models.efficientnet_b1, 32, 1280, models.EfficientNet_B1_Weights.DEFAULT.transforms),
    'b2': (models.efficientnet_b2, 32, 1408, models.EfficientNet_B2_Weights.DEFAULT.transforms),
    'b4': (models.efficientnet_b4, 48, 1792, models.EfficientNet_B4_Weights.DEFAULT.transforms),
    's': (models.efficientnet_v2_s, 24, 1280, models.EfficientNet_V2_S_Weights.DEFAULT.transforms),
    'm': (models.efficientnet_v2_s, 24, 1280, models.EfficientNet_V2_M_Weights.DEFAULT.transforms),
}

RESNET_VERSIONS = {
    '18': (models.resnet18, 64, 512, models.ResNet18_Weights.DEFAULT.transforms),
    '34': (models.resnet34, 64, 512, models.ResNet34_Weights.DEFAULT.transforms),
    '50': (models.resnet50, 64, 2048, models.ResNet50_Weights.DEFAULT.transforms),
}


class FeatureExtractor(nn.Module):
    def __init__(self, *args, **kwargs, ):
        super(FeatureExtractor, self).__init__()
        self.model = None
        self.non_frozen_layers = []

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.non_frozen_layers:
            param.requires_grad = True

    def preprocessor(self, x, *args, **kwargs, ):
        outputs = dict()
        outputs['pixel_values'] = self.preprocess(torch.tensor(x.transpose([2, 0, 1])).float())
        return outputs

    def forward(self, x, *args, **kwargs, ):
        x = self.model(x)
        return x


class EfficientNetFeatureExtractor(FeatureExtractor):
    def __init__(self, input_channels=1, num_classes=2, pretrained=True, version='b0', *args, **kwargs, ):
        super(EfficientNetFeatureExtractor, self).__init__()

        version = 'b0' if version not in EFFICIENTNET_VERSIONS.keys() else version
        efficentnet, out_channels, num_features, transforms = EFFICIENTNET_VERSIONS[version]
        self.preprocess = transforms()

        # Load pretrained EfficientNet
        self.efficientnet = efficentnet(pretrained=pretrained)

        # Modify first conv layer for different input channels
        self.efficientnet.features[0][0] = nn.Conv2d(input_channels, out_channels,
                                                     kernel_size=3, stride=2, padding=1, bias=False)

        # Replace classifier
        self.efficientnet.classifier = nn.Linear(num_features, num_classes)

        self.non_frozen_layers = [self.efficientnet.features[0][0], self.efficientnet.classifier]
        self.model = self.efficientnet


class ResNetFeatureExtractor(FeatureExtractor):
    def __init__(self, input_channels=1, num_classes=2, pretrained=True, version='50', *args, **kwargs, ):
        super(ResNetFeatureExtractor, self).__init__()

        version = 'b0' if version not in RESNET_VERSIONS.keys() else version
        _resnet, out_channels, num_features, transforms = RESNET_VERSIONS[version]
        self.preprocess = transforms()

        # Load a small pretrained ResNet (e.g., resnet18)
        self.resnet = _resnet(pretrained=pretrained)

        # Modify the first conv layer to accept a 1-channel input instead of 3
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # New fully connected layer to project to 2048 dimensions
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        self.non_frozen_layers = [self.resnet.conv1, self.resnet.fc]
        self.model = self.resnet


class MedSamFeatureExtractor(nn.Module):
    def __init__(self, opt=None, device='gpu', *args, **kwargs):
        assert opt is not None, "opt must be provided"
        super(MedSamFeatureExtractor, self).__init__()

        model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base",
                                         num_labels=opt.num_classes)  # .to('cuda:0')

        input_size = opt.input_size[0] if self.force_resolution else 1024
        self.preprocess = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base",
                                                       size={"longest_edge": input_size},
                                                       pad_size={'height': input_size, 'width': input_size})

        # Access the model's configuration
        if opt.force_resolution:
            ve = model.vision_encoder
            ve.patch_embed.image_size = opt.input_size
            ve.image_size = opt.input_size
            ve.patch_embed.projection = nn.Conv2d(opt.in_channel,
                                                  model.config.vision_config.hidden_size,
                                                  kernel_size=opt.input_size[0] // 64,
                                                  stride=opt.input_size[0] // 64).to(device)

        if opt.tuning == 'lora':
            # Apply LoRA to specific layers
            model = apply_lora_to_sam(
                model, opt.lora_target_layers, r=opt.lora_r, alpha=opt.lora_alpha).to('cuda:0')

            for name, param in model.named_parameters():
                if 'lora' not in name:
                    param.requires_grad = False
                if opt.force_resolution and name.startswith("vision_encoder.patch_embed"):
                    param.requires_grad = True
        elif opt.tuning == 'partial':
            # make sure we only compute gradients for mask decoder
            if 'sam' in model.config.architectures[-1].lower():
                for name, param in model.named_parameters():
                    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                        param.requires_grad = False
                    if opt.force_resolution and name.startswith("vision_encoder.patch_embed"):
                        param.requires_grad = True
            else:
                # Set requires_grad=False for all parameters to freeze the backbone
                for param in model.parameters():
                    param.requires_grad = False
        self.model = model

    def preprocessor(self, x, *args, **kwargs, ):
        return self.preprocessor(to_0_255(x), input_boxes=None, return_tensors="pt")


class LoRALayer(nn.Module):
    """
    A LoRA (Low-Rank Adaptation) layer applied to a specific SAM model layer.
    """

    def __init__(self, original_layer, r=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Initialize low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(original_layer.weight.size(0), r) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(r, original_layer.weight.size(1)) * 0.01)

    def forward(self, x):
        # Original output
        original_output = self.original_layer(x)

        # LoRA output
        lora_update = x @ self.lora_B.T @ self.lora_A.T  # Ensure correct shape
        # lora_update = torch.matmul(self.lora_A, self.lora_B)
        lora_update = lora_update * self.scaling

        return original_output + lora_update


def to_0_255(_x):
    _x = (_x - _x.min()) / (_x.max() - _x.min())
    _x *= 255
    # _x = _x.astype(np.uint8)
    return _x


def set_model(opt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = opt.model_name
    cnn_models = {
        'resnet': ResNetFeatureExtractor,
        'efficientnet': EfficientNetFeatureExtractor,
    }
    if model_name in cnn_models:
        model = cnn_models[model_name](input_channels=opt.in_channel,
                                       num_classes=opt.num_classes,
                                       version=opt.model_version)
    elif model_name == 'vit':
        model = MedSamFeatureExtractor(opt, device=device).model
    else:
        raise ValueError(f"Model {opt.model} not supported.")

    model.to(device)

    return model


class PrototypeTransformer(nn.Module):
    def __init__(self, prototypical_ensemble, dist_ensemble, num_layers=2, num_heads=4,
                 ff_dim=128, dropout=0.1, d_model=32, train_mode='full'):
        super().__init__()
        num_classes, num_prototypes, prototype_dim = prototypical_ensemble.shape

        self.num_classes = num_classes
        self.num_prototypes = num_prototypes

        requires_grad = (train_mode != 'freeze')
        module_list = []

        for i in range(num_prototypes):
            module = IsoMaxPlusLossFirstPart(prototype_dim, num_classes)
            module.prototypes = nn.Parameter(prototypical_ensemble[:, i], requires_grad=requires_grad)
            module.distance_scale = nn.Parameter(dist_ensemble[i], requires_grad=requires_grad)
            module_list.append(module)

        self.prototypes = nn.ModuleList(module_list)

        # Project prototype logits to transformer input dimension
        self.logit_proj = nn.Sequential(
            nn.LayerNorm(num_classes),
            nn.Linear(num_classes, d_model)
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_prototypes + 1, d_model))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            norm_first=True,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        B = x.size(0)

        # Compute prototype logits for each ensemble head
        prototype_logits = torch.stack([proto(x) for proto in self.prototypes], dim=1)  # [B, P, C]

        # Project logits to transformer token embeddings
        tokens = self.logit_proj(prototype_logits)  # [B, P, d_model]

        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        tokens = torch.cat([cls_token, tokens], dim=1)  # [B, P+1, d_model]

        # Add positional embeddings
        tokens = tokens + self.pos_embedding[:, :tokens.size(1)]
        tokens = self.pos_dropout(tokens)

        # Transformer encoding
        transformed = self.transformer_encoder(tokens)  # [B, P+1, d_model]
        cls_out = transformed[:, 0]  # CLS token output

        # Final classification logits
        logits = self.classifier(cls_out)
        return logits
