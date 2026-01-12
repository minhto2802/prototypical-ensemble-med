from glob import glob

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from transformers import BertModel
from transformers import BertForSequenceClassification

from .loss_functions import IsoMaxPlusLossFirstPart
from .datasets_med import IMAGE_DATASETS


class BertFeatureWrapper(torch.nn.Module):

    def __init__(self, model, hparams=None):
        super().__init__()
        if hparams is None:
            hparams = {'last_layer_dropout': .0}
        self.model = model
        self.n_outputs = model.config.hidden_size
        classifier_dropout = (
            hparams['last_layer_dropout'] if hparams['last_layer_dropout'] != 0. else model.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, x):
        kwargs = {
            'input_ids': x[:, :, 0],
            'attention_mask': x[:, :, 1]
        }
        if x.shape[-1] == 3:
            kwargs['token_type_ids'] = x[:, :, 2]
        output = self.model(**kwargs)
        if hasattr(output, 'pooler_output'):
            return self.dropout(output.pooler_output)
        else:
            return self.dropout(output.last_hidden_state[:, 0, :])


def get_backbone(dataset_name, pretrained_in=False, model_name='resnet50'):
    if dataset_name in IMAGE_DATASETS:
        weights = 'IMAGENET1K_V2' if pretrained_in else None
        match model_name:
            case 'resnet50':
                model = torchvision.models.resnet50(weights=weights)
            case 'resnet152':
                model = torchvision.models.resnet152(weights=weights)
            case 'resnext101':
                model = torchvision.models.resnext101_64x4d(weights=weights)
                breakpoint()
        backbone = torch.nn.Sequential(*list(model.children())[:-1])
        emb_dim = model.fc.in_features
    elif dataset_name in TEXT_DATASETS:
        backbone = BertFeatureWrapper(BertModel.from_pretrained('bert-base-uncased'))
        # backbone = BertFeatureWrapper(
        #     BertForSequenceClassification.from_pretrained('bert-base-uncased'))
        emb_dim = backbone.n_outputs
    else:
        raise ValueError(f'Dataset {dataset_name} not supported.')
    return backbone, emb_dim


def get_model(dataset_name, num_classes, train_mode='full',
              pretrained_path=None, pretrained_in=False, loss_name='ce',
              model=None, verbose=True, resume=False, num_concepts=4, concept_hidden_dim=128,
              model_name='resnet50', emb_dim=None):
    if resume:
        assert pretrained_path is not None

    def overwrite(default, value):
        if value is not None:
            return value
        return default

    ckpt = None
    if pretrained_path is not None:
        if '*' in pretrained_path:
            pretrained_path = glob(pretrained_path)
            assert len(pretrained_path) == 1
            pretrained_path = pretrained_path[0]
        ckpt = torch.load(pretrained_path, map_location="cpu")

    if model is None:  # Stage 0 or training resume
        backbone, _emb_dim = get_backbone(dataset_name, pretrained_in, model_name=model_name)
        emb_dim = overwrite(_emb_dim, emb_dim)

        if (((ckpt is not None) and ('prototypes' in [k.split('.')[-1] for k in ckpt.keys()])) or
                (loss_name == 'isomax')):
            head = IsoMaxPlusLossFirstPart(emb_dim, num_classes)
        elif loss_name == 'ce_concept':

            # head = nn.Sequential(
            #     nn.Linear(num_concepts, concept_hidden_dim),
            #     nn.Linear(concept_hidden_dim, num_classes)
            # )
            head = MLP(input_dim=num_concepts, num_classes=num_classes, expand_dim=0)
            concept_head = nn.Linear(emb_dim, num_concepts)
            # for param in concept_head.parameters():
            #     param.requires_grad = False
            model = nn.Sequential(backbone, nn.Flatten(), concept_head, head)
            if ckpt is not None:
                backbone.load_state_dict(ckpt, strict=False)
                ckpt = None
        else:
            head = nn.Linear(emb_dim, num_classes)
        if model is None:
            model = nn.Sequential(backbone, nn.Flatten(), head)
    else:  # ongoing training with model is passed from the previous stage
        backbone = torch.nn.Sequential(*list(model.children())[:-1])  # extract the current backbone
        head = None

    if ckpt is not None:
        try:
            model.load_state_dict(ckpt, strict=True)
        # # weight mismatched
        except Exception as e:
            print(e)
            print('Load partial state dict...')
            model.load_state_dict(ckpt, strict=False)

    if head is None:
        if not resume:  # Create a new a head for training the next stage
            if hasattr(model[-1], 'prototypes'):
                _emb_dim = model[-1].prototypes.shape[1]
            else:
                _emb_dim = model[-1].in_features
            emb_dim = overwrite(_emb_dim, emb_dim)
        print(f'Building a new classifier ({loss_name})...')
        if loss_name == 'isomax':
            head = IsoMaxPlusLossFirstPart(emb_dim, num_classes)  # head is reset (even when loading checkpoint)
        elif loss_name == 'ce':
            head = nn.Linear(emb_dim, num_classes, bias=False)
        else:
            raise ValueError('loss_name must be either "ce" or "isomax"')
        model = nn.Sequential(backbone, nn.Flatten(), head)

    if train_mode == 'freeze':
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()  # Freeze everything including batchnorm
        # backbone.train()

    if verbose:
        pytorch_total_params = sum(p.numel() for p in model.parameters()) / 1e6
        pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable params: {pytorch_total_trainable_params}/{pytorch_total_params:.2f}M')

    return model


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, expand_dim):
        super(MLP, self).__init__()
        self.expand_dim = expand_dim
        if self.expand_dim:
            self.linear = nn.Linear(input_dim, expand_dim)
            self.activation = torch.nn.ReLU()
            self.linear2 = nn.Linear(expand_dim, num_classes) #softmax is automatically handled by loss function
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        if hasattr(self, 'expand_dim') and self.expand_dim:
            x = self.activation(x)
            x = self.linear2(x)
        return x


class PrototypeTransformer(nn.Module):
    def __init__(self, prototypical_ensemble, dist_ensemble, num_layers=2, num_heads=4,
                 ff_dim=128, dropout=0.1, d_model=32, train_mode='full'):
        super(PrototypeTransformer, self).__init__()
        num_classes, num_prototypes, prototype_dim = prototypical_ensemble.shape

        self.num_prototypes = num_prototypes
        self.num_classes = num_classes

        # prototypical_ensemble shape: (num_classes, num_prototypes, prototype_dim)
        module_list = []
        requires_grad = False if train_mode == 'freeze' else True
        for i in range(num_prototypes):
            module = IsoMaxPlusLossFirstPart(prototype_dim, num_classes)
            module.prototypes = torch.nn.Parameter(prototypical_ensemble[:, i], requires_grad=requires_grad)
            module.distance_scale = torch.nn.Parameter(dist_ensemble[i], requires_grad=requires_grad)
            module_list.append(module)
        self.prototypes = nn.ModuleList(module_list)

        # Project the 7-dimensional logit vector into a d_model-dim embedding for the transformer.
        self.logit_proj = nn.Linear(num_classes, d_model)

        # A learnable classification token that will aggregate the info.
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embeddings for the tokens (1 CLS token + 40 prototype tokens = 41 tokens).
        self.pos_embedding = nn.Parameter(torch.randn(1, num_prototypes+1, d_model))

        # Transformer encoder to aggregate tokens.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            # batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        prototype_logits = torch.stack([proto(x) for proto in self.prototypes], dim=1)

        # Project the num_classes-dimensional logit vector into a d_model-dim embedding for the transformer.
        tokens = self.logit_proj(prototype_logits)

        # Prepend the CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1) # (batch_size, num_prototypes+1, d_model)

        # Add positional embeddings
        tokens = tokens + self.pos_embedding # (batch_size, num_prototypes+1, d_model)

        # Transformer expects (sequence, batch_size, d_model)
        tokens = tokens.transpose(0, 1)
        transformed_tokens = self.transformer_encoder(tokens)

        # Use the output corresponding to the CLS token as the aggregated representation
        agg = transformed_tokens[0]  # (batch_size, d_model)

        # Final classification logits
        out_logits = self.classifier(agg)

        return out_logits


class PrototypeTransformerV1(nn.Module):
    def __init__(self, prototypical_ensemble, dist_ensemble, num_layers=2, num_heads=4,
                 ff_dim=512, dropout=0.1, d_model=128, train_mode='full'):
        super(PrototypeTransformerV1, self).__init__()
        num_classes, num_prototypes, prototype_dim = prototypical_ensemble.shape

        self.num_prototypes = num_prototypes
        self.num_classes = num_classes

        # prototypical_ensemble shape: (num_classes, num_prototypes, prototype_dim)
        module_list = []
        requires_grad = False if train_mode == 'freeze' else True
        for i in range(num_prototypes):
            module = IsoMaxPlusLossFirstPart(prototype_dim, num_classes)
            module.prototypes = torch.nn.Parameter(prototypical_ensemble[:, i], requires_grad=requires_grad)
            module.distance_scale = torch.nn.Parameter(dist_ensemble[i], requires_grad=requires_grad)
            module_list.append(module)
        self.prototypes = nn.ModuleList(module_list)

        # Transformer encoder to aggregate tokens.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_classes,
            nhead=num_classes,
            dim_feedforward=ff_dim,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        prototype_logits = torch.stack([proto(x) for proto in self.prototypes], dim=1)

        # Transformer expects (sequence, batch_size, d_model)
        refined_logits = self.transformer_encoder(prototype_logits.transpose(0, 1))

        # Final classification logits
        final_logits = refined_logits.mean(dim=0)

        return final_logits


class PrototypeTransformerCrossAttn(nn.Module):
    def __init__(self, prototypical_ensemble, dist_ensemble, num_layers=2, num_heads=4,
                 ff_dim=128, dropout=0.1, d_model=32, train_mode='full'):
        """
        Args:
            prototypical_ensemble: Tensor of shape (num_classes, num_prototypes, prototype_dim)
            dist_ensemble: List or Tensor with length=num_prototypes (distance scales)
            num_layers (int): Number of transformer decoder layers.
            num_heads (int): Number of attention heads.
            ff_dim (int): Feedforward network dimension.
            dropout (float): Dropout probability.
            d_model (int): Transformer embedding dimension.
            train_mode (str): 'freeze' or 'full' (freeze or train prototypes).
        """
        super(PrototypeTransformerCrossAttn, self).__init__()
        num_classes, num_prototypes, prototype_dim = prototypical_ensemble.shape

        self.num_prototypes = num_prototypes
        self.num_classes = num_classes

        # Create a ModuleList of IsoMaxPlusLossFirstPart modules.
        # Note: IsoMaxPlusLossFirstPart expects (num_features, num_classes).
        module_list = []
        requires_grad = False if train_mode == 'freeze' else True
        for i in range(num_prototypes):
            module = IsoMaxPlusLossFirstPart(prototype_dim, num_classes)
            # Replace the module's prototypes with the given ensemble slice.
            module.prototypes = torch.nn.Parameter(prototypical_ensemble[:, i], requires_grad=requires_grad)
            module.distance_scale = torch.nn.Parameter(dist_ensemble[i], requires_grad=requires_grad)
            module_list.append(module)
        self.prototypes = nn.ModuleList(module_list)

        # Project each prototype's logit vector (of dimension num_classes) into a d_model token.
        self.logit_proj = nn.Linear(num_classes, d_model)

        # Instead of a learnable CLS token, we now project image features to obtain the query token.
        # (Assuming image features have dimension `prototype_dim`.)
        self.query_proj = nn.Linear(prototype_dim, d_model)

        # Positional embeddings for the prototype tokens (there are num_prototypes tokens).
        self.pos_embedding = nn.Parameter(torch.randn(1, num_prototypes, d_model))

        # Transformer decoder layer for cross-attention.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final classifier mapping the aggregated decoder output to class logits.
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, prototype_dim)
               (Image features from your frozen vision encoder)
        Returns:
            out_logits: Tensor of shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]

        # Compute per-prototype logits.
        # Each prototype module returns (batch_size, num_classes)
        # Stacking produces: (batch_size, num_prototypes, num_classes)
        prototype_logits = torch.stack([proto(x) for proto in self.prototypes], dim=1)

        # Project each logit vector into a d_model-dimensional token.
        # Resulting shape: (batch_size, num_prototypes, d_model)
        prototype_tokens = self.logit_proj(prototype_logits)

        # Add positional embeddings.
        prototype_tokens = prototype_tokens + self.pos_embedding

        # Transpose to shape (num_prototypes, batch_size, d_model) for the transformer decoder memory.
        memory = prototype_tokens.transpose(0, 1)

        # Create a query token from the image features.
        # Shape: (batch_size, d_model) -> (1, batch_size, d_model)
        query_token = self.query_proj(x).unsqueeze(0)

        # Use the transformer decoder: query attends to the prototype tokens.
        # The output shape is (1, batch_size, d_model)
        decoded = self.transformer_decoder(tgt=query_token, memory=memory)
        decoded = decoded[0]  # (batch_size, d_model)

        # Final classification logits.
        out_logits = self.classifier(decoded)  # (batch_size, num_classes)
        return out_logits


class PrototypeTransformerFused(nn.Module):
    def __init__(self, prototypical_ensemble, dist_ensemble, num_layers=2, num_heads=4,
                 ff_dim=128, dropout=0.1, d_model=32, train_mode='full'):
        """
        Args:
            prototypical_ensemble: Tensor of shape (num_classes, num_prototypes, prototype_dim)
            dist_ensemble: List/Tensor of length=num_prototypes (distance scales)
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads.
            ff_dim (int): Feedforward network dimension.
            dropout (float): Dropout probability.
            d_model (int): Transformer embedding dimension.
            train_mode (str): 'freeze' or 'full' (whether to freeze the prototypes).
        """
        super(PrototypeTransformerFused, self).__init__()
        num_classes, num_prototypes, prototype_dim = prototypical_ensemble.shape

        self.num_prototypes = num_prototypes
        self.num_classes = num_classes

        # Build a ModuleList of IsoMaxPlusLossFirstPart modules.
        module_list = []
        requires_grad = False if train_mode == 'freeze' else True
        for i in range(num_prototypes):
            # IsoMaxPlusLossFirstPart expects (num_features, num_classes)
            module = IsoMaxPlusLossFirstPart(prototype_dim, num_classes)
            module.prototypes = torch.nn.Parameter(prototypical_ensemble[:, i],
                                                   requires_grad=requires_grad)
            module.distance_scale = torch.nn.Parameter(dist_ensemble[i],
                                                       requires_grad=requires_grad)
            module_list.append(module)
        self.prototypes = nn.ModuleList(module_list)

        # Fusion: We will fuse the raw prototype logits (dim = num_classes)
        # with the cosine similarity row for that prototype (dim = num_prototypes).
        # Thus, the fused token will have dimension (num_classes + num_prototypes).
        self.fusion_proj = nn.Linear(num_classes + num_prototypes, d_model)

        # A learnable CLS token to aggregate information.
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embeddings for tokens (CLS token + prototype tokens).
        self.pos_embedding = nn.Parameter(torch.randn(1, num_prototypes + 1, d_model))

        # Transformer encoder to aggregate tokens.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classification head.
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, prototype_dim)
               (Image features from your frozen vision encoder)
        Returns:
            out_logits: Tensor of shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        # 1. Compute prototype logits.
        # Each prototype module returns (batch_size, num_classes).
        # Stacking gives: (batch_size, num_prototypes, num_classes)
        prototype_logits = torch.stack([proto(x) for proto in self.prototypes], dim=1)

        # 2. Normalize the logits (across classes) to compute cosine similarity.
        proto_norm = F.normalize(prototype_logits, p=2, dim=-1)  # shape: (batch_size, num_prototypes, num_classes)

        # 3. Compute the pairwise cosine similarity between prototype logit vectors.
        # For each sample, this yields a matrix of shape (num_prototypes, num_prototypes).
        cos_sim_matrix = torch.bmm(proto_norm, proto_norm.transpose(1, 2))  # (batch_size, num_prototypes, num_prototypes)

        # 4. For each prototype, take its own raw logit vector (dim = num_classes)
        # and the corresponding cosine similarity row (dim = num_prototypes),
        # then concatenate them to get a fused vector of dimension (num_classes + num_prototypes).
        fused_tokens = torch.cat((prototype_logits, cos_sim_matrix), dim=-1)  # (batch_size, num_prototypes, num_classes+num_prototypes)

        # 5. Project the fused tokens into the transformer embedding space.
        tokens = self.fusion_proj(fused_tokens)  # (batch_size, num_prototypes, d_model)

        # 6. Prepend the CLS token.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        tokens = torch.cat((cls_tokens, tokens), dim=1)  # (batch_size, num_prototypes+1, d_model)

        # 7. Add positional embeddings.
        tokens = tokens + self.pos_embedding  # (batch_size, num_prototypes+1, d_model)

        # 8. Transformer encoder expects input shape: (sequence_length, batch_size, d_model).
        tokens = tokens.transpose(0, 1)
        transformed_tokens = self.transformer_encoder(tokens)

        # 9. Use the output corresponding to the CLS token as the aggregated representation.
        agg = transformed_tokens[0]  # (batch_size, d_model)

        # 10. Final classification logits.
        out_logits = self.classifier(agg)  # (batch_size, num_classes)
        return out_logits
