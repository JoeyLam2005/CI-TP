import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import make_grid
from domainbed.lib.cvt import tiny_cvt, small_cvt
from domainbed.lib.t2t_vit import t2t_vit_t_14
from domainbed.lib.t2t_vit import *
from domainbed.lib.t2t_utils import load_for_transfer_learning
import clip
from domainbed.prs_hook import hook_prs_logger
from domainbed.factory import create_model_and_transforms
from .mixstyle import MixStyle

trans1 = T.ToTensor()
import copy
import numpy as np
from .visiontransformer import DecTransformer
from .fourier import colorSpectrumMix

from torchvision.transforms import Normalize

try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, ParamDict, MovingAverage, l2_between_dicts
)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel, BlipModel
from PIL import Image

ALGORITHMS = [
    'CI_TP'
]
def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

def freeze_model_parameters(model: nn.Module):
    """
    Freeze the parameters of the given model, so that they are not updated during training.
    
    Args:
    model (nn.Module): The model whose parameters are to be frozen.
    """
    for param in model.parameters():
        param.requires_grad = False

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class CI_TP(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__(input_shape, num_classes, num_domains, hparams)

        self.k = hparams["k"]
        self.style_suffixes = hparams.get("style_suffixes", [
            # 10 style
            "in art painting style",
            "in cartoon style",
            "in sketch style",
            "in photo style",
            "in realistic style",
            "in minimalist design style",
            "in retro 90s style",
            "in pixel art style",
            "in surrealist painting style",
            "in abstract geometric style",
            # 20 background
            "during the day",
            "at dawn",
            "at dusk",
            "at night",
            "under the moonlight",
            "at sunset",
            "at sunrise",
            "at midday",
            "at the golden hour",
            "under the starlit sky",
            "on a sunny day",
            "on a rainy day",
            "on a windy day",
            "in a snowstorm",
            "on a cloudy day",
            "on a foggy morning",
            "during a thunderstorm",
            "under a clear blue sky",
            "during a windstorm",
            "on a misty afternoon"
        ])

        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        if hasattr(self.blip_processor, 'image_processor') and self.blip_processor.image_processor is not None:
            self.blip_processor.image_processor.do_rescale = False
            print("Successfully set blip_processor.image_processor.do_rescale = False")
        else:
            print("Warning: Could not directly access 'self.blip_processor.image_processor' to set 'do_rescale'. " +
                  "Please verify your Transformers library version and BlipProcessor structure " +
                  "if rescaling issues persist. Consider re-initializing BlipProcessor with a " +
                  "custom BlipImageProcessor where do_rescale is set to False.")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_text_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")

        self.model = CausalNet_TextAdapted(input_shape, num_classes, num_domains, hparams)

        base_lr = self.hparams["lr"]
        lr_backbone = self.hparams.get("lr_backbone", base_lr / 10.0) 
        lr_text_modules = self.hparams.get("lr_text_modules", base_lr) 
        lr_combined_modules = self.hparams.get("lr_combined_modules", base_lr ) 

        optimizer_grouped_parameters = [
            {"params": self.model.network.parameters(), "lr": lr_backbone},
            {"params": self.model.text_reduction_fc.parameters(), "lr": lr_text_modules},
            {"params": self.model.text_path_classifier_fc.parameters(), "lr": lr_text_modules},
            {"params": self.model.combined_classifier_mlp.parameters(), "lr": lr_combined_modules}
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            weight_decay=self.hparams['weight_decay']
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.normalization_transform = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self._setup_device()

        self.lr_scheduler = None

    def _setup_device(self):
        self.blip_model = self.blip_model.to(self.device)
        self.blip_text_model = self.blip_text_model.to(self.device)
        self.model = self.model.to(self.device)

    def _generate_text_features(self, x):

        B = x.shape[0]

        # Batch generate caption
        x_cpu = x.detach().cpu()
        inputs = self.blip_processor(images=list(x_cpu), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.blip_model.generate(**inputs, do_sample=False, max_length= 50)
        base_captions = [
            self.blip_processor.decode(out[i], skip_special_tokens=True)
            for i in range(B)
        ]

        # Paste the suffix
        all_styled_captions = []
        for cap in base_captions:
            styled_caps = [f"{cap} {suffix}" for suffix in self.style_suffixes[:self.k]]
            all_styled_captions.extend(styled_caps)

        # Use BLIP to extract text features
        text_inputs = self.blip_processor(
            text=all_styled_captions,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_features_all = self.blip_text_model.get_text_features(**text_inputs)

        text_features_batch = text_features_all.view(B, self.k, -1)
        return text_features_batch

    def update(self, minibatches, style_minibatches, unlabeled=None):
        x, y = minibatches
        with torch.no_grad():
            text_features = self._generate_text_features(x)
        final_logits1, final_logits2 = self.model(x, text_features)

        loss1 = F.cross_entropy(final_logits1, y)
        loss2 = F.cross_entropy(final_logits2, y)

        total_loss = loss1 + loss2

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {'loss': total_loss.item(), 'loss1': loss1.item(), 'loss2': loss2.item()}

    def predict(self, x, style_minibatches=None):
        with torch.no_grad():
            text_features = self._generate_text_features(x)
            final_logits1, _ = self.model(x, text_features)
        return final_logits1

    def update_lr(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

class VisualFeatureExpectationModule(nn.Module):
    def __init__(self, feature_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, 
                                               num_heads=num_heads, 
                                               dropout=dropout, 
                                               batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        if x.dim() == 4: 
            B, C, H, W = x.shape
            x = x.view(B, C, -1).permute(0, 2, 1) 
        elif x.dim() == 2:
            x = x.unsqueeze(1) 

        attn_output, _ = self.attention(query=x, key=x, value=x)
        x = self.norm(x + attn_output)

        if x.size(1) > 1:
            x = x.mean(dim=1) 
        else:
            x = x.squeeze(1)
        return x

class CausalNet_TextAdapted(nn.Module):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super().__init__()

        self.image_feature_dim = hparams.get("image_feature_dim", 2048)

        self.text_feature_dim = hparams.get("text_feature_dim", 512)

        self.text_reduced_dim = hparams.get("text_reduced_dim", 256)

        self.network = return_backbone_network(hparams['backbone'], num_classes, hparams, input_shape=(3, 224, 224))
        
        self.visual_expectation_module = VisualFeatureExpectationModule(
            feature_dim=self.image_feature_dim,
            num_heads=4 
        )

        self.text_reduction_fc = nn.Linear(self.text_feature_dim, self.text_reduced_dim)

        self.text_path_classifier_fc = nn.Linear(self.text_reduced_dim, num_classes)

        combined_input_dim = self.image_feature_dim + self.text_reduced_dim

        self.combined_classifier_mlp = nn.Sequential(
            nn.Linear(combined_input_dim, 1152), 
            nn.ReLU(),
            nn.Linear(1152, num_classes)
        )

    def forward(self, x, text_features):

        raw_image_feats = self.network(x) 
        if raw_image_feats.dim() == 4 and raw_image_feats.shape[2] == 1:
             raw_image_feats = raw_image_feats.squeeze(-1).squeeze(-1)

        image_feats_b = self.visual_expectation_module(raw_image_feats)

        B, K, D_text_actual = text_features.shape

        all_logits1 = []
        all_logits2 = []

        for i in range(K):
            current_text_feat_i = text_features[:, i, :]

            combined_feat_i = torch.cat([image_feats_b, current_text_feat_i], dim=1)

            reduced_text_feat_i = F.relu(self.text_reduction_fc(current_text_feat_i))

            logits2_i = self.text_path_classifier_fc(reduced_text_feat_i)
            all_logits2.append(logits2_i)

            combined_feat_final = torch.cat([image_feats_b, reduced_text_feat_i], dim=1) 
            logits1_i = self.combined_classifier_mlp(combined_feat_final)
            
            all_logits1.append(logits1_i)
    
        logits1_stack = torch.stack(all_logits1, dim=1)  
        final_logits1 = logits1_stack.mean(dim=1)      

        logits2_stack = torch.stack(all_logits2, dim=1) 
        final_logits2 = logits2_stack.mean(dim=1)      

        return final_logits1, final_logits2


def return_backbone_network(network_name, num_classes, hparams, input_shape=None):
    if (network_name == "DeitSmall"):
        network = torch.hub.load('./domainbed/pretrained_models/DeiT_models', 'deit_small_patch16_224',
                                 pretrained=True, source='local', in_chans=input_shape[0])
        network.head = nn.Linear(384, num_classes)
        return network
    elif (network_name == "CVTSmall"):
        network = small_cvt(pretrained=True)
        network.head = nn.Linear(384, num_classes)
        return network
    elif (network_name == "T2T14"):
        network = t2t_vit_t_14()
        # load the pretrained weights
        pretrained_path = "./domainbed/pretrained_models/t2t/81.7_T2T_ViTt_14.pth"
        load_for_transfer_learning(network, pretrained_path, use_ema=True, strict=True, num_classes=1000)
        network.head = nn.Linear(384, num_classes)
        return network
    elif network_name == "CLIP_ViT":
        # device = "cuda" if torch.cuda.is_available() else "cpufe"
        clip_model, preprocess = clip.load("ViT-B/16", device=device)
        return clip_model.visual
    elif network_name == 'ResNet50':
        hparams['resnet18'] = False
        return networks.Featurizer(input_shape, hparams)
    elif network_name == 'ResNet18':
        hparams['resnet18'] = True  
        return networks.Featurizer(input_shape, hparams)
    elif network_name == 'CLIP_ResNet50':
        clip_model, preprocess = clip.load("RN50", device=device)
        return clip_model.visual
    elif network_name == 'CLIP_ViT_all':
        clip_model, preprocess = clip.load("ViT-B/16", device=device)
        return clip_model
    elif network_name == 'CLIP_ViT_PRS':
        model, _, preprocess = create_model_and_transforms(
            'ViT-B-16', pretrained='laion2b_s34b_b88k'
        )
        return model, preprocess
