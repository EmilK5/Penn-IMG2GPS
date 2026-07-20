import torch
from torch import nn
from torchvision import models
from typing import Any, Iterable, List


class ResidualBlock(nn.Module):
    def __init__(self, size, dropout_p=0.2):
        super().__init__()
        self.fc = nn.Linear(size, size)
        self.bn = nn.BatchNorm1d(size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.bn(out)
        out = self.gelu(out)
        out = self.dropout(out)
        return out + residual

class EfficientNetGPS(nn.Module):
    def __init__(self):
        super(EfficientNetGPS, self).__init__()

        # Backbone: EfficientNetB0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Readout: Custom Linear Model with Residual Block
        in_features = self.backbone.classifier[1].in_features

        # Replace the classifier with the optimized head from Trial 6
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.27),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            ResidualBlock(512, 0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        if self.training:
            return self.backbone(x)
        else:
            # TTA: Testing Time Augmentation (Averages 3 views for better stability)
            out_orig = self.backbone(x)

            # Zoomed View
            x_zoom = torch.nn.functional.interpolate(
                x[:, :, 16:-16, 16:-16], size=(224, 224), mode='bilinear', align_corners=False
            )
            out_zoom = self.backbone(x_zoom)

            # Brightness Boost
            x_bright = torch.clamp(x * 1.15, min=-3.0, max=3.0)
            out_bright = self.backbone(x_bright)

            return (out_orig + out_zoom + out_bright) / 3


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = EfficientNetGPS()
        
        # Hardcoded High-Precision Constants for Penn Campus (Calculated with Delta 0.01)
        # Origin: (Latitude, Longitude)
        self.register_buffer('origin', torch.tensor([39.95032175798604, -75.19261019047501], dtype=torch.float32))
        
        # Scale: (meters_per_lat, meters_per_lon)
        self.register_buffer('scale', torch.tensor([110989.47192002088, 85421.60655702464], dtype=torch.float32))

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        result = super().load_state_dict(state_dict, strict)
        
        # Restore coordinates immediately after load
        self.origin.copy_(torch.tensor([39.95032175798604, -75.19261019047501]))
        self.scale.copy_(torch.tensor([110989.47192002088, 85421.60655702464]))
        
        return result

    def forward(self, x):
        # local_meters[:, 0] = Easting (X)
        # local_meters[:, 1] = Northing (Y)
        local_meters = self.base_model(x)
        
        dx_easting = local_meters[:, 0]
        dy_northing = local_meters[:, 1]

        # Convert relative meters to GPS degree offsets
        lat_offset = dy_northing / self.scale[0]
        lon_offset = dx_easting / self.scale[1]

        return torch.stack([self.origin[0] + lat_offset, self.origin[1] + lon_offset], dim=1)

    def eval(self) -> None:
        super().eval()

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        self.eval() 
        device = next(self.parameters()).device

        if isinstance(batch, torch.Tensor):
            inputs = batch.to(device)
        else:
            inputs = torch.stack(list(batch)).to(device)

        with torch.no_grad():
            preds = self.forward(inputs)
            
        return preds.cpu().tolist()

def get_model() -> Model:
    return Model()