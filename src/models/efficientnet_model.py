import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
from collections import namedtuple

from src.constants import *
from src.models.auxiliary_heads import AuxiliaryGeo, LocationCoordinateHead

# Named Tuple for model outputs (same as in resnet_model)
ModelOutput = namedtuple('ModelOutput', 'preds_climate preds_month preds_state preds_county preds_city preds_lat preds_lng')

class GeoEfficientNet(nn.Module):
    def __init__(self, model_name: str = 'efficientnet_b3', use_context: bool = True, pretrained: bool = True):
        """Initialize EfficientNet-B3 model with optional auxiliary prediction heads.
        
        Model Description:
            - Base EfficientNet-B3 model
            - Location Coordinate: lat, lon
            - Auxiliary prediction heads (only used if use_context is True)
                - Location Classification: state, county, city
                - Climate Classification: Climate
                - Month Classification: Month
        """
        super(GeoEfficientNet, self).__init__()
        
        if model_name != 'efficientnet_b3':
             raise ValueError(f"This class only supports efficientnet_b3, got {model_name}")

        # Load pre-trained EfficientNet-B3
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        base_effnet = models.efficientnet_b3(weights=weights)
        
        # Extract features up to the adaptive average pooling layer
        self.base_model = nn.Sequential(
            base_effnet.features,
            base_effnet.avgpool
        )
        self.hidden_size = 1536 # Output features from EfficientNet-B3 avgpool
        
        self.use_context = use_context
        
        # Auxiliary prediction heads (only used if use_context is True)
        if use_context:
            self.auxiliary_heads = AuxiliaryGeo(self.hidden_size)
            
            # Calculate input size for location coordinate head with auxiliary features
            aux_input_size = (
                self.hidden_size +  # Base embeddings
                NUM_CLIMATES +      # Climate logits
                NUM_MONTHS +        # Month logits
                NUM_STATES +        # State logits
                NUM_COUNTIES +      # County logits
                NUM_CITIES          # City logits
            )
            
            # Location prediction head (using auxiliary features)
            self.location_coord_head = LocationCoordinateHead(aux_input_size, use_auxiliary=True)
        else:
            # Simple coordinate prediction head without context
            self.location_coord_head = LocationCoordinateHead(self.hidden_size, use_auxiliary=False)
        
        # Freeze base model layers (adjust freezing strategy as needed)
        self._freeze_base_model()
        
    def _freeze_base_model(self):
        """Freeze initial layers of the base model (e.g., first 2/3)."""
        # Freeze all features initially
        for param in self.base_model[0].parameters(): # base_model[0] is base_effnet.features
            param.requires_grad = False
            
        # Unfreeze the last few blocks (e.g., last 1/3 of blocks in features)
        # EfficientNet-B3 has 8 blocks in features (0-7)
        num_blocks_to_unfreeze = 3 # Adjust as needed
        total_blocks = 8
        for i in range(total_blocks - num_blocks_to_unfreeze, total_blocks):
             for param in self.base_model[0][i].parameters():
                 param.requires_grad = True

        # Ensure avgpool is always trainable (it has no parameters, but just in case)
        # for param in self.base_model[1].parameters(): # base_model[1] is base_effnet.avgpool
        #    param.requires_grad = True # avgpool typically has no parameters

            
    def forward(self, pixel_values=None, **kwargs):
        """Forward pass through the model.
    
        Args:
            pixel_values: Input images

        Returns:
            ModelOutput: Named tuple containing all predictions
        """
        # Get base model embeddings
        image_embeddings = self.base_model(pixel_values)
        image_embeddings = image_embeddings.view(image_embeddings.size(0), -1)  # Flatten
        
        if self.use_context:
            # Get auxiliary predictions
            aux_output = self.auxiliary_heads(image_embeddings)
            
            # Location coordinate predictions (latitude, longitude)
            location_preds = self.location_coord_head(aux_output.combined_features)
            
            # Extract individual predictions
            climate_logits = aux_output.climate_logits
            month_logits = aux_output.month_logits
            state_logits = aux_output.state_logits
            county_logits = aux_output.county_logits
            city_logits = aux_output.city_logits
        else:
            # Simple coordinate prediction without context
            location_preds = self.location_coord_head(image_embeddings)
            climate_logits = None
            month_logits = None
            state_logits = None
            county_logits = None
            city_logits = None
        
        lat_preds, lng_preds = location_preds[:, 0], location_preds[:, 1]
        
        return ModelOutput(
            preds_climate=climate_logits,
            preds_month=month_logits,
            preds_state=state_logits,
            preds_county=county_logits,
            preds_city=city_logits,
            preds_lat=lat_preds,
            preds_lng=lng_preds
        )
        
    # Note: save_pretrained and from_pretrained methods could be added similarly 
    # to GeoResNet if needed, but are omitted for brevity here. They are not
    # strictly required for the training script to run. 