import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
from collections import namedtuple

from src.constants import *
from src.models.auxiliary_heads import AuxiliaryGeo, LocationCoordinateHead

# Named Tuple for model outputs
ModelOutput = namedtuple('ModelOutput', 'preds_climate preds_month preds_state preds_county preds_city preds_lat preds_lng')

class GeoResNet(nn.Module):
    def __init__(self, model_name: str = 'resnet50', use_context: bool = True, pretrained: bool = True):
        """Initialize ResNet model with optional auxiliary prediction heads.
        
        Model Description:
            - Base ResNet model
            - Location Coordinate: lat, lon
            - Auxiliary prediction heads (only used if use_context is True)
                - Location Classification: state, county, city
                - Climate Classification: Climate
                - Month Classification: Month
        """
        super(GeoResNet, self).__init__()
        
        # Base ResNet model
        resnet_models = {
            'resnet18': (models.resnet18, 512),
            'resnet34': (models.resnet34, 512),
            'resnet50': (models.resnet50, 2048),
            'resnet101': (models.resnet101, 2048),
            'resnet152': (models.resnet152, 2048)
        }
        
        if model_name not in resnet_models:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
            
        model_fn, self.hidden_size = resnet_models[model_name]
        self.base_model = nn.Sequential(*list(model_fn(pretrained=pretrained).children())[:-1])
        
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
        
        # Freeze base model except last layer
        self._freeze_base_model()
        
    def _freeze_base_model(self):
        """Freeze all layers of base model except the last one."""
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze last layer
        for param in self.base_model[-1].parameters():
            param.requires_grad = True
            
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
        
    def save_pretrained(self, save_directory: str):
        """Save the model's state and configuration to a directory.
        
        Args:
            save_directory (str): Directory to save the model to.
        """
       
        # Create the directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the model's state dict
        torch.save(self.state_dict(), os.path.join(save_directory, "geo_resnet.pt"))
        
        # Save model configuration
        config = {
            "model_name": "resnet50",  # Default for now
            "hidden_size": self.hidden_size,
            "use_context": self.use_context,
            "num_climates": NUM_CLIMATES,
            "num_months": NUM_MONTHS,
            "num_states": NUM_STATES,
            "num_counties": NUM_COUNTIES,
            "num_cities": NUM_CITIES
        }
        
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {save_directory}")
        
    @classmethod
    def from_pretrained(model_path: str, use_context: bool = True):
        model = GeoResNet()
        state_dict = torch.load(os.path.join(model_path, "geo_resnet.pt"))
        model.load_state_dict(state_dict)
        return model