import os
import json
import torch
import torch.nn as nn
from transformers import CLIPVisionModel as BaseCLIPModel
from collections import namedtuple
import math
from warnings import filterwarnings

from src.constants import *
from src.models.auxiliary_heads import AuxiliaryGeo 

filterwarnings('ignore', category=UserWarning, module='transformers')

# Named Tuple for model outputs
ModelOutput = namedtuple('ModelOutput', 'preds_climate preds_month preds_state preds_county preds_city preds_lat preds_lng')

class GeoCLIP(nn.Module):
    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32',
                 use_context: bool = True,
                 freeze_base_model: bool = True):
        """Initialize CLIP model with optional auxiliary prediction heads.
        
        Model Description:
            - Base CLIP model
            - Location Coordinate: lat, lon
            - Auxiliary prediction heads (only used if use_context is True)
                - Location Classification: state, county, city
                - Climate Classification: Climate
                - Month Classification: Month
            
        Args:
            model_name (str): Name of the base CLIP model to use or a path to a checkpoint dir.
            use_context (bool): Whether to use contextual features (climate, city, state, etc.)
        """
        super(GeoCLIP, self).__init__()
        
        # Base CLIP model
        self.base_model = BaseCLIPModel.from_pretrained(model_name)
        self.hidden_size = self.base_model.config.hidden_size
        self.use_context = use_context
        
        # Auxiliary prediction heads (only used if use_context is True)
        if use_context:
            self.auxiliary_heads = AuxiliaryGeo (self.hidden_size)
            
            # Calculate input size for location coordinate head with auxiliary features
            aux_input_size = (
                NUM_CLIMATES +      # Climate logits
                NUM_MONTHS +        # Month logits
                NUM_STATES +        # State logits
                NUM_COUNTIES +      # County logits
                NUM_CITIES          # City logits
            )
            self.location_coord_input_size = aux_input_size + self.hidden_size
        else:
            self.location_coord_input_size = self.hidden_size

        # lat/lng prediction head
        self.location_coord_head = nn.Sequential(
            nn.Linear(self.location_coord_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2)  # Outputs [latitude, longitude]
        )

        if freeze_base_model:
            print("Freezing base model (except last layer).")
            self._freeze_base_model()
        
    def _freeze_base_model(self):
        """Freeze all layers of base model except the last one."""
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze last layer
        for param in self.base_model.vision_model.encoder.layers[-1].parameters():
            param.requires_grad = True
            
    def forward(self, pixel_values):
        """Forward pass through the model.
        Returns: ModelOutput: Named tuple containing all predictions
        """
        # Get base model embeddings
        assert pixel_values is not None, "pixel_values must be provided!"
        outputs = self.base_model(pixel_values=pixel_values)
        image_embeddings = outputs.pooler_output
        
        # Get auxiliary predictions if use_context is True
        if self.use_context:
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
        """
       
        # Create the directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the model's state dict
        torch.save(self.state_dict(), os.path.join(save_directory, "geo_clip.pt"))
        
        # Save model configuration
        config = {
            "model_name": self.base_model.name_or_path,
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
            
        # Save the base model configuration
        self.base_model.config.save_pretrained(save_directory)
        
        print(f"Model saved to {save_directory}")
        
    @classmethod
    def from_pretrained(model_path: str, use_context: bool = True):
        model = GeoCLIP()
        state_dict = torch.load(os.path.join(model_path, "geo_clip.pt"))
        model.load_state_dict(state_dict)
        return model