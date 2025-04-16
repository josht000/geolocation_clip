import os
import json
import torch
import torch.nn as nn
from transformers import CLIPVisionModel as BaseCLIPModel
from collections import namedtuple
import math
from warnings import filterwarnings

from src.constants import *

filterwarnings('ignore', category=UserWarning, module='transformers')

# Named Tuple for model outputs
ModelOutput = namedtuple('ModelOutput', 'preds_climate preds_month preds_state preds_county preds_city preds_lat preds_lng')

def l2_distance_tensor(pred_coords, true_coords):
    """Calculate L2 distance between predicted and true coordinates using tensors.
    
    Args:
        pred_coords: Tensor of shape [batch_size, 2] containing predicted [lat, lng]
        true_coords: Tensor of shape [batch_size, 2] containing true [lat, lng]
        
    Returns:
        Tensor of shape [batch_size] containing L2 distances
    """
    return torch.sqrt(torch.sum((pred_coords - true_coords) ** 2, dim=1))

class GeoCLIP(nn.Module):
    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32', use_context: bool = True):
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
            self.climate_head = nn.Linear(self.hidden_size, NUM_CLIMATES)
            self.month_head = nn.Linear(self.hidden_size, NUM_MONTHS)
            
            # Hierarchical location classification head
            self.location_class_head = nn.Sequential(
                nn.Linear(self.hidden_size, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, NUM_STATES + NUM_COUNTIES + NUM_CITIES)  # Combined output for all location levels
            )
            
            # Location prediction head (using auxiliary features)
            self.location_coord_head = nn.Sequential(
                nn.Linear(self.hidden_size + NUM_CLIMATES + NUM_MONTHS + NUM_STATES + NUM_COUNTIES + NUM_CITIES, 512),
                nn.ReLU(),
                nn.Linear(512, 2)  # Outputs [latitude, longitude]
            )
        else:
            # Simple coordinate prediction head without context
            self.location_coord_head = nn.Sequential(
                nn.Linear(self.hidden_size, 512),
                nn.ReLU(),
                nn.Linear(512, 2)  # Outputs [latitude, longitude]
            )
        
        # Freeze base model except last layer
        self._freeze_base_model()
        
    def _freeze_base_model(self):
        """Freeze all layers of base model except the last one."""
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze last layer
        for param in self.base_model.vision_model.encoder.layers[-1].parameters():
            param.requires_grad = True
            
    def forward(self, pixel_values=None, **kwargs):
        """Forward pass through the model.
        
        Args:
            pixel_values: Input images
            **kwargs: Additional arguments that may be provided by the dataloader
                     (these are ignored but allow for flexible input)
            
        Returns:
            ModelOutput: Named tuple containing all predictions
        """
        # Get base model embeddings
        outputs = self.base_model(
            pixel_values=pixel_values,
        )
        
        image_embeddings = outputs.pooler_output
        
        if self.use_context:
            # Auxiliary predictions
            climate_logits = self.climate_head(image_embeddings).to(torch.float32)
            month_logits = self.month_head(image_embeddings).to(torch.float32)
            
            # Hierarchical location predictions
            location_logits = self.location_class_head(image_embeddings)
            state_logits = location_logits[:, :NUM_STATES]
            county_logits = location_logits[:, NUM_STATES:NUM_STATES + NUM_COUNTIES]
            city_logits = location_logits[:, NUM_STATES + NUM_COUNTIES:]
            
            # Concatenate all features for coordinate prediction
            combined_features = torch.cat([
                image_embeddings,
                torch.softmax(climate_logits, dim=-1),
                torch.softmax(month_logits, dim=-1),
                torch.softmax(state_logits, dim=-1),
                torch.softmax(county_logits, dim=-1),
                torch.softmax(city_logits, dim=-1)
            ], dim=-1)
            
            # Location coordinate predictions (latitude, longitude)
            location_preds = self.location_coord_head(combined_features)
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