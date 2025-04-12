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
ModelOutput = namedtuple('ModelOutput', 'loss loss_climate loss_month loss_location loss_distance \
                         preds_climate preds_month preds_state preds_county preds_city preds_lat preds_lng')


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points on Earth.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees
        
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def l2_distance(lat1, lon1, lat2, lon2):
    """Calculate the L2 (Euclidean) distance between two points.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees
        
    Returns:
        L2 distance in degrees
    """
    return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

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
            
            # Loss functions
            self.climate_loss = nn.CrossEntropyLoss()
            self.month_loss = nn.CrossEntropyLoss()
            self.location_loss = nn.CrossEntropyLoss()
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
            
    def forward(self, pixel_values=None, input_ids=None, attention_mask=None,
                climate_labels=None, month_labels=None, state_labels=None,
                county_labels=None, city_labels=None, lat_labels=None, lng_labels=None,
                is_training=True):
        """Forward pass through the model.
        
        Args:
            pixel_values: Input images
            input_ids: Text input IDs
            attention_mask: Text attention mask
            climate_labels: Climate classification labels (only used if use_context is True)
            month_labels: Month classification labels (only used if use_context is True)
            state_labels: State classification labels (only used if use_context is True)
            county_labels: County classification labels (only used if use_context is True)
            city_labels: City classification labels (only used if use_context is True)
            lat_labels: Latitude regression labels
            lng_labels: Longitude regression labels
            is_training: Boolean indicating whether the model is in training mode
            
        Returns:
            ModelOutput: Named tuple containing all losses and predictions
        """
        # Get base model embeddings
        outputs = self.base_model(
            pixel_values=pixel_values,
        )
        
        image_embeddings = outputs.pooler_output
        
        if self.use_context:
            # Convert input labels to float32 if they aren't already
            if climate_labels is not None:
                climate_labels = climate_labels.to(torch.long)
            if month_labels is not None:
                month_labels = month_labels.to(torch.long)
            if state_labels is not None:
                state_labels = state_labels.to(torch.long)
            if county_labels is not None:
                county_labels = county_labels.to(torch.long)
            if city_labels is not None:
                city_labels = city_labels.to(torch.long)
            
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
            
            # Compute classification losses
            loss_climate = self.climate_loss(climate_logits, climate_labels) * CLIMATE_LOSS_SCALING
            loss_month = self.month_loss(month_logits, month_labels) * MONTH_LOSS_SCALING
            
            # Combined location classification loss
            loss_state = self.location_loss(state_logits, state_labels)
            loss_county = self.location_loss(county_logits, county_labels)
            loss_city = self.location_loss(city_logits, city_labels)
            loss_location = (loss_state + loss_county + loss_city) * LOCATION_LOSS_SCALING
        else:
            # Simple coordinate prediction without context
            location_preds = self.location_coord_head(image_embeddings)
            loss_climate = torch.tensor(0.0, device=self.base_model.device)
            loss_month = torch.tensor(0.0, device=self.base_model.device)
            loss_location = torch.tensor(0.0, device=self.base_model.device)
            climate_logits = None
            month_logits = None
            state_logits = None
            county_logits = None
            city_logits = None
        
        lat_preds, lng_preds = location_preds[:, 0], location_preds[:, 1]
        
        # Calculate distance loss
        if lat_labels is not None and lng_labels is not None:
            try:
                # Stack coordinates into [batch_size, 2] tensors
                pred_coords = torch.stack([lat_preds, lng_preds], dim=1)
                true_coords = torch.stack([lat_labels, lng_labels], dim=1)
                distances = l2_distance_tensor(pred_coords, true_coords) # vectorized L2 distance
                distances = torch.nan_to_num(distances, nan=10000.0, posinf=10000.0, neginf=10000.0)
                loss_distance = distances.mean() * DISTANCE_LOSS_SCALING
            except Exception as e:
                print(f"Error calculating distance: {e}")
                loss_distance = torch.tensor(10000.0, device=self.base_model.device) * DISTANCE_LOSS_SCALING
        else:
            # If labels are not provided, set a default loss
            print('No lat/lng labels provided, setting default distance loss')
            loss_distance = torch.tensor(0.0, device=self.base_model.device)
        
        # Total loss
        loss = loss_climate + loss_month + loss_location + loss_distance
        
        return ModelOutput(
            loss=loss,
            loss_climate=loss_climate,
            loss_month=loss_month,
            loss_location=loss_location,
            loss_distance=loss_distance,
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
            "num_cities": NUM_CITIES,
            "climate_loss_scaling": CLIMATE_LOSS_SCALING,
            "month_loss_scaling": MONTH_LOSS_SCALING,
            "location_loss_scaling": LOCATION_LOSS_SCALING,
            "distance_loss_scaling": DISTANCE_LOSS_SCALING
        }
        
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
            
        # Save the base model configuration
        self.base_model.config.save_pretrained(save_directory)
        
        print(f"Model saved to {save_directory}")
        
    @classmethod
    def from_pretrained(cls, model_path: str):
        """Load a pretrained GeoCLIP model from a directory.
        
        Args:
            model_path (str): Path to the pretrained model directory.
            
        Returns:
            GeoCLIP: Loaded model.
        """
        import os
        import json
        
        # Load configuration
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = json.load(f)
            
        # Create model instance with use_context parameter
        model = cls(model_name=config["model_name"], use_context=config.get("use_context", True))
        
        # Load state dict
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        model.load_state_dict(state_dict)
        
        return model