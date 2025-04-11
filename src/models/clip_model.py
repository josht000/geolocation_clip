import torch
import torch.nn as nn
from transformers import CLIPVisionModel as BaseCLIPModel
from collections import namedtuple
import math
from warnings import filterwarnings

filterwarnings('ignore', category=UserWarning, module='transformers')

# Named Tuple for model outputs
ModelOutput = namedtuple('ModelOutput', 'loss loss_climate loss_month loss_location loss_distance \
                         preds_climate preds_month preds_state preds_county preds_city preds_lat preds_lng')

# Constants
NUM_CLIMATES = 30  # 0 to 29
NUM_MONTHS = 12    # 0 to 12
NUM_STATES = 10    # 0 to 9
NUM_COUNTIES = 564 # unique counties
NUM_CITIES = 1593  # unique cities

# Loss scaling factors
CLIMATE_LOSS_SCALING = 1
MONTH_LOSS_SCALING = 1
LOCATION_LOSS_SCALING = 2 # Weighted sum of state, county, city losses
DISTANCE_LOSS_SCALING = 3 # L2 distance loss from target

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
    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32'):
        """Initialize CLIP model with auxiliary prediction heads.
        
        Args:
            model_name (str): Name of the base CLIP model to use or a path to a checkpoint dir.
        """
        super(GeoCLIP, self).__init__()
        
        # Base CLIP model
        self.base_model = BaseCLIPModel.from_pretrained(model_name)
        self.hidden_size = self.base_model.config.hidden_size
        
        # Auxiliary prediction heads
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
            climate_labels: Climate classification labels
            month_labels: Month classification labels
            state_labels: State classification labels
            county_labels: County classification labels
            city_labels: City classification labels
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
        
        # For BaseModelOutputWithPooling from a vision model, you can access embeddings directly:
        # Option 1: Use pooler_output (gives you the [CLS] token representation)
        image_embeddings = outputs.pooler_output
        
        # OR Option 2: Extract CLS token from last_hidden_state manually
        # image_embeddings = outputs.last_hidden_state[:, 0]  # Use CLS token
        
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
        
        # Ensure all input tensors are float32 for coordinates
        if lat_labels is not None:
            lat_labels = lat_labels.to(torch.float32)
        if lng_labels is not None:
            lng_labels = lng_labels.to(torch.float32)
        
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
        lat_preds, lng_preds = location_preds[:, 0], location_preds[:, 1]
        
        # Compute classification losses
        loss_climate = self.climate_loss(climate_logits, climate_labels) * CLIMATE_LOSS_SCALING
        loss_month = self.month_loss(month_logits, month_labels) * MONTH_LOSS_SCALING
        
        # Combined location classification loss (weighted sum of state, county, city losses)
        loss_state = self.location_loss(state_logits, state_labels)
        loss_county = self.location_loss(county_logits, county_labels)
        loss_city = self.location_loss(city_logits, city_labels)
        loss_location = (loss_state + loss_county + loss_city) * LOCATION_LOSS_SCALING
        
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
                # Fallback to a default loss if there's an error
                loss_distance = torch.tensor(10000.0, device=self.base_model.device) * DISTANCE_LOSS_SCALING
        else:
            # If labels are not provided, set a default loss
            print('No lat/lng labels provided, setting default distance loss')
            loss_distance = torch.tensor(0.0, device=self.base_model.device)
        
        # Total loss
        loss = loss_climate + loss_month + loss_location + loss_distance
        
        if not is_training:
            # During evaluation, we want to track the distance loss separately
            print(f"Evaluation distance loss: {loss_distance.item()}")
        
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