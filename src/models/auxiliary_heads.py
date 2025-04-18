import torch
import torch.nn as nn
from collections import namedtuple

from src.constants import *

# Named Tuple for model outputs
AuxiliaryOutput = namedtuple('AuxiliaryOutput', 'climate_logits month_logits state_logits county_logits city_logits combined_features')

class AuxiliaryGeo(nn.Module):
    """Auxiliary prediction heads for geolocation models.
    
    This class contains prediction heads for:
    - Climate classification
    - Month classification
    - Hierarchical location classification (state, county, city)
    
    These heads can be used with any base model that provides embeddings.
    """
    
    def __init__(self, hidden_size: int):
        super(AuxiliaryGeo, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Climate classification head
        self.climate_head = nn.Linear(hidden_size, NUM_CLIMATES)
        
        # Month classification head
        self.month_head = nn.Linear(hidden_size, NUM_MONTHS)
        
        # Hierarchical location classification head
        self.location_class_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_STATES + NUM_COUNTIES + NUM_CITIES)  # Combined output for all location levels
        )
        
    def forward(self, embeddings):
        """Forward pass through the auxiliary prediction heads.
        
        Args:
            embeddings: Input embeddings from the base model
            
        Returns:
            AuxiliaryOutput: Named tuple containing all auxiliary predictions and combined features
        """
        # Get all logits
        climate_logits = self.climate_head(embeddings).to(torch.float32)
        month_logits = self.month_head(embeddings).to(torch.float32)
        location_logits = self.location_class_head(embeddings)
        state_logits = location_logits[:, :NUM_STATES]
        county_logits = location_logits[:, NUM_STATES:NUM_STATES + NUM_COUNTIES]
        city_logits = location_logits[:, NUM_STATES + NUM_COUNTIES:]
        
        # Create combined features for coordinate prediction
        combined_features = torch.cat([
            embeddings,
            torch.softmax(climate_logits, dim=-1),
            torch.softmax(month_logits, dim=-1),
            torch.softmax(state_logits, dim=-1),
            torch.softmax(county_logits, dim=-1),
            torch.softmax(city_logits, dim=-1)
        ], dim=-1)
        
        return AuxiliaryOutput(
            climate_logits=climate_logits,
            month_logits=month_logits,
            state_logits=state_logits,
            county_logits=county_logits,
            city_logits=city_logits,
            combined_features=combined_features
        )

class LocationCoordinateHead(nn.Module):
    def __init__(self, input_size, use_auxiliary=False):
        super(LocationCoordinateHead, self).__init__()
        self.use_auxiliary = use_auxiliary

        self.mlp = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output: [latitude, longitude]
        )

    def forward(self, x):
        return self.mlp(x)
