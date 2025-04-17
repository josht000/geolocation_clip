import torch
import torch.nn as nn
import math
from src.constants import *

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance (in deg) between two points on Earth.
    Returns: Distance in kilometers
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
    """Calculate the L2 (Euclidean) distance (degrees) between two points.
    Returns: L2 distance in degrees
    """
    return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

def l2_distance_tensor(pred_coords, true_coords):
    """Calculate L2 distance between predicted and true coordinates using tensors.
    Returns: Tensor of shape [batch_size] containing L2 distances
    """
    return torch.sqrt(torch.sum((pred_coords - true_coords) ** 2, dim=1))

class GeoCLIPLoss:
    def __init__(self, device):
        """Initialize loss functions for GeoCLIP model.
        """
        self.device = device
        self.climate_loss = nn.CrossEntropyLoss()
        self.month_loss = nn.CrossEntropyLoss()
        self.location_loss = nn.CrossEntropyLoss()
        
    def __call__(self, model_output, climate_labels=None, month_labels=None, 
                 state_labels=None, county_labels=None, city_labels=None, 
                 lat_labels=None, lng_labels=None):
        """Calculate all losses for the GeoCLIP model.
        Returns: tuple: (total_loss, dict of individual losses)
        """
        # Initialize losses
        loss_climate = torch.tensor(0.0, device=self.device)
        loss_month = torch.tensor(0.0, device=self.device)
        loss_location = torch.tensor(0.0, device=self.device)
        loss_distance = torch.tensor(0.0, device=self.device)
        
        # Convert labels to appropriate type if provided
        if climate_labels is not None:
            climate_labels = climate_labels.to(torch.long)
            loss_climate = self.climate_loss(model_output.preds_climate, climate_labels) * CLIMATE_LOSS_SCALING
            
        if month_labels is not None:
            month_labels = month_labels.to(torch.long)
            loss_month = self.month_loss(model_output.preds_month, month_labels) * MONTH_LOSS_SCALING
            
        if state_labels is not None and county_labels is not None and city_labels is not None:
            state_labels = state_labels.to(torch.long)
            county_labels = county_labels.to(torch.long)
            city_labels = city_labels.to(torch.long)
            
            loss_state = self.location_loss(model_output.preds_state, state_labels)
            loss_county = self.location_loss(model_output.preds_county, county_labels)
            loss_city = self.location_loss(model_output.preds_city, city_labels)
            loss_location = (loss_state + loss_county + loss_city) * LOCATION_LOSS_SCALING
            
        if lat_labels is not None and lng_labels is not None:
            try:
                # Stack coordinates into [batch_size, 2] tensors
                pred_coords = torch.stack([model_output.preds_lat, model_output.preds_lng], dim=1)
                true_coords = torch.stack([lat_labels, lng_labels], dim=1)
                distances = l2_distance_tensor(pred_coords, true_coords)
                distances = torch.nan_to_num(distances, nan=10000.0, posinf=10000.0, neginf=10000.0)
                loss_distance = distances.mean() * DISTANCE_LOSS_SCALING
            except Exception as e:
                print(f"Error calculating distance: {e}")
                loss_distance = torch.tensor(10000.0, device=self.device) * DISTANCE_LOSS_SCALING
                
        # Calculate total loss
        total_loss = loss_climate + loss_month + loss_location + loss_distance
        
        # Return total loss and individual losses for monitoring
        losses = {
            'total': total_loss,
            'climate': loss_climate,
            'month': loss_month,
            'location': loss_location,
            'distance': loss_distance
        }
        
        return total_loss, losses

def calculate_batch_metrics(model_output, inputs, use_context=True):
    """Calculate metrics for a single batch of data.
    
    This function handles the calculation of location accuracy and distance error
    for a single batch, without the data loading and loop structure.
    
    Args:
        model_output: Output from the model
        inputs: Dictionary of input tensors
        use_context: Whether to use contextual features
        
    Returns:
        Dictionary containing batch metrics
    """
    metrics = {}
    
    # Calculate location classification accuracy
    if use_context and 'state_labels' in inputs:
        state_preds = torch.argmax(model_output.preds_state, dim=1)
        county_preds = torch.argmax(model_output.preds_county, dim=1)
        city_preds = torch.argmax(model_output.preds_city, dim=1)
        
        correct_state = (state_preds == inputs['state_labels']).sum().item()
        correct_county = (county_preds == inputs['county_labels']).sum().item()
        correct_city = (city_preds == inputs['city_labels']).sum().item()
        total_samples = inputs['state_labels'].size(0)
        
        metrics.update({
            "correct_state": correct_state,
            "correct_county": correct_county,
            "correct_city": correct_city,
            "total_samples": total_samples
        })
    
    # Calculate coordinate prediction error
    if 'lat_labels' in inputs and 'lng_labels' in inputs:
        pred_coords = torch.stack([model_output.preds_lat, model_output.preds_lng], dim=1)
        true_coords = torch.stack([inputs['lat_labels'], inputs['lng_labels']], dim=1)
        distances = l2_distance_tensor(pred_coords, true_coords)
        total_distance_error = distances.sum().item()
        
        metrics.update({
            "total_distance_error": total_distance_error,
            "batch_size": inputs['lat_labels'].size(0)
        })
    
    return metrics 