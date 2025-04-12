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