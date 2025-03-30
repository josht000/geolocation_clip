# Overview:

# cols: 
# id,latitude,longitude,country,captured_at,lon_bin,lat_bin,cell,land_cover,road_index,drive_side,climate,soil,dist_sea,region,sub-region,city,unique_city,unique_sub-region,unique_region,unique_country,quadtree_10_1000,creator_username,creator_id

* train only the last block of clip base 32 for contrastive image geolocation. Add addtional prediction heads for each of the following:
    1. cross entropy classification for 28 climate class IDs.
    2. cross entropy month (season) 
    3. haversine loss: latitude
    4. haversine loss: longitude

* text for contrastive pretraining of each shall include the following:
    - region: this is the county the image was taken in
    - city: city the image was taken in
    - climate of the region in the image
    - 

CLIP Pretraining.
    Location: A photo in the city, county, state of: CITY, COUNTY, STATE.
    Climate: This location has a temperate oceanic climate.
    Month: This photo was taken in December.
    Geographic: This location has a (land_cover) land type and (soil) soil type


Modify CLIPModel class with the follow to make it similar to SuperGuessr class.

# Prediction heads:
    Auxilary prediction heads are: climate, month, land_cover, state, county, city. These should all have their final predictions be whole integer values. Latitude and longitude should be predicted AFTER the auxilary heads ie. the auxilary predictions are input to the final latitude and longitude values. Climate, month, state, county, and city will get MSE loss functions as they are treated as classification tasks, while latitude and longitude will get cross entropy for the final regression task.

    1. Climate (0 to 29)       - Cross entropy loss 
    2. month (int: 1 to 12)    - Cross entropy loss
    3. state (int: 0 to 9)     - Cross entropy loss
    4. unique_county (564 unique values) - Cross entropy loss
    5. unique_city (1593 unique values)  - Cross entropy loss
    6. latitude (float): L2 distance loss from target
    7. longitude (float): L2 distance loss from target


## 
 Take the three functions in train_modes.py and write similar but not idential, simple versions that get the job done. Take liberty to change function and variable names, codiing style, and logic but accomplish training, eval, and profiling in a similar way. These functions will be added to src/train_modes.py.