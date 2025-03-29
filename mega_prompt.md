# Overview:

# cols: 
# id,latitude,longitude,country,captured_at,lon_bin,lat_bin,cell,land_cover,road_index,drive_side,climate,soil,dist_sea,region,sub-region,city,unique_city,unique_sub-region,unique_region,unique_country,quadtree_10_1000,creator_username,creator_id

* train only the last block of clip base 32 for contrastive image geolocation. Add addtional prediction heads for each of the following:
    1. cross entropy classifcation for 28 climiate class IDs.
    2. cross entropy month (season) 
    3. MSE regression: temperature
    4. MSE regression: percipitation
    5. MSE regression: elevation
    6. MSE regression: population density
    7. haversine loss: latitude
    8. haversine loss: longitude
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



# Prediction heads:
    All numbers but lat/lon are ints.

    climate (1 to 29)
    month (season 1 to 12) (capture at epoch time in ms)
    land_cover (1 to 11, not all represented, 9 values)
    soil (0 to 14, not all rep, 8 values)
    state (10)
    unique_county (564 values)
    unique_city (1593 values)
    latitude
    longitude
