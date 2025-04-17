# geolocation_clip -> Image based Geolocation
Training and eval code for image Geolocation on the OSV-Mini-129k. Image in, latitude and longitude prediction estimate out. Trained with street view imagery. This project provides modular geolocation models that can predict the location (latitude, longitude) of images. The models can optionally use auxiliary prediction heads to improve accuracy by predicting additional contextual information such as climate, month, state, county, and city.

We offer several image classification backbones for training with an option to use contextual geographic features in the classification head to further aid in location refinement.

## GeoCLIP Model Description:

- For the CLIP backbone:
  - MultiModal Vision/Text Base CLIP model
  - Contrastive Pretraining
    - Pretrained CLIP model, trained again with contrastive pretraining for the vision/text alignment. 
    - The contrastive pretraining aligns the text embeddings and image embeddings with geographic features.
    - For this step, I created a caption with with the desired contextual features: Climate, city, county, state and month.
  
**Classification Heads**
- Location Coordinate: lat, lon
  - Always present to produce the latitude and longitude, whether or not the auxilary heads and contextual features are present. 
- Auxiliary heads (only used if use_context is True)
    - Location Classification: state, county, city
    - Climate Classification: Climate (0-29 integer)
    - Month Classification: 
    - The logits from these heads are passed as input to to the location pred head to improve accuracy.

```mermaid
graph TD
        A[Image Input] --> B[CLIP Vision Model]
    subgraph Input
        B --> C[Image Embeddings]
    end

    subgraph ContextualPath[Contextual Path - use_context=True]
        C --> D1[Climate Head]
        C --> D2[Month Head]
        C --> D3[Location Classification Head]
        
        D1 --> E1[Climate Logits]
        D2 --> E2[Month Logits]
        D3 --> E3[State Logits]
        D3 --> E4[County Logits]
        D3 --> E5[City Logits]
        
        E1 --> F[Combined Features]
        E2 --> F
        E3 --> F
        E4 --> F
        E5 --> F
        C --> F
    end

    subgraph NonContextualPath[Non-Contextual Path - use_context=False]
        C --> G[Simple Coordinate Head]
    end

    subgraph Output
        F --> H1[Coordinate Prediction]
        G --> H1
        H1 --> I1[Latitude]
        H1 --> I2[Longitude]
    end

    style Input fill:#e6f3ff,stroke:#333,stroke-width:2px,rx:10

    style ContextualPath fill:#6b9fcf,stroke:#333,stroke-width:2px,rx:10
    style NonContextualPath fill:#6b9fcf,stroke:#333,stroke-width:2px,rx:10
    style Output fill:#3bc482,stroke:#333,stroke-width:2px,rx:10
    
    classDef default rx:8,ry:8
    classDef node rx:8,ry:8,stroke:#000,fill:#f5f5f5
    class A,B,C,D1,D2,D3,E1,E2,E3,E4,E5,F,G,H1,I1,I2 node
```
