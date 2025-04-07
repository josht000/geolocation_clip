# geolocation_clip
Training and eval code for image geolocation on OSV-Mini-129k

still issue with clip_model.py:L209 = loss_distance = torch.tensor(0.0, device=lat_preds.device)
Might be async error. 