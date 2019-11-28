import torch
model_file_path="/home/crow/Desktop/tesi_models/log/cnndm/model_435000_1558355323"
state = torch.load(model_file_path, map_location=lambda storage, location: storage)

print("a")