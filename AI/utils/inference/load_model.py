import os
import torch


def load_saved_model(model: torch.nn, model_saved_path: str, device: torch.device):
    # model_saved_path = os.path.join(model_saved_path, "best.pt")
    status = torch.load(model_saved_path, map_location=torch.device("cpu"))
    model.load_state_dict(status["model"])

    return model.to(device)
