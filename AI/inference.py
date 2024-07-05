import torch
import ttach as tta

from models.model import SkinlesionModel
from utils.inference.load_model import load_saved_model
from utils.inference.single_image_aug import augment_single_image
from models.unet_model import SegmentationModel, process_image_mask


def predict(image_path, args, model_saved_path, device, voting=True):
    model = SkinlesionModel(**args.__dict__)
    model = load_saved_model(model, model_saved_path, device)

    if args.tta:
        tta_transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
            ]
        )
        model = tta.ClassificationTTAWrapper(model, tta_transforms)

    model = model.to(device)
    # ==========================================================================
    model.eval()
     # Segment the lesion part using UNet if args.use_unet is True
    if args.predict_mask:
        unet = SegmentationModel("Unet", "resnet34", 3, 1)
        unet.load_state_dict(torch.load(args.unet_checkpoint)["state_dict"])
        unet = unet.to(device)
        unet.eval()
        image_tensor = process_image_mask(image_path, unet)
    else:
        image_tensor = augment_single_image(img_path=image_path, img_size=args.img_size)

# Predict using SkinlesionModel
    with torch.no_grad():
        img = image_tensor.unsqueeze(0)  # add batch size of 1
        img = img.to(device)
        output = model(img)
        if voting:
            prediction = output.data.cpu().numpy()
        else:
            prediction = torch.argmax(output, dim=-1).data.cpu().numpy()

    # ==========================================================================
    return prediction[0]
