import pandas as pd
from glob import glob

image_paths = sorted(
    glob("./data/dataset/*/*"), key=lambda x: x.split("/")[3], reverse=True
)

labels = [image_path.split("/")[3] for image_path in image_paths]
data_frame = pd.DataFrame({"image": image_paths, "label": labels})
data_frame = data_frame.sample(frac=1).reset_index(drop=True)
data_frame.to_csv("./train.csv", index=False)
