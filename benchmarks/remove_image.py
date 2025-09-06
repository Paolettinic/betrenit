import os
import pandas as pd
from PIL import Image
import io

# Paths
input_parquet = "llavabench_coco.parquet"
output_parquet = "output.parquet"
image_dir = "images"

# Make sure output folder exists
os.makedirs(image_dir, exist_ok=True)

# Load parquet
df = pd.read_parquet(input_parquet)
# Save images
for idx, row in df.iterrows():
    image_id = row["image_id"]
    image_dict = row["image"]  # this is a dict with 'bytes' and 'path'
    image_bytes = image_dict["bytes"]  # extract actual bytes

    # Convert bytes → PIL Image
    img = Image.open(io.BytesIO(image_bytes))

    # Save image
    img.save(os.path.join(image_dir, image_id))

# Drop image column
df_no_image = df.drop(columns=["image"])

# Save parquet without images
df_no_image.to_parquet(output_parquet, index=False)

print(f"✅ Saved {len(df)} images into {image_dir}")
print(f"✅ New parquet file: {output_parquet}")

