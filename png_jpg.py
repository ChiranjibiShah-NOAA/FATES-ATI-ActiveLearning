from PIL import Image
import os

# Folder containing .png images
folder_path = "/work/mn918/data/VOC2007/JPEGImages"

# Iterate through the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        # Open the .png image
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)

        # Convert and save as .jpg
        jpg_path = os.path.join(folder_path, os.path.splitext(filename)[0] + ".jpg")
        img.convert("RGB").save(jpg_path)

        # Remove the original .png file
        os.remove(image_path)

        print(f"{filename} converted to .jpg format and removed.")

print("Conversion complete.")
