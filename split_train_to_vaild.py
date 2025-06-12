import shutil
import os
import random
import tqdm

train_dir = "dataset/yolo_dataset/train/"

valid_dir = "dataset/yolo_dataset/valid/"
if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)

train_images_dir = os.path.join(train_dir, "images")
train_labels_dir = os.path.join(train_dir, "labels")

valid_images_dir = os.path.join(valid_dir, "images")
valid_labels_dir = os.path.join(valid_dir, "labels")
if not os.path.exists(valid_images_dir):
    os.makedirs(valid_images_dir)
if not os.path.exists(valid_labels_dir):
    os.makedirs(valid_labels_dir)

# Move 15% of images and labels to validation set
train_images = os.listdir(train_images_dir)
# shuffle train_images
random.shuffle(train_images)
num_valid = int(len(train_images) * 0.15)
for i in tqdm.tqdm(range(num_valid)):
    image_file = train_images[i]
    label_file = image_file.replace(".png", ".txt")
    
    # Move image
    shutil.move(os.path.join(train_images_dir, image_file), os.path.join(valid_images_dir, image_file))
    
    # Move label
    shutil.move(os.path.join(train_labels_dir, label_file), os.path.join(valid_labels_dir, label_file))