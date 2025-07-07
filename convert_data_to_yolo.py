import pandas as pd
import os
import shutil
import tqdm

def convert_to_yolo(csv_file_path:str, output_labels_dir:str, output_imgs_dir:str):
    """
    Converts traffic sign label data from a CSV file to YOLO format.

    Args:
        csv_file_path (str): The path to the input CSV file.
        output_dir (str): The directory where the YOLO format .txt label files
                          will be saved. Defaults to "yolo_labels".
    """
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)
        print(f"Created output directory: {output_labels_dir}")
    if not os.path.exists(output_imgs_dir):
        os.makedirs(output_imgs_dir)
        print(f"Created output directory: {output_imgs_dir}")

    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded CSV file: {csv_file_path}")
        print(f"Total entries to process: {len(df)}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Group data by image path to process all bounding boxes for one image together
    # This is crucial because each image should have one .txt file containing all its labels
    grouped = df.groupby('Path')

    src_dir = os.path.dirname(csv_file_path)

    processed_count = 0
    for image_path, image_data in tqdm.tqdm(grouped):
        # Extract the base filename (e.g., "00000.jpg" from "path/to/00000.jpg")
        image_filename = os.path.basename(image_path)
        dst_image_path = os.path.join(output_imgs_dir, image_filename)
        shutil.copy(os.path.join(src_dir, image_path), dst_image_path)
        # Construct the corresponding YOLO label file name (e.g., "00000.txt")
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_file_path = os.path.join(output_labels_dir, label_filename)

        with open(label_file_path, 'w') as f:
            for _, row in image_data.iterrows():
                width = row['Width']
                height = row['Height']
                class_id = row['ClassId']

                # Bounding box coordinates from CSV (Roi.X1, Roi.Y1, Roi.X2, Roi.Y2)
                # These are pixel coordinates for top-left (X1, Y1) and bottom-right (X2, Y2)
                x1 = row['Roi.X1']
                y1 = row['Roi.Y1']
                x2 = row['Roi.X2']
                y2 = row['Roi.Y2']

                # Calculate center coordinates and width/height of the bounding box
                # in absolute pixel values
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2
                box_width = x2 - x1
                box_height = y2 - y1

                # Normalize the coordinates and dimensions to be between 0 and 1
                # This is the core requirement for YOLO format
                normalized_center_x = box_center_x / width
                normalized_center_y = box_center_y / height
                normalized_box_width = box_width / width
                normalized_box_height = box_height / height

                # Write the data to the label file in YOLO format:
                # class_id center_x center_y width height (all normalized)
                # Use f-string for precise formatting (5 decimal places for coordinates)
                f.write(
                    f"{class_id} {normalized_center_x:.6f} {normalized_center_y:.6f} "
                    f"{normalized_box_width:.6f} {normalized_box_height:.6f}\n"
                )
        processed_count += 1
        if processed_count % 100 == 0: # Print progress every 100 images
            print(f"Processed {processed_count} images so far...")

    print(f"\nConversion complete! {processed_count} YOLO label files created in '{output_labels_dir}'.")

if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Replace 'your_labels.csv' with the actual path to your CSV file
    input_csv = './dataset/Traffic/Data/Test.csv'
    # Optional: Change the output directory name if you prefer
    output_labels_dir = './dataset/yolo_data/test/labels'
    output_imgs_dir = './dataset/yolo_data/test/images'
    # -------------------

    # Run the conversion function
    convert_to_yolo(input_csv, output_labels_dir, output_imgs_dir)

    print("\n--- Next Steps ---")
    print(f"1. Ensure your original images are in a directory accessible to your YOLO training.")
    print(f"2. Place the '{output_labels_dir}' folder containing the .txt labels alongside your image folder or in a structured way as required by your YOLO training setup (e.g., in a 'labels' directory parallel to an 'images' directory).")
    print(f"3. Verify that your `ClassId` values are correct and correspond to your class names list for YOLO.")
