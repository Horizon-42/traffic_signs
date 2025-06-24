import os
import json # Import the json module
from ultralytics import YOLO

def test_yolo_model(model_path, data_yaml_path, img_size=640, batch_size=16, conf_threshold=0.25, iou_threshold=0.7, device=None, save_txt=False, save_json_predictions=False, plots=True, verbose=True, output_metrics_json_path="val_metrics.json"):
    """
    Tests a YOLO model using Ultralytics' val mode and saves summarized metrics to a JSON file.

    Args:
        model_path (str): Path to the trained YOLO model weights (e.g., 'yolov8n.pt' or 'path/to/best.pt').
        data_yaml_path (str): Path to the dataset configuration file (e.g., 'coco128.yaml' or 'path/to/your_dataset.yaml').
        img_size (int, optional): Image size for validation. Defaults to 640.
        batch_size (int, optional): Batch size for validation. Defaults to 16.
        conf_threshold (float, optional): Object confidence threshold for detection. Defaults to 0.25.
        iou_threshold (float, optional): Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Defaults to 0.7.
        device (str, optional): Device to run on (e.g., 'cpu', '0', '0,1'). If None, Ultralytics will auto-select.
        save_txt (bool, optional): Save individual prediction results to .txt files. Defaults to False.
        save_json_predictions (bool, optional): Save COCO-style prediction results to a 'predictions.json' file. Defaults to False.
        plots (bool, optional): Save plots (e.g., confusion matrix, F1 curve). Defaults to True.
        verbose (bool, optional): Display detailed information during validation. Defaults to True.
        output_metrics_json_path (str, optional): Path to save the summarized metrics JSON file. Defaults to "val_metrics.json".
    """
    try:
        # Load the YOLO model
        model = YOLO(model_path)
        print(f"Successfully loaded model from: {model_path}")

        # Validate the model
        print(f"\nStarting validation of the model on dataset: {data_yaml_path}...")
        metrics = model.val(
            data=data_yaml_path,
            imgsz=img_size,
            batch=batch_size,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            save_txt=save_txt,
            save_json=save_json_predictions, # This saves predictions, not general metrics
            plots=plots,
            verbose=verbose
        )

        # Extract key metrics
        val_metrics = {
            "mAP50-95": metrics.box.map,
            "mAP50": metrics.box.map50,
            "mAP75": metrics.box.map75,
            # Add other metrics you're interested in
            "precision": metrics.box.mp, # Mean Precision
            "recall": metrics.box.mr,    # Mean Recall
            # "fitness": metrics.box.fitness # Fitness score
        }

        # Include per-class mAP if available
        if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
            class_names = getattr(model.model, 'names', {})
            per_class_map = {}
            for i, map_val in enumerate(metrics.box.maps):
                class_name = class_names.get(i, f"class_{i}")
                per_class_map[class_name] = float(map_val) # Convert to float for JSON serialization
            val_metrics["per_class_mAP50-95"] = per_class_map

        # Print key metrics to console
        print("\n--- Validation Results ---")
        for key, value in val_metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            elif isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v:.4f}")
            else:
                print(f"{key}: {value}")
        
        # Save summarized metrics to a JSON file
        try:
            with open(output_metrics_json_path, 'w') as f:
                json.dump(val_metrics, f, indent=4)
            print(f"\nSummarized validation metrics saved to: {output_metrics_json_path}")
        except Exception as json_err:
            print(f"Error saving summarized metrics to JSON: {json_err}")

        # Information about where the Ultralytics-generated files are saved (plots, predictions.json if enabled)
        print(f"All Ultralytics validation outputs (plots, etc.) are saved to: {metrics.save_dir}")
        if save_json_predictions:
            predictions_json_path = os.path.join(metrics.save_dir, 'predictions.json')
            print(f"COCO-style prediction JSON is available in: {predictions_json_path}")

        print("\nValidation complete!")

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path} or data YAML file not found at {data_yaml_path}.")
    except Exception as e:
        print(f"An unexpected error occurred during model testing: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    YOLO_MODEL_PATH = 'runs/detect/train2/weights/best.pt' 
    DATASET_YAML_PATH = 'train.yaml' 

    if not os.path.exists(DATASET_YAML_PATH):
        print(f"Warning: Dataset YAML file not found at '{DATASET_YAML_PATH}'. "
              f"If using a built-in Ultralytics dataset, it might be downloaded automatically. "
              f"If it's a custom dataset, please ensure the path is correct and the file exists.")

    # Optional: Adjust validation parameters
    IMAGE_SIZE = 64
    BATCH_SIZE = 32
    CONF_THRESHOLD = 0.001 
    IOU_THRESHOLD = 0.7    
    DEVICE = 'cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu' 
    SAVE_TXT_RESULTS = False
    SAVE_JSON_PREDICTIONS = True # Set to True if you also want the COCO-style prediction JSON
    SAVE_PLOTS = True 
    VERBOSE_OUTPUT = True
    
    # New parameter for the summarized metrics JSON file
    OUTPUT_METRICS_JSON_FILE = "yolo_val_metrics_summary.json"

    # --- Run the test ---
    test_yolo_model(
        model_path=YOLO_MODEL_PATH,
        data_yaml_path=DATASET_YAML_PATH,
        img_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        device=DEVICE,
        save_txt=SAVE_TXT_RESULTS,
        save_json_predictions=SAVE_JSON_PREDICTIONS, # Pass the parameter for prediction JSON
        plots=SAVE_PLOTS,
        verbose=VERBOSE_OUTPUT,
        output_metrics_json_path=OUTPUT_METRICS_JSON_FILE # Pass the path for summary metrics JSON
    )