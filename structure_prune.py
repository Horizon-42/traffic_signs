import torch
from ultralytics import YOLO
import torch_pruning as tp
import os
from thop import profile # Used for calculating FLOPs and parameter count

def optimize_yolo_structured_pruning(
    model_path: str,
    pruning_ratio: float,
    save_path: str,
    example_input_size: tuple = (1, 3, 640, 640),
    ignored_layers: list = None # Can be module objects or module name strings
):
    """
    Optimized YOLO model structured pruning function.

    Performs structured pruning on a YOLO model, providing detailed logs,
    pruning effect evaluation, and intelligent handling of ignored layers.

    Args:
        model_path (str): Path to the pre-trained YOLO model (e.g., 'yolov8n.pt').
        pruning_ratio (float): The pruning ratio, between 0.0 and 1.0
                                (e.g., 0.5 means 50% pruning).
        save_path (str): Path to save the pruned model
                         (e.g., 'yolov8n_pruned_optimized.pt').
        example_input_size (tuple): Example input size for the model (N, C, H, W),
                                    used for graph analysis. Defaults to (1, 3, 640, 640)
                                    common for YOLOv8/v11.
        ignored_layers (list, optional): List of layers (modules) to exclude from pruning.
                                    Can be module objects themselves (e.g., model.model.fc)
                                    or module name strings (e.g., 'model.22', 'model.23.m').
                                    Automatically identifies common YOLO detection heads by default.
    Returns:
        YOLO: The pruned YOLO model object, or None if pruning fails.
    """
    if not (0.0 <= pruning_ratio <= 1.0):
        raise ValueError("pruning_ratio must be between 0.0 and 1.0.")

    print(f"--- Starting optimized YOLO model structured pruning ---")
    print(f"Pruning Ratio: {pruning_ratio*100:.2f}%")
    print(f"Example Input Size: {example_input_size}")

    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model will be pruned on {device}.")

    # 1. Load YOLO Model
    try:
        yolo_model_obj = YOLO(model_path)
        # Get the underlying nn.Module
        base_model = yolo_model_obj.model.to(device)
        print(f"Model '{model_path}' loaded successfully!")
        
        # Record parameters and FLOPs before pruning
        dummy_input = torch.randn(example_input_size).to(device)
        macs_before, params_before = profile(base_model, inputs=(dummy_input,), verbose=False)
        print(f"Model Parameters Before Pruning: {params_before / 1e6:.2f}M, FLOPs: {macs_before / 1e9:.2f}G")

    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Ensure the path is correct or Ultralytics can auto-download.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # 2. Define Pruning Strategy (L1 Magnitude Importance)
    importance = tp.importance.MagnitudeImportance(p=1)
    print("Using L1 Magnitude as pruning importance metric.")

    # 3. Smartly handle and identify layers to ignore
    actual_ignored_layers_list = []
    if ignored_layers:
        for item in ignored_layers:
            if isinstance(item, str): # If it's a string name
                found = False
                for name, module in base_model.named_modules():
                    if name == item:
                        actual_ignored_layers_list.append(module)
                        print(f"  Manually ignoring layer: '{name}' (module found).")
                        found = True
                        break
                if not found:
                    print(f"  Warning: Module named '{item}' not found, skipping this ignored layer setting.")
            elif isinstance(item, torch.nn.Module): # If it's directly a module object
                actual_ignored_layers_list.append(item)
                print(f"  Manually ignoring layer: '{item.__class__.__name__}' (module object).")
            else:
                print(f"  Warning: Unrecognized ignored layer type: {type(item)}. Skipping.")

    # Auto-identify YOLO's last few Detection Head layers
    yolo_head_modules = []
    if hasattr(base_model, 'model') and isinstance(base_model.model, torch.nn.Sequential):
        # Common heuristic: YOLO heads are often the last module in the sequential model.
        # This might need adjustment based on specific YOLO versions (v8, v9, v11) and tasks (Detect, Segment, Pose).
        if len(base_model.model) > 0:
            last_module = base_model.model[-1]
            if 'detect' in str(last_module.__class__.__name__).lower() or \
               'segment' in str(last_module.__class__.__name__).lower() or \
               'pose' in str(last_module.__class__.__name__).lower():
                yolo_head_modules.append(last_module)
                print(f"  Auto-identified and ignoring YOLO head: '{last_module.__class__.__name__}'.")
            
            # For some YOLO models, the head might be one of the last few.
            # Example: Sometimes the head is preceded by a "C2f" or "SPPF" that's critical.
            # It's always best to `print(yolo_model_obj.model)` to inspect.
            # You might need to add more specific logic here if the auto-detection is insufficient.
            # For instance, if 'model.22' is the head in YOLOv8n and it's not the absolute last:
            # if len(base_model.model) > 22 and isinstance(base_model.model[22], (Detect, Segment, Pose)):
            #     yolo_head_modules.append(base_model.model[22])
            #     print(f"  Auto-identified and ignoring YOLO head: 'model.22'.")


    for m in yolo_head_modules:
        if m not in actual_ignored_layers_list:
            actual_ignored_layers_list.append(m)

    actual_ignored_layers_list = []
    if not actual_ignored_layers_list:
        print("No layers specified or auto-identified to ignore. All layers might be pruned, proceed with caution.")
    else:
        # Get names of ignored modules for logging
        ignored_names = [name for name, module in base_model.named_modules() if module in actual_ignored_layers_list]
        print(f"Final ignored layers ({len(actual_ignored_layers_list)}): {ignored_names}")


    # 4. Build Dependency Graph
    print("Building model dependency graph...")
    try:
        DG = tp.DependencyGraph().build_dependency(base_model, example_inputs=dummy_input)
        DG.verbose
        print("Dependency graph built successfully.")
    except Exception as e:
        print(f"Error: Failed to build dependency graph: {e}")
        print("Please check if example_input_size matches model input.")
        return None

    # 5. Initialize and Execute Pruning
    print(f"Initializing pruner and executing pruning step...")
    try:
        pruner = tp.pruner.MagnitudePruner(
            base_model,
            dummy_input,
            importance,
            DG, # Pass the dependency graph
            pruning_ratio=float(pruning_ratio),
            ignored_layers=actual_ignored_layers_list,
        )
        # CORRECTED: Call pruner.step() instead of pruner.prune()
        pruner.step() 
        print("Structured pruning operation completed!")
    except Exception as e:
        print(f"Error: Failed to execute pruning operation: {e}")
        return None
    
    # 6. Record parameters and FLOPs after pruning
    macs_after, params_after = profile(base_model, inputs=(dummy_input,), verbose=False)
    print(f"Model Parameters After Pruning: {params_after / 1e6:.2f}M, FLOPs: {macs_after / 1e9:.2f}G")
    print(f"Parameter Reduction: {((params_before - params_after) / params_before * 100):.2f}%")
    print(f"FLOPs Reduction: {((macs_before - macs_after) / macs_before * 100):.2f}%")

    # 7. Save the pruned model
    try:
        # Re-assign the pruned nn.Module back to the YOLO object's .model attribute
        yolo_model_obj.model = base_model.cpu() # Move to CPU for broader save compatibility
        yolo_model_obj.save(save_path)
        print(f"Pruned model successfully saved to: {save_path}")
    except Exception as e:
        print(f"Error saving pruned model: {e}")
        if os.path.exists(save_path):
            os.remove(save_path) # Clean up partial files if save fails
        return None

    print("--- Pruning process completed ---")
    return yolo_model_obj # Return the pruned YOLO model object

# --- Usage Example ---
if __name__ == "__main__":
    # --- Configuration Parameters ---
    # Replace with your YOLO model file path (e.g., yolov8n.pt, yolov11m.pt)
    yolo_model_file = 'models/best.pt'
    output_pruned_model_file = 'yolov11m_pruned_optimized.pt'
    pruning_ratio_val = 0.9 # Prune 35%
    input_shape = (1, 3, 64, 64)  # YOLO model input size (N, C, H, W)

    # Layers to ignore: can be a list of module name strings or module objects.
    # For YOLO models, the detection head is typically not pruned.
    # You can print your model structure (e.g., `print(YOLO('model.pt').model)`)
    # to find the exact names of these layers.
    # For instance, YOLOv8n's Detection Head is often 'model.22' or 'model.23' depending on the task.
    # ignored_layers_list = ['model.22'] # Example, adjust based on your model's structure
    ignored_layers_list = [] # Leave empty for the function to attempt auto-identification

    # --- Execute Pruning ---
    pruned_model_obj = optimize_yolo_structured_pruning(
        model_path=yolo_model_file,
        pruning_ratio=pruning_ratio_val,
        save_path=output_pruned_model_file,
        example_input_size=input_shape,
        ignored_layers=ignored_layers_list # Pass the list of layers to ignore
    )

    if pruned_model_obj:
        # --- Verify the Pruned Model ---
        print(f"\nAttempting to load the pruned model: {output_pruned_model_file}")
        try:
            loaded_model_for_test = YOLO(output_pruned_model_file)
            print("Pruned model loaded successfully!")

            # You can now use the loaded_model_for_test for inference or fine-tuning
            # print("\nPerforming a simple inference test...")
            # # Ensure you have a test image path
            # # results = loaded_model_for_test('path/to/your_test_image.jpg')
            # # for r in results:
            # #     r.show() # Display detection results
            
            print("\n--- IMPORTANT: Fine-tuning the pruned model is highly recommended to recover performance! ---")
            print("Example fine-tuning command:")
            print(f"loaded_model_for_test.train(data='your_dataset.yaml', epochs=50, imgsz={input_shape[2]}, batch=16)")

        except Exception as e:
            print(f"Error loading the pruned model: {e}")
            print("Please check the save path and file integrity.")
    else:
        print("\nPruning process did not complete successfully.")