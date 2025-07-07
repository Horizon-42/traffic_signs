import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

def prune_model(model, amount=0.1):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

MODEL_PATH = ""
PRUNE_MODEL_SAVE_PATH = ""
YAML_PATH = ""

model = YOLO(MODEL_PATH)

results = model.val(data=YAML_PATH)
print(f"mAP50-95: {results.box.map}")

torch_model = model.model
print(torch_model)

print("Pruning...")
pruned_torch_model = prune_model(torch_model, amount=0.1)
print("Model pruned.")

model.model = pruned_torch_model

print("Saving pruned model...")

model.save(PRUNE_MODEL_SAVE_PATH)

print("Pruned model saved.")

# Evaluate
model = YOLO(PRUNE_MODEL_SAVE_PATH)
results = model.val(data=YAML_PATH)
print(f"mAP50-95: {results.box.map}")