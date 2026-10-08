import json
import numpy as np
import torch

# Sample data with the given structure
data = {
    'img_shape': (640, 640, 3),
    'ori_shape': (264, 500),
    'scale_factor': (1.28, 1.2765151515151516),
    'batch_input_shape': (640, 640),
    'pad_shape': (640, 640),
    'img_path': None,
    'pad_param': np.array([151., 152., 0., 0.], dtype=float).tolist(),
    'img_id': 0,
    'ignored_instances': {
        'labels': [],
        'bboxes': [],
    },
    'gt_instances': {
        'labels': [],
        'bboxes': [],
    },
    'pred_instances': {
        'scores': [0.9648, 0.0020, 0.0014],
        'labels': [0, 0, 0],
        'bboxes': [
            [255.3041, 64.5711, 318.7285, 104.8894],
            [15.6494, 234.8727, 91.5717, 264.0000],
            [0.9689, 232.6046, 72.0075, 264.0000],
        ],
    },
}

# Convert the dictionary to JSON-serializable format
def convert_to_serializable(item):
    if isinstance(item, (np.ndarray, np.generic)):
        return item.tolist()
    elif isinstance(item, torch.Tensor):
        return item.cpu().tolist()
    elif isinstance(item, dict):
        return {key: convert_to_serializable(value) for key, value in item.items()}
    elif isinstance(item, (list, tuple)):
        return [convert_to_serializable(element) for element in item]
    else:
        return item

def write_json(json_path, data):
    # Save the serializable data to a JSON file
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Convert the dictionary to a serializable format
serializable_data = convert_to_serializable(data)

