import sys
import os
import json

from .descriptor_utils import CombinedDescriptorData

def default_model_file():
    
    try:
        sys._MEIPASS
        return "model.json"
    except Exception:
        return "..\\bin\\model.json"
    
def default_output_file():
    
    try:
        sys._MEIPASS
        return "output.csv"
    except Exception:
        return "..\\bin\\output.csv"
    
def read_model_file(default_model_file):

    known_shape_descriptors = dict()
    try:
        with open(default_model_file,"r") as f:

            data = json.load(f)
            for shape_name, json_descriptors in data.items():
                combined_descriptor_data = CombinedDescriptorData()
                combined_descriptor_data.load_json(json_descriptors)
                known_shape_descriptors[shape_name] = combined_descriptor_data
    except Exception as e:
        print(e)

    return known_shape_descriptors

def save_model_file(known_shape_descriptors):

    if not known_shape_descriptors:
        return

    data = dict()
    for shpe_name, shape_descriptors in known_shape_descriptors.items():
        data[shpe_name] = shape_descriptors.to_json()

    with open(default_model_file(), "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, sort_keys=True)

        print("Training model saved to:", default_model_file())

def save_result_file(image_qualifications, known_shape_descriptors):

    with open(default_output_file(),"w") as f:
        line = (",").join(known_shape_descriptors.keys())
        f.write(f"correspondance:,{line}\n")

        for img_qualf in image_qualifications:
            line = str()
            line = img_qualf.name
            for ordered_descriptor in img_qualf.ordered_descriptors:
                for known_shape_name in known_shape_descriptors.keys():
                    if known_shape_name in ordered_descriptor.scores:
                        line += "," + f"{ordered_descriptor.scores[known_shape_name]:.4f}"
                    else:
                        line += "," + f"{np.nan}" # Error scenario
            f.write(f"{line}\n")

        print("Shape results saved to:", default_output_file())