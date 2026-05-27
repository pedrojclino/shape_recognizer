import numpy as np
import json

from .descriptor_utils import ImgData

def save_model_file(model_file_path, labeled_img_datas):

    if not labeled_img_datas:
        return

    data = dict()
    for label, img_datas in labeled_img_datas.items():
        img_datas_list = list()
        for img_data in img_datas:
            img_datas_list.append(img_data.to_json())
        data[label] = img_datas_list

    with open(model_file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, sort_keys=True)

        print("Training model saved to:", model_file_path)
    
def load_model_file(model_file_path):
    pass
    labeled_img_datas = dict()
    try:
        with open(model_file_path,"r") as f:

            img_datas = json.load(f)
            for label, json_img_datas in img_datas.items():
                img_datas : list[ImgData] = list()
                for json_img_data in json_img_datas:
                    img_data = ImgData()
                    img_data.from_json(json_img_data)
                    img_datas.append(img_data)
                labeled_img_datas[label] = img_datas

    except Exception as e:
        print(e)

    return labeled_img_datas

def save_classifications_to_file(output_file_path, img_datas, img_descriptor_classifications, labels, separator = ","):

    with open(output_file_path,"w") as f:
        line = (separator).join(labels)
        f.write(f"correspondance:{separator}{line}\n")

        for img_data in img_datas:
            for descriptor in img_data.descriptors:
                
                descriptor_hash_key = (img_data.name, descriptor.location)
                if descriptor_hash_key not in img_descriptor_classifications:
                    continue
                
                str_prob_list = str()
                classification = img_descriptor_classifications[descriptor_hash_key]
                for label in labels:
                    if label in classification:
                        str_prob_list += separator + f"{classification[label]["score"]:.4f}"
                    else:
                        str_prob_list += separator + f"{np.nan:.4f}"

                line = img_data.name + str(descriptor.location) + str_prob_list
                f.write(f"{line}\n")

        print("Classification results saved to:", output_file_path)