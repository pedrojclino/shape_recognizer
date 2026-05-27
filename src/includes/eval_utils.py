import sys
import os
from pathlib import Path
import cv2

from .classification_utils import classify_descriptor

from .descriptor_utils import ImgData

from .file_utils import load_model_file, save_classifications_to_file

from .drawing_utils import user_drawing

from .plot_utils import plot_xvalidation_matrices

class Settings:

    def __init__(self, sample_path, use_user_input, model_file_path, output_file_path, diagnose, classify, classification_th):
        self.sample_path = sample_path
        self.use_user_input = use_user_input
        self.model_file_path = model_file_path
        self.output_file_path = output_file_path
        self.diagnose = diagnose
        self.classify = classify
        self.classification_th = classification_th

def validate_args(argv):

    # Validate the input
    sample_path = ""
    use_user_input = False
    if '-usr' in argv:
        use_user_input = True
    elif '-src' in argv:
        idx = argv.index('-src')
        if idx+1 >= len(argv) or not os.path.exists(argv[idx+1]):
            print(f"Failed to find the specified path after '-src'.")
            sys.exit(-1)
        sample_path = argv[idx+1]
    else:
        print(f"No '-src' or '-usr' found on the argument list.")
        sys.exit(-1)

    # Validate the model
    if '-model' not in argv:
        print(f"No '-model' found on the argument list.")
        sys.exit(-1)

    idx = argv.index('-model')
    if idx +1 >= len(argv) or not os.path.exists(argv[idx+1]):
        print(f"Failed to find the specified path after '-model'.")
        sys.exit(-1)

    model_file_path = argv[idx+1]

    # Validate the output
    if '-dst' not in argv:
        print(f"No '-dst' found on the argument list.")
        sys.exit(-1)

    idx = argv.index('-dst')
    if idx +1 >= len(argv):
        print(f"Failed to find the specified path after '-dst'.")
        sys.exit(-1)

    output_file_path = argv[idx+1]
    if Path(output_file_path).suffix != ".csv":
        print(f"Destination file extension must be '.csv'.")
        sys.exit(-1)

    # Classification
    classify = True if '-cls' in argv else False
    classification_th = 0
    if classify:
        idx = argv.index('-cls')
        if idx +1 >= len(argv):
            print(f"Failed to accept the threshold value defined after '-cls'.")
            sys.exit(-1)
        classification_th = float(argv[idx+1])

    diagnose = True if '-d' in argv else False
    return Settings(sample_path, use_user_input, model_file_path, output_file_path, diagnose, classify, classification_th)

def read_images(sample_path, use_user_input):
    img_datas : list[ImgData] = list()
    if use_user_input:
        img_data = ImgData("User Drawing", user_drawing())
        img_data.find_descriptors()
        img_datas.append(img_data)
    else:
        if os.path.isdir(sample_path):
            sample_img_paths = [os.path.join(sample_path,f) for f in os.listdir(sample_path) if os.path.isfile(os.path.join(sample_path, f))]
        else:
            sample_img_paths = [sample_path]

        for sample_img_path in sample_img_paths:
            try:
                img_name = Path(sample_img_path).stem
                img_data = ImgData(img_name, cv2.imread(sample_img_path))
                img_data.find_descriptors()
                if img_data.descriptors:
                    img_datas.append(img_data)
            except Exception as e:
                print(e)
                continue

    if not img_datas:
        print("Couldn't evaluate any of the input images for valid descriptors")
        sys.exit(-1)
    
    return img_datas

def eval(argv):
    
    # Interpret args
    settings = validate_args(argv)

    # Load model
    labeled_img_datas = load_model_file(settings.model_file_path)

    # Load input images
    img_datas = read_images(settings.sample_path, settings.use_user_input)

    # Evaluate the input samples
    img_descriptor_classifications = dict()
    for img_data in img_datas:

        for descriptor in img_data.descriptors:

            classification = classify_descriptor(descriptor.contour, labeled_img_datas)
            
            if settings.classify:
                max_score = max([value["score"] for value in classification.values()])
                for label, value in classification.items():
                    classification[label]["score"] = 1 if ( value["score"] == max_score and settings.classification_th < max_score) else 0

            descriptor_hash_key = (img_data.name,descriptor.location)
            img_descriptor_classifications[descriptor_hash_key] = classification

    # Plot results
    if settings.diagnose:
        plot_xvalidation_matrices(img_datas, img_descriptor_classifications, labeled_img_datas.keys(), settings.classification_th)

    # Save to results to csv file
    save_classifications_to_file(settings.output_file_path, img_datas, img_descriptor_classifications, labeled_img_datas.keys())

