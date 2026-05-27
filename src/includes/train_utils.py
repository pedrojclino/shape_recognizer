import sys
import os
from pathlib import Path
import cv2

from .file_utils import save_model_file

from .descriptor_utils import ImgData

from .plot_utils import plot_descriptors

class Settings:

    def __init__(self, training_dir_path, model_file_path, diagnose):
        self.training_dir_path = training_dir_path
        self.model_file_path = model_file_path
        self.diagnose = diagnose

def validate_args(argv):

    # Validate the input
    if '-src' not in argv:
        print(f"No '-src' found on the argument list.")
        sys.exit(-1)

    idx = argv.index('-src')
    if idx +1 >= len(argv) or not os.path.exists(argv[idx+1]):
        print(f"Failed to find the specified path after '-src'.")
        sys.exit(-1)

    training_dir_path = argv[idx+1]

    # Validate the output
    if '-dst' not in argv:
        print(f"No '-dst' found on the argument list.")
        sys.exit(-1)

    idx = argv.index('-dst')
    if idx+1 >= len(argv):
        print(f"Failed to find the specified path after '-dst'.")
        sys.exit(-1)

    model_file_path = argv[idx+1]
    if Path(model_file_path).suffix != ".json":
        print(f"Destination file extension must be '.json'.")
        sys.exit(-1)

    if os.path.exists(model_file_path) and not '-ow' in argv:
        print(f"Destination file already exists. To overwrite add the flag '-ow'.")
        sys.exit(-1)

    diagnose = True if '-d' in argv else False
    return Settings(training_dir_path, model_file_path, diagnose)

def read_images(imgs_dir_path):
    labeled_img_datas = dict()
    labeled_dirs = [f for f in os.listdir(imgs_dir_path) if os.path.isdir(os.path.join(imgs_dir_path, f))] 
    for labeled_dir in labeled_dirs:
        labeled_dir_path = os.path.join(imgs_dir_path, labeled_dir)
        training_imgs = [f for f in os.listdir(labeled_dir_path) if os.path.isfile(os.path.join(labeled_dir_path, f))]

        # Create a label based on the name of the folder
        label = Path(labeled_dir_path).name
        img_datas : list[ImgData] = list()
        for training_img in training_imgs:
            full_img_path = os.path.join(labeled_dir_path, training_img)
            try:
                img_data = ImgData(training_img, cv2.imread(full_img_path))
                img_data.find_descriptors()
                if img_data.descriptors: # No point adding if no descriptors were found
                    img_datas.append(img_data)
            except Exception as e:
                print(e)
                continue

        if img_datas:
            labeled_img_datas[label] = img_datas

    return labeled_img_datas

def train(argv):

    # Interpret args
    settings = validate_args(argv)

    # Read multiple images and build a model
    labeled_img_datas = read_images(settings.training_dir_path)
    if not labeled_img_datas:
        print("Unable to find descriptors on the training dataset!")
        sys.exit(-1)

    # Diagnose the descriptors
    if settings.diagnose:
        plot_descriptors(labeled_img_datas)

    # Save the model
    save_model_file(settings.model_file_path, labeled_img_datas)