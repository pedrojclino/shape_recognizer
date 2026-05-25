import cv2
import numpy as np
import sys
import os

from includes.descriptor_utils import ImageQualification

from includes.file_utils import save_result_file, default_model_file, default_output_file, read_model_file

from includes.ui_utils import user_drawing

from includes.plot_utils import plot_confusion_matrix


# INFO: Compilable in /src with:
# > py -m PyInstaller --onefile --icon=../res/icons/app_icon.ico --distpath ../bin --workpath ../build --specpath ../build --name shape_recoginzer main.py

if __name__ == "__main__":
    
    if len(sys.argv) == 1 or '-h' in sys.argv or '"-help"' in sys.argv:
        print("Use '-p' followed by the path to an image or a folder containing images to be diagnosed for trained shapes.")
        print("Use '-u' to draw an image to be diagnosed for trained shapes.")
        print("Use '-h' for help.")
        print("Add '-r' to overwrite the previous output.csv file.")
        print("Add '-s' to show the shape finding results.")
        print("Add '-r' to overwrite the previous output.csv file.")
        sys.exit(0)

    # Load the JSON file containing teh training
    known_shape_descriptors = read_model_file(default_model_file())

    if not known_shape_descriptors:
        print(f"Failed to read valid model data! Please check '{default_model_file()}' file integrity.")
        sys.exit(-1)

    input_imgs = dict()
    if '-u' in sys.argv:
        input_imgs["user"] = user_drawing()

    # Attempt to find viable images on the user path
    elif '-p' in sys.argv:

        idx = sys.argv.index('-p')
        if idx +1 >= len(sys.argv) or not os.path.exists(sys.argv[idx+1]):
            print(f"Failed to find the specified path after '-p'.")
            sys.exit(-1)
        
        input_path = sys.argv[idx+1]
    
        # For a single image input
        if os.path.isfile(input_path):
            img_file_name = os.path.splitext(os.path.basename(input_path))[0]
            try:
                input_imgs[img_file_name] = cv2.imread(input_path)
            except Exception as e:
                pass

        # For folder with multiple images input
        elif os.path.isdir(input_path):

            img_file_basenames = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))] 
            for img_file_basename in img_file_basenames:
                full_img_path = os.path.join(input_path, img_file_basename)
                img_file_name = os.path.splitext(img_file_basename)[0]
                try:
                    input_imgs[img_file_name] = cv2.imread(full_img_path)
                except Exception as e:
                    pass

    if not input_imgs:
        print(f"Unable to accept the inputs for training. Use '-h' for details on how to run the executable.")
        sys.exit(-1)

    image_qualifications = list()
    for name, img in input_imgs.items():
        img_qualf = ImageQualification(name, img, debug=False, sensitivity=0.01)
        img_qualf.qualify(known_shape_descriptors)
        image_qualifications.append(img_qualf)

    # If descriptors found
    if not any( 0 < len(img_qualf.ordered_descriptors) for img_qualf in image_qualifications):
        print(f"Unable to qualify any of the input images. Couldn't find any correlation between input and training images.")
        sys.exit(-1)

    # Plot results
    if '-s' in sys.argv:
        plot_confusion_matrix(image_qualifications, known_shape_descriptors)

    # Write results file
    output_file_path = default_output_file()
    if '-r' not in sys.argv and os.path.exists(output_file_path):
        print(f"Output file '{output_file_path}' already exists. To overwrite it add '-r' as a flag to the running executable.")
        sys.exit(-1)


    save_result_file(image_qualifications, known_shape_descriptors)

    print("Finished")
    sys.exit(0)