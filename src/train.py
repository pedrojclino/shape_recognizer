import sys
import os
import cv2

# INFO: Compilable in /src with:
# > py -m PyInstaller --onefile --icon=../res/icons/app_icon.ico --distpath ../bin --workpath ../build --specpath ../build --name train_shapes train.py

from includes.file_utils import default_model_file, save_model_file

from includes.descriptor_utils import gen_descriptors, CombinedDescriptorData, ShapeDescriptor

if __name__ == "__main__":

    if len(sys.argv) == 1 or '-h' in sys.argv or '"-help"' in sys.argv:
        print("Use '-p' followed by the path to a folder containing subfolders each with a group of references images to compose the model.")
        print("Add '-r' to overwrite the previous model.json file.")
        print("Add '-d' to debug the shapes being trained.")
        print("Use '-h' for help.")
        sys.exit(0)

    if not '-p' in sys.argv:
        print("Missing a folder path as the argument! Missing the '-p' flag.")
        sys.exit(-1)

    idx = sys.argv.index('-p')
    if idx +1 >= len(sys.argv) or not os.path.exists(sys.argv[idx+1]):
        print(f"Failed to find the specified path after '-p'.")
        sys.exit(-1)

    training_dir_path = sys.argv[idx+1]
    if not os.path.isdir(training_dir_path):
        print("Expecting the folder to have multiple sub folders with the images to be used for descriptor training.")
        sys.exit(-1)

    model_file_path = default_model_file()
    if '-r' not in sys.argv and os.path.exists(model_file_path):
        print(f"Model file '{model_file_path}' already exists. To overwrite it add '-r' as a flag to the running executable.")
        sys.exit(-1)

    debug = True if '-d' in sys.argv else False

    # Read multiple images and build a model
    known_shape_descriptors = dict()
    shape_dirs = [f for f in os.listdir(training_dir_path) if os.path.isdir(os.path.join(training_dir_path, f))] 
    for shape_dir in shape_dirs:
        shape_dir_path = os.path.join(training_dir_path, shape_dir)
        training_imgs = [f for f in os.listdir(shape_dir_path) if os.path.isfile(os.path.join(shape_dir_path, f))]
        
        descriptors : list[ShapeDescriptor] = list()
        for training_img in training_imgs:
            full_img_path = os.path.join(shape_dir_path, training_img)

            try:
                # Concat arrays
                descriptors += gen_descriptors(cv2.imread(full_img_path), debug)
            except Exception as e:
                pass

        combined_descriptor_data = CombinedDescriptorData()
        combined_descriptor_data.load_descriptors(descriptors)
        known_shape_descriptors[shape_dir] = combined_descriptor_data

    # Save the model
    save_model_file(known_shape_descriptors)

    print("Finished")
    sys.exit(0)