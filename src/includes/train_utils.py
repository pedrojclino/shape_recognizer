import sys
import os
import cv2

from includes.file_utils import save_model_file

from includes.descriptor_utils import gen_descriptors, CombinedDescriptorData, ShapeDescriptor

def train(argv):
    """Creates a model file with the descriptor data from a group of image samples separated on a folder basis"""

    # Validate destination
    if '-dst' not in argv:
        print(f"No '-dst' found on the argument list")
        sys.exit(-1)

    idx = argv.index('-dst')
    if idx+1 >= len(argv):
        print(f"No path set after '-dst'") 
        sys.exit(-1)

    dst_file_path = argv[idx+1]
    texts = os.path.splitext(dst_file_path)
    if len(texts) <= 1 or texts[1] != ".json":
        print(f"Path specified after '-dst' is not a .json.")
        sys.exit(-1)

    # Write results file
    if '-ow' not in argv and os.path.exists(dst_file_path):
        print(f"Output file '{dst_file_path}' already exists. To overwrite it add '-ow' as a flag to the running executable.")
        sys.exit(-1)

    # Validate the input
    if '-src' not in argv:
        print(f"No '-src' found on the argument list")
        sys.exit(-1)

    idx = argv.index('-src')
    if idx +1 >= len(argv) or not os.path.exists(argv[idx+1]):
        print(f"Failed to find the specified path after '-src'.")
        sys.exit(-1)

    training_dir_path = argv[idx+1]

    # Set debug active
    debug = True if '-d' in sys.argv else False

    # Set sensitivity value
    sensitivity = 0.01
    if '-s' in argv:
        idx = argv.index('-s')
        if idx+1 >= len(argv):
            print(f"No sensitivity set after '-s'.") 
            sys.exit(-1)

        if 0 < float(argv[idx+1]) < 0.1:
            sensitivity = float(argv[idx+1])
        else:
            print(f"Invalid sensitivity value. Fallback to 0.01.") 
            sensitivity = 0.01

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
                img = cv2.imread(full_img_path)                
            except Exception as e:
                continue

            # Concat arrays
            descriptors += gen_descriptors(img, debug, sensitivity=sensitivity)

        combined_descriptor_data = CombinedDescriptorData()
        combined_descriptor_data.load_descriptors(descriptors)
        known_shape_descriptors[shape_dir] = combined_descriptor_data

    # Save the model
    save_model_file(dst_file_path, known_shape_descriptors)

    print("Finished")
    sys.exit(0)