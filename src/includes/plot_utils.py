import numpy as np
import threading

# Set to an interactive backend suitable for your OS
import matplotlib
matplotlib.use('TkAgg')  # Options: 'TkAgg', 'QtAgg', 'MacOSX'

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Shared flag to signal exit
exit_event = threading.Event()

# Function to wait for Enter key press
def wait_for_enter():
    input("Press Enter to continue...\n")
    exit_event.set()

def plot_xvalidation_matrices(img_datas, img_descriptor_classifications, labels, classification_th):

    for img_data in img_datas:
        
        # Setup plot handles
        fig = plt.figure(figsize=(11, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], figure=fig)

        # Show the image
        ax_left = fig.add_subplot(gs[:, 0])
        ax_left.set_title(img_data.name)
        ax_left.imshow(img_data.img)
        ax_left.set_aspect('equal', adjustable='box')
        ax_left.axis('off')

        # Show descriptors and their classification
        for descriptor_idx, descriptor in enumerate(img_data.descriptors):
            
            descriptor_hash_key = (img_data.name,descriptor.location)
            if descriptor_hash_key not in img_descriptor_classifications:
                continue

            classification = img_descriptor_classifications[descriptor_hash_key]

            # Get the label with the highest score
            label = max(classification, key=lambda i: classification[i]["score"])
            if classification[label]["score"] < classification_th:
                label = None

            # Label text
            location = descriptor.location
            prefix = str(descriptor_idx) + ":"
            ax_left.text(location[0], location[1], prefix, fontsize=12, ha='left', va='bottom', color = 'blue')
            ax_left.text(location[0] + 15 * len(prefix), location[1], label, fontsize=12, ha='left', va='bottom', color = "magenta")
            
            # Input shape contour
            shape_contour = classification[label]["input_poly"]
            if shape_contour.shape[0]:
                closed_contour = np.vstack([shape_contour, shape_contour[0,:]])
                ax_left.plot(closed_contour[:, 0], closed_contour[:, 1], color='blue', linewidth=2, alpha = 0.5)

            # Associated closest labeled reference contour, used for the classification
            if not label:
                continue

            # The the best knwon aproximation contour 
            best_known_poly_aprox = classification[label]["poly"]
            if best_known_poly_aprox.shape[0]:
                closed_contour = np.vstack([best_known_poly_aprox, best_known_poly_aprox[0,:]])
                ax_left.plot(closed_contour[:, 0], closed_contour[:, 1], color='magenta', linewidth=2, alpha = 0.5)

        # Right side: Shape validation matrix
        ax_right = fig.add_subplot(gs[:, 1])
        ax_right.set_title("Validation Matrix")
        ax_right.set_aspect('equal', adjustable='box')


        # Build validation matrix
        n_labels = len(labels)
        validation_matrix = np.empty((0,n_labels))
        for descriptor_idx, descriptor in enumerate(img_data.descriptors):
            
            descriptor_hash_key = (img_data.name,descriptor.location)
            if descriptor_hash_key not in img_descriptor_classifications:
                continue

            classification = img_descriptor_classifications[descriptor_hash_key]

            prob_list = [value["score"] for value in classification.values()]

            validation_matrix = np.vstack((validation_matrix, prob_list)) 

        # Show matrix
        ax_right.imshow(validation_matrix, cmap='Greens', interpolation='nearest', vmin=0, vmax=1)
        ax_right.set_xticks(np.arange(-0.5, validation_matrix.shape[1], 1), minor=True)
        ax_right.set_yticks(np.arange(-0.5, validation_matrix.shape[0], 1), minor=True)
        ax_right.grid(which='minor', color='black', linewidth=1)
        ax_right.tick_params(which='minor', size=0)
        
        # Add text annotations
        for i in range(validation_matrix.shape[0]):
            max_idx = np.argmax(validation_matrix[i, :])
            for j in range(validation_matrix.shape[1]):
                ax_right.text(j, i, f"{validation_matrix[i, j]:.3f}", ha='center', va='center', color='black', 
                        fontweight='bold' if j == max_idx else 'normal' )

        ax_right.set_xticks(list(range(0,len(labels),1)))
        ax_right.set_xticklabels(labels, rotation=45)
        ax_right.tick_params(axis='x',labelcolor='magenta')
        
        input_names = [str(idx) for idx in range(len(img_data.descriptors))]
        ax_right.set_yticks(list(range(0,len(input_names),1)))
        ax_right.set_yticklabels(input_names, rotation=45)
        ax_right.tick_params(axis='y',labelcolor='blue')
        
    plt.tight_layout()
    plt.ion()  # Enable interactive mode
    plt.rcParams["figure.raise_window"] = False # Stop plt windows from raising on pause()
    
    # Allow the user to proceed on "Enter"
    input_thread = threading.Thread(target=wait_for_enter, daemon=True)
    input_thread.start()
    plt.show(block=False)

    # Poll until all windows are closed
    while not exit_event.is_set():
        # Keeps GUI responsive
        fig.canvas.draw()
        fig.canvas.flush_events()

        if not plt.get_fignums():
            print("All windows closed.")
            break

    plt.close('all')


def plot_descriptors(labeled_img_datas):

    for label, img_datas in labeled_img_datas.items():
        for img_data in img_datas:

            # Setup plot handles
            fig, ax = plt.subplots()
            fig.suptitle(f"Descriptors with Label: {label}")

            # Show the image
            ax.set_title(img_data.name)
            ax.imshow(img_data.img)
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')
            
            # Show descriptors and their classification
            for descriptor in img_data.descriptors:
                closed_contour = np.vstack([descriptor.contour, descriptor.contour[0]])
                ax.plot(closed_contour[:, 0], closed_contour[:, 1], color='blue', linewidth=2)

    plt.tight_layout()
    plt.ion()  # Enable interactive mode
    plt.rcParams["figure.raise_window"] = False # Stop plt windows from raising on pause()
    
    # Allow the user to proceed on "Enter"
    input_thread = threading.Thread(target=wait_for_enter, daemon=True)
    input_thread.start()
    plt.show(block=False)

    # Poll until all windows are closed
    while not exit_event.is_set():
        # Keeps GUI responsive
        fig.canvas.draw()
        fig.canvas.flush_events()

        if not plt.get_fignums():
            print("All windows closed.")
            break

    plt.close('all')