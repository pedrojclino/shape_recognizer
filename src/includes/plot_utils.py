import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import Rectangle

def plot_confusion_matrix(image_qualifications, known_shape_descriptors):


    for img_qualf in image_qualifications:
        
        # Setup plot handles
        fig = plt.figure(figsize=(12, 3))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], figure=fig)

        # Show the image
        ax_left = fig.add_subplot(gs[:, 0])
        ax_left.set_title(img_qualf.name)
        ax_left.imshow(img_qualf.img)
        ax_left.set_aspect('equal', adjustable='box')
        ax_left.axis('off')

        # Add found descriptors
        for ordered_descriptor in img_qualf.ordered_descriptors:

            best_shape_name = max(ordered_descriptor.scores, key=ordered_descriptor.scores.get)

            # Threshold that defines unknown
            if ordered_descriptor.scores[best_shape_name] < 0.01:
                best_shape_name = "?"

            # TEXT
            descriptor_region = ordered_descriptor.descriptor.region
            text = ordered_descriptor.name + ":" + best_shape_name
            ax_left.text(descriptor_region[0], descriptor_region[1], text , fontsize=12, ha='left', va='bottom', color = "blue")
            
            # ROI
            rect = Rectangle((descriptor_region[0], descriptor_region[1]), descriptor_region[2], descriptor_region[3], linewidth=1, edgecolor='b', facecolor='none')
            ax_left.add_patch(rect)

        # Right side: Shape comfusion matrix
        ax_right = fig.add_subplot(gs[:, 1])
        ax_right.set_title("Confusion Matrix")
        ax_right.set_aspect('equal', adjustable='box')


        # Build confusion matrix
        n_known_shapes = len(known_shape_descriptors.keys())
        confusion_matrix = np.empty((0,n_known_shapes))
        for ordered_descriptor in img_qualf.ordered_descriptors:
            new_col = np.empty(n_known_shapes)
            for idx, known_shape_name in enumerate(known_shape_descriptors.keys()):
                if known_shape_name in ordered_descriptor.scores:
                    new_col[idx] = ordered_descriptor.scores[known_shape_name]
                else:
                    new_col[idx] = np.nan
            confusion_matrix = np.vstack((confusion_matrix, new_col)) 

        # Show matrix
        ax_right.imshow(confusion_matrix, cmap='Greens', interpolation='nearest', vmin=0, vmax=1)
        ax_right.set_xticks(np.arange(-0.5, confusion_matrix.shape[1], 1), minor=True)
        ax_right.set_yticks(np.arange(-0.5, confusion_matrix.shape[0], 1), minor=True)
        ax_right.grid(which='minor', color='black', linewidth=1)
        ax_right.tick_params(which='minor', size=0)
        
        # Add text annotations
        for i in range(confusion_matrix.shape[0]):
            max_idx = np.argmax(confusion_matrix[i, :])
            for j in range(confusion_matrix.shape[1]):
                ax_right.text(j, i, f"{confusion_matrix[i, j]:.4f}", ha='center', va='center', color='black', 
                        fontweight='bold' if j == max_idx else 'normal' )

        shape_names = known_shape_descriptors.keys()
        ax_right.set_xticks(list(range(0,len(shape_names),1)))
        ax_right.set_xticklabels(shape_names)
        
        input_names = [ordered_descriptor.name for ordered_descriptor in img_qualf.ordered_descriptors]
        ax_right.set_yticks(list(range(0,len(input_names),1)))
        ax_right.set_yticklabels(input_names)
        
    plt.tight_layout()
    plt.show(block=False)
    input("Press 'Enter' to close the plots.")
    plt.close('all')