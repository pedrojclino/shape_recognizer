import numpy as np
import cv2
from scipy import signal

def find_contour_children(child_contours, parent_idx, root_hierarchy):
    for child_idx, row in enumerate(root_hierarchy):
        if row[3] == parent_idx:
            child_contours.append(child_idx)
            find_contour_children(child_contours, child_idx, root_hierarchy)

def gen_descriptors(image, debug : bool, sensitivity = 0.01):

    # Binarize
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    _, binary =  cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    root_contours, root_hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    descriptors = list()
    img_area = (image.shape[0] * image.shape[1])
    
    # Consider only contours on the top most hierarchy
    for idx in range(0,len(root_contours)):
        
        # Only run the external edge contours
        if root_hierarchy[0][idx][3] == -1:
            ext_contour = root_contours[idx]

            # Count child contours recursively
            child_contour_indexes = []
            find_contour_children(child_contour_indexes, idx, root_hierarchy[0])
            n_child_contours = len(child_contour_indexes) 

            # Filter contours
            x, y, w, h = cv2.boundingRect(ext_contour)
            if w * h < sensitivity * img_area or 0.8 * img_area < w * h:
                continue

            # Create a roi of the binary filled with white contour
            binary_local = np.zeros((image.shape[0] , image.shape[1]), dtype=np.uint8)
            cv2.drawContours(binary_local, [ext_contour], -1, (255, 255, 255), thickness=-1)

            sub_contours, _ = cv2.findContours(binary_local, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            descriptors.append(ShapeDescriptor((x, y, w, h), sub_contours[0], n_child_contours))

            if debug:
                cv2.drawContours(image, [ descriptors[-1].simple_contour ], -1, (0, 0, 255), 2)
                cv2.imshow("Debug",image)
                cv2.waitKey(1000)

    return descriptors


class ImageQualification:

    class NamedDescriptor:
        def __init__(self, name, descriptor):
            self.name = name
            self.descriptor = descriptor
            self.scores = dict()

    def __init__(self, name, img, debug, sensitivity):
        self.name = name
        self.img = img
        self.ordered_descriptors : list[ImageQualification.NamedDescriptor] = list()
        descriptors = gen_descriptors(img, debug, sensitivity)
        for idx, descriptor in enumerate(descriptors):
            self.ordered_descriptors.append(ImageQualification.NamedDescriptor(f"{idx}", descriptor))

    def qualify(self, known_shape_descriptors : list[CombinedDescriptorData]):
        for known_shape_name, known_shape_descriptor in known_shape_descriptors.items():
            for ordered_descriptor in self.ordered_descriptors:
                ordered_descriptor.scores[known_shape_name] = known_shape_descriptor.compare(ordered_descriptor.descriptor)

class CombinedDescriptorData:

    class LocalDescriptorData:
        def __init__(self):
            self.n_child_contours = None
            self.resolution = None
            self.lengths = None
            self.angles = None

        def to_json_descriptor(self):
            json_descriptor = dict()
            json_descriptor["angles"] = self.angles.tolist()
            json_descriptor["lengths"] = self.lengths.tolist()
            json_descriptor["n_child_contours"] = self.n_child_contours
            json_descriptor["resolution"] = self.resolution
            return json_descriptor

        def from_json_descriptor(self, json_descriptor : dict):
            self.n_child_contours = json_descriptor["n_child_contours"]
            self.resolution = json_descriptor["resolution"]
            self.lengths = np.array(json_descriptor["lengths"])
            self.angles = np.array(json_descriptor["angles"])

        def from_shape_descriptor(self, shape_descriptor : ShapeDescriptor):
            self.n_child_contours = shape_descriptor.n_child_contours
            self.resolution = shape_descriptor.resolution
            self.lengths = shape_descriptor.lengths
            self.angles = shape_descriptor.angles

    def __init__(self):
        self.local_descriptors : list[CombinedDescriptorData.LocalDescriptorData] = list() 

    def load_json(self, json_descriptors):
        self.local_descriptors : list[CombinedDescriptorData.LocalDescriptorData] = list() 
        for json_descriptor in json_descriptors:
            local_descriptor_data = CombinedDescriptorData.LocalDescriptorData()
            local_descriptor_data.from_json_descriptor(json_descriptor)
            self.local_descriptors.append(local_descriptor_data)

    def to_json(self):
        json_descriptors = list()
        for descriptor in self.local_descriptors:
            json_descriptors.append(descriptor.to_json_descriptor())

        return json_descriptors

    def load_descriptors(self, shape_descriptors : list[ShapeDescriptor]):
        self.local_descriptors : list[CombinedDescriptorData.LocalDescriptorData] = list() 
        for shape_descriptor in shape_descriptors:
            descriptor = CombinedDescriptorData.LocalDescriptorData()
            descriptor.from_shape_descriptor(shape_descriptor)
            self.local_descriptors.append(descriptor)

    def compare(self, shape_descriptor : ShapeDescriptor):

        n = len(self.local_descriptors)
        descriptor_similarities = np.zeros(n)
        for idx, local_descriptor in enumerate(self.local_descriptors):

            # Only compare descriptors of the same size
            if local_descriptor.resolution != shape_descriptor.resolution:
                continue

            if local_descriptor.n_child_contours != shape_descriptor.n_child_contours:
                continue
            
            # Improve correlations by aligning by angles
            a_centered = local_descriptor.angles - np.mean(local_descriptor.angles)
            b_centered = shape_descriptor.angles - np.mean(shape_descriptor.angles)

            # Compute cross-correlation
            corr = signal.correlate(a_centered, b_centered, mode='full')
            shift = np.argmax(corr) - (local_descriptor.resolution - 1)
            if local_descriptor.resolution // 4 < np.abs(shift): # Protect agaist -+90º rotations
                shift = 0 
            shape_descriptor_angles = np.roll(shape_descriptor.angles, shift)
            shape_descriptor_lengths = np.roll(shape_descriptor.lengths, shift)

            # NOTE: already normalized
            angle_sim = local_descriptor.angles - shape_descriptor_angles
            angle_sim = 1 - np.dot(angle_sim,angle_sim)
            lenght_sim = local_descriptor.lengths - shape_descriptor_lengths
            lenght_sim = 1 - np.dot(lenght_sim,lenght_sim)

            descriptor_similarities[idx] = angle_sim * lenght_sim

        if 0 < len(descriptor_similarities):
            return descriptor_similarities.max()
        else:
            return 0

class ShapeDescriptor:
    
    def __init__(self, region, contour, n_child_contours):
        self.region = region # (x, y, w, h)
        self.n_child_contours = n_child_contours

        # Approximate the contour
        arc_len = cv2.arcLength(contour, True)
        epsilon = 0.04* arc_len
        self.simple_contour = cv2.approxPolyDP(contour, epsilon, True)

        self.simple_contour = self.simple_contour.reshape(-1, 2)

        p_range = range(1,len(self.simple_contour))
        self.resolution = len(p_range)
        self.angles = np.empty(self.resolution)
        self.lengths = np.empty(self.resolution)
        for idx, p_idx in enumerate(p_range):
            p_prev = self.simple_contour[p_idx-1]
            p_curr = self.simple_contour[p_idx]

            if p_idx+1 >= len(self.simple_contour):
                p_next = self.simple_contour[0]
            else:
                p_next = self.simple_contour[p_idx+1]

            v_prev = p_curr - p_prev
            v_next = p_next - p_curr

            v_prev_norm = np.linalg.norm(v_prev)
            v_prev = v_prev / v_prev_norm
            v_next = v_next / np.linalg.norm(v_next)
            self.angles[idx] = np.dot(v_prev,v_next)
            
            v_perp = np.cross(v_prev,v_next)
            if v_perp < 0:
                self.angles[idx] *=-1

            self.lengths[idx] = v_prev_norm

        # Normalize [0..1]
        self.angles = (self.angles + np.pi) / (2 * np.pi)
        self.lengths /= arc_len


