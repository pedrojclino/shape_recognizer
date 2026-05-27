import numpy as np
import cv2

def gen_descriptors(image, debug : bool, sensitivity):

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

            # Filter contours
            x, y, w, h = cv2.boundingRect(ext_contour)
            if w * h < sensitivity * img_area or 0.8 * img_area < w * h:
                continue

            # Create a roi of the binary filled with white contour
            binary_local = np.zeros((image.shape[0] , image.shape[1]), dtype=np.uint8)
            
            cv2.drawContours(binary_local, [ext_contour], -1, (255, 255, 255), thickness=-1)

            sub_contours, _ = cv2.findContours(binary_local, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(sub_contours) == 0:
                continue

            descriptors.append(ShapeDescriptor((x, y, w, h), sub_contours[0]))

            if debug:
                for pt_idx, pt in enumerate(descriptors[-1].contour):
                   cv2.putText(image, str(pt_idx), (pt[0], pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.drawContours(image, [ descriptors[-1].contour ], -1, (0, 0, 255), 2)
                cv2.imshow("Debug",image)
                cv2.waitKey(1000)

    return descriptors


class ImageQualification:

    valid_score_th = 0.7

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
            self.versors = None
            self.resolution = None

        def to_json_descriptor(self):
            json_descriptor = dict()
            json_descriptor["versors"] = self.versors.tolist()
            return json_descriptor

        def from_json_descriptor(self, json_descriptor : dict):
            self.versors = np.array(json_descriptor["versors"])
            self.resolution = len(self.versors)

        def from_shape_descriptor(self, shape_descriptor : ShapeDescriptor):
            self.versors = shape_descriptor.versors
            self.resolution = len(self.versors)

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
    
            # Only compare descriptors with different resolutions on specific cases
            if shape_descriptor.resolution == 0:
                continue
            
            if local_descriptor.resolution != shape_descriptor.resolution:
                continue
            
            # Evaluate vector differences
            versors_diff = local_descriptor.versors - shape_descriptor.versors
            n = np.sum([np.linalg.norm(vec) for vec in versors_diff])
            descriptor_similarities[idx] = 1 - n / 2

        if 0 < len(descriptor_similarities):
            return descriptor_similarities.max()
        else:
            return 0

class ShapeDescriptor:

    def __init__(self, region, contour):
        self.region = region # (x, y, w, h)

        # Simplify the contour
        arc_len = cv2.arcLength(contour, True)
        epsilon = 0.04* arc_len
        self.contour = cv2.approxPolyDP(contour, epsilon, True)

        self.contour = self.contour.reshape(-1, 2)


        arc_len = cv2.arcLength(self.contour, True)
        p_range = range(1,len(self.contour))
        self.resolution = len(p_range)
        lengths = np.empty(self.resolution)
        self.versors = np.empty((self.resolution,2))
        for idx, p_idx in enumerate(p_range):
            p_prev = self.contour[p_idx-1]
            p_curr = self.contour[p_idx]

            v_prev = p_curr - p_prev

            v_prev_norm = np.linalg.norm(v_prev)

            lengths[idx] = v_prev_norm
            self.versors[idx] = v_prev / v_prev_norm

        # Multiply versors by their normalized lengths
        lengths /= arc_len
        self.versors = self.versors * lengths.reshape(-1, 1)


