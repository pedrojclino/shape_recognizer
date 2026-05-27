import numpy as np
import cv2

class ImgData:

    def __init__(self, name = None, img = None):
        self.name = name
        self.img = img
        self.descriptors : list[Descriptor] = list()
    
    def find_descriptors(self):
        self.descriptors = find_descriptors(self.img)

    def to_json(self):
        json_data = dict()
        json_data["name"] = self.name
        #json_data["img"] = self.img # NOTE: currently unused
        json_descriptors = list()
        for descriptor in self.descriptors:
            json_descriptors.append({"location":descriptor.location,"contour":descriptor.contour.tolist()})
        json_data["descriptors"] = json_descriptors

        return json_data

    def from_json(self,json_data):
        self.name = json_data["name"]
        self.img = None
        self.descriptors : list[Descriptor] = list()
        json_descriptors = json_data["descriptors"]
        for json_descriptor in json_descriptors:
            self.descriptors.append(Descriptor(json_descriptor["location"], np.array(json_descriptor["contour"])))
    
def find_descriptors(image):

    # Binarize
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    _, binary =  cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    root_contours, root_hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    descriptors = list()
    
    # Consider only contours on the top most hierarchy
    for idx in range(0,len(root_contours)):
        
        # Only run the external edge contours
        if root_hierarchy[0][idx][3] == -1:
            external_contour = root_contours[idx]

            # Create a discriptor based uniquely on the outer contour
            local_binary = np.zeros((image.shape[0] , image.shape[1]), dtype=np.uint8)
            cv2.drawContours(local_binary, [external_contour], -1, (255, 255, 255), thickness=-1)

            local_contour, _ = cv2.findContours(local_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if not local_contour:
                continue

            # Append descriptor
            descriptors.append(Descriptor(cv2.boundingRect(external_contour), local_contour[0].squeeze()))

    return descriptors

class Descriptor:

    def __init__(self, location, contour):
        self.location = location # (x, y, w, h)
        self.contour = contour

    def foo(self):

        # Simplify the contour
        arc_len = cv2.arcLength(self.contour, True)
        epsilon = 0.04* arc_len
        self.contour = cv2.approxPolyDP(self.contour, epsilon, True)

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


