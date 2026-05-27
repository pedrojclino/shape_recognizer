import cv2
import numpy as np

def user_drawing():
    # Create a blank white image
    img = np.zeros((512, 512, 3), np.uint8)
    drawing = False  # True if mouse is pressed

    def draw_circle(event, x, y, flags, param):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            param["drawing"] = True
            param["adding"] = True
            cv2.circle(param["img"], (x, y), 5, (255,255,255), -1)
        if event == cv2.EVENT_RBUTTONDOWN:
            param["drawing"] = True
            param["adding"] = False
            cv2.circle(param["img"], (x, y), 5, (0,0,0), -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if param["drawing"]:
                if param["adding"]:
                    cv2.circle(param["img"], (x, y), 5, (255,255,255), -1)
                else:
                    cv2.circle(param["img"], (x, y), 5, (0,0,0), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            param["drawing"] = False
            param["adding"] = True
            cv2.circle(param["img"], (x, y), 5, (255,255,255), -1)        
        elif event == cv2.EVENT_RBUTTONUP:
            param["drawing"] = False
            param["adding"] = False
            cv2.circle(param["img"], (x, y), 5, (0,0,0), -1)

    window_name = 'Draw with Mouse (c: clear, a: accept)'
    cv2.namedWindow(window_name)
    param = dict()
    param["img"] = img
    param["drawing"] = drawing
    param["adding"] = True
    cv2.setMouseCallback(window_name, draw_circle, param)

    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF  # Wait for a key press
        if key == ord('c'):
            param["img"].fill(0)
        if key == ord('a') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

    img = param["img"]
    
    # Connect close strokes
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=5)
    eroded = cv2.erode(dilated, kernel, iterations=5)

    return eroded