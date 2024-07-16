import cv2
import pickle
import numpy as np
from skimage.transform import resize


# helper function
def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots

EMPTY = True
NOT_EMPTY = False

MODEL = pickle.load(open("./model/model.p", "rb"))


def empty_or_not(spot_bgr):

    flat_data = []

    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY

video_path = './data/parking_crop_loop.mp4'
mask_path = './data/mask_crop.png'

cap = cv2.VideoCapture(video_path)

mask = cv2.imread(mask_path, 0) # 0: gray scale

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S) # find the connected components by using mask

spots = get_parking_spots_bboxes(connected_components)

print(spots) 

ret = True
frame_no = 0
step = 30

spots_status = [None for j in spots]

while ret:
    ret, frame = cap.read() # ret = if a frame is read successfully (boo), frame = actual image

    if frame_no % step == 0:

        for spot_idx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_idx] = spot_status

    for spot_idx, spot in enumerate(spots):
        spot_status = spots_status[spot_idx]
        x1, y1, w, h = spots[spot_idx]

        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    cv2.imshow('frame', frame) # show video
    if cv2.waitKey(25) & 0xFF == ord('q'): # close the window if press 'q'
        break

    frame_no += 1