import torch
import cv2
import time
import torchvision
import numpy as np
import easyocr
import imutils
from craft_text_detector import Craft
import os


### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx(frame, model, conf=0.25, iou=0.45, agnostic = False, multi_label = False, classes = None, max_det = 1000, amp = False):
    frame = [frame]
    model.conf = conf
    model.iou = iou
    model.agnostic = agnostic
    model.multi_label = multi_label
    model.classes = classes
    model.max_det = max_det
    model.amp = amp
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])
    # labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return results

def draw_number_licence(frame, text_result=None, cord=None):
    x1,y1,x2,y2 = cord
    x1 = int(x1.numpy()) + 1
    x2 = int(x2.numpy()) + 1
    y1 = int(y1.numpy()) + 1
    y2 = int(y2.numpy()) + 1

    if text_result:
        lw = max(round(sum(frame.shape) / 2 * 0.003), 2)  # line width
        txt_color = (255, 255, 255)
        tf = max(lw - 1, 1)

        w, h = cv2.getTextSize(text_result[0], 0, fontScale=lw / 3, thickness=tf)[0]
        for c, number in enumerate(text_result):
            cv2.putText(frame, str(number), (x1, y2 + int(h * (1.8 * c + 1))), 0,
                        lw / 3,
                        txt_color, thickness=tf, lineType=cv2.LINE_AA)


### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame, classes):
    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels
    """
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    ### looping through the detections
    lw = max(round(sum(frame.shape) / 2 * 0.003), 2)  # line width
    color = (0, 0, 255)
    txt_color = (255, 255, 255)
    tf = max(lw - 1, 1)
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55:  ### threshold value for detection. We are discarding everything below this value

            text_d = classes[int(labels[i])]
            p1, p2 = (int(row[0] * x_shape), int(row[1] * y_shape)), (int(row[2] * x_shape), int(row[3] * y_shape))
            cv2.rectangle(frame, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

            if text_d:
                # tf = max(lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(text_d, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(frame, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(frame,
                            text_d + f' {round(float(row[4]), 2)}', (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            lw / 3,
                            txt_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA)
            # print(text_d)

def check_shape(image):
    h, w = image.shape[:2]
    if w > h:
        return True
    else:
        return False

def xoay_anh(img, p1, p2, p3, p4, width, height):
    pts1 = np.float32([p1, p2, p3, p4])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (width, height))

    return dst

def xu_ly_text(image, reader):
    blur = cv2.medianBlur(image, 7)
    craft = Craft(crop_type="poly", cuda=False)
    prediction_result = craft.detect_text(blur)

    boxes = prediction_result['boxes']

    craft.unload_craftnet_model()
    craft.unload_refinenet_model()

    text_result = []
    for p in boxes:
        # print(p)
        p1 = (p[0, :])
        p2 = (p[1, :])
        p3 = (p[2, :])
        p4 = (p[3, :])

        p1 = [int(p1[0]), int(p1[1])]
        p2 = [int(p2[0]), int(p2[1])]
        p3 = [int(p3[0]), int(p3[1])]
        p4 = [int(p4[0]), int(p4[1])]

        width = int(np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2)))
        height = int(np.sqrt(((p1[0] - p4[0]) ** 2) + ((p1[1] - p4[1]) ** 2)))

        new_image = xoay_anh(image, p1, p2, p4, p3, width + 1, height + 1)

        # hsv = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)
        #
        # imgHue, imgSaturation, imgValue = cv2.split(hsv)
        #
        # kernel = np.ones((5, 5), np.uint8)
        # erode = cv2.dilate(imgValue, kernel)

        result = reader.readtext(new_image, detail=1)
        for (bbox, text, prob) in result:
            text = "".join(
                [c if 96 < ord(c) < 123 or 64 < ord(c) < 91 or 58 > ord(c) > 47 else "" for c in text]).strip()
            text.upper()
            text_result.append(text)

    print(text_result)

    return text_result

### ---------------------------------------------- Main function -----------------------------------------------------
def main(imgs=None, vid_path=None, vid_out=None, img_path_save=None):
    model = torch.hub.load('D:/bai tap python/yolov5-master', 'custom', source='local', path='best.pt', force_reload=True)

    # model2 = torch.hub.load('D:/bai tap python/yolov5-master', 'custom', source='local', path='recognition_best.pt', force_reload=True)

    classes = model.names  ### class names in string format

    # reader = easyocr.Reader(['en'], gpu=False)
    reader = easyocr.Reader(['en'], gpu=False)
    if imgs != None:
        for img in imgs[25:26]:
            print(f"[INFO] Working with image: {img}")
            frame = cv2.imread(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = detectx(frame, model=model, conf=0.25)  ### DETECTION HAPPENING HERE
            # NMS
            # results = non_max_suppression(results, max_det=max_det)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            crops = results.crop(save=False)
            # print(crops)

            for i, crop in enumerate(crops):
                roi = crop['im']
                # cv2.imshow(f'crop {i}', roi)

                # if check_shape(roi):
                #     img = imutils.resize(roi, height=400)
                # else:
                #     img = imutils.resize(roi, width=400)

                """-----------Text------------"""
                # text_result = []
                # result = reader.readtext(new_img, detail=1)
                # for (bbox, text, prob) in result:
                #     text = "".join(
                #         [c if 96 < ord(c) < 123 or 64 < ord(c) < 91 or 58 > ord(c) > 47 else "" for c in text]).strip()
                #     text_result.append(text)

                text_result = xu_ly_text(roi, reader)
                draw_number_licence(frame, text_result, crop['box'])
                """-----------Text------------"""

            plot_boxes(results, frame, classes=classes)

            name_image = img.split('/')[1]
            cv2.imwrite(img_path_save + '/' + name_image, frame)
            print("--------END---------")
        # cv2.namedWindow("img_only", cv2.WINDOW_NORMAL)  ## creating a free windown to show the result
        # cv2.imshow("img_only", frame)
        # cv2.waitKey(0)

    # elif vid_path != None:
    #     print(f"[INFO] Working with video: {vid_path}")
    #
    #     ## reading the video
    #     cap = cv2.VideoCapture(vid_path)
    #
    #     if vid_out:  ### creating the video writer if video output path is given
    #
    #         # by default VideoCapture returns float instead of int
    #         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #         fps = int(cap.get(cv2.CAP_PROP_FPS))
    #         codec = cv2.VideoWriter_fourcc(*'mp4v')  ##(*'XVID')
    #         out = cv2.VideoWriter(vid_out, codec, fps, (width, height))
    #
    #     # assert cap.isOpened()
    #     frame_no = 1
    #
    #     cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
    #     while True:
    #         # start_time = time.time()
    #         ret, frame = cap.read()
    #         if ret:
    #             print(f"[INFO] Working with frame {frame_no} ")
    #
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             results = detectx(frame, model=model)
    #             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #             frame = plot_boxes(results, frame, classes=classes)
    #
    #             cv2.imshow("vid_out", frame)
    #             if vid_out:
    #                 print(f"[INFO] Saving output video. . . ")
    #                 out.write(frame)
    #
    #             if cv2.waitKey(5) & 0xFF == 27:
    #                 break
    #             frame_no += 1
    #
    #     print(f"[INFO] Clening up. . . ")
    #     ### releaseing the writer
    #     out.release()
    #
    #     ## closing all windows
    #     cv2.destroyAllWindows()


### -------------------  calling the main function-------------------------------


# main(vid_path="facemask.mp4",vid_out="facemask_result.mp4") ### for custom video
# main(vid_path=0,vid_out="webcam_facemask_result.mp4") #### for webcam

path_save = "output2"
path_test = "test_anh_thuc"

imgs = []
for root, dir, file in os.walk(path_test):
    for i in file:
        imgs.append(root + "/" + i)

main(imgs=imgs, img_path_save=path_save)  ## for image