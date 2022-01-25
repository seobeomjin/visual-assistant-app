from flask import Flask, render_template,Response 
import cv2 

import numpy as np
import pyttsx3
import time
import argparse

import os



app = Flask(__name__)
cam = cv2.VideoCapture(0)

def numberingObjects(info_dict):
    counter = dict()
    for d in info_dict : 
        if d['Label'] not in counter : 
            counter[d['Label']] = 1 
            d['Label'] = d['Label'] + ' ' + str(1)
        else : 
            counter[d['Label']] += 1 
            d['Label'] = d['Label'] + ' ' + str(counter[d['Label']])
    # print("each object is numbered!")
    return info_dict, counter 

def calculate_object_size(info_dict, image_size):
    (H, W) = image_size
    image_area = H*W
    for item in info_dict : 
        item['object size'] = item['Height']*item['Width']
        item['size ratio'] = round((item['object size']/image_area)*100, 2)
        if item['size ratio'] > 2.0 : 
            item['close'] = True
        else : item['close'] = False
    # print("sizes, ratios and close-or-not information were recored!")
    return info_dict

def calculate_location_info(info_dict, image_size):
    """
    e.g. 
    - At the bottom right side of this scene, there are 2 peoples and 1 cat 
    - At the bottom left side of this secen, 
    """
    (H, W) = image_size # image size = output['image_size] 

    for idx, item in enumerate(info_dict): # item = {'Label' : ~, '': ~ }
        if item['X'] <= W/2 and item['Y'] <= H/2 : # Quadrant 1 # top left # both low
            if item['right_below_x_point'] <=  W/2 and item['right_below_y_point'] <= H/2: 
                item['location'] = 'top left' 
            elif item['right_below_x_point'] <=  W/2 and item['right_below_y_point'] > H/2: 
                item['location'] = 'left'
            else : 
                # middle
                item['location'] = 'middle'
        elif item['X'] > W/2 and item['Y'] > H/2 : # Quadrant 2 # bottom right # both gigh
            if item['right_below_x_point'] >  W/2 and item['right_below_y_point'] > H/2:
                item['location'] = 'bottom right'
            else : 
                # middle 
                item['location'] = 'middle'
        elif item['X'] <= W/2 and item['Y'] > H/2 : # Quadrant 3 # bottom left # small x, large y 
            if item['right_below_x_point'] <=  W/2 and item['right_below_y_point'] > H/2: 
                item['location'] = 'bottom left'
            else :
                # middle 
                item['location'] = 'middle'
        else : # 4 # large x, small y 
            if item['right_below_x_point'] > W/2 and item['right_below_y_point'] <= H/2:
                item['location'] = 'top right'
            elif item['right_below_x_point'] > W/2 and item['right_below_y_point'] > H/2:
                item['location'] = 'right'
            else : 
                # middle
                item['location'] = 'middle'
    
    return info_dict
        

def filteringObjects( info_dict) : 
    # low confidence OR too small image 
    before_filter = len(info_dict)
    for item in info_dict[:] : 
        # if item['confidence'] < args.threshold or item['size ratio'] < 0.1 : 
        if item['confidence'] < 0.53 or item['size ratio'] < 0.1 : 
            info_dict.remove(item)
    # print(f"{before_filter-len(info_dict)} objects were deleted! ({before_filter}->{len(info_dict)})")
    return info_dict

def refined_text(info_dict) : 
    info = []

    for item in info_dict :
        if item['location'] == 'middle' : 
            if item['close'] : 
                info.append(f"{item['Label']} is in front of you. It's close, so be careful.")
            else : 
                info.append(f"{item['Label']} is in front of you.")
        else : 
            if item['close'] : 
                info.append(f"{item['Label']} is on your {item['location']}, It's close, so be careful.")
            else : 
                info.append(f"{item['Label']} is on your {item['location']}")

    info_text = ""
    for sentence in info : 
        info_text += sentence 
        info_text +='\n'

    return info_text
    

def text_to_speech(text, gender):
    """
    Function to convert text to speech
    :param text: text
    :param gender: gender
    :return: None
    """
    voice_dict = {'Male': 0, 'Female': 1}
    code = voice_dict[gender]

    engine = pyttsx3.init()

    # Setting up voice rate
    engine.setProperty('rate', 150) # 125 -> 150

    # Setting up volume level  between 0 and 1
    engine.setProperty('volume', 0.8)

    # Change voices: 0 for male and 1 for female
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[code].id)

    engine.say(text)
    engine.runAndWait()

def gen_frames():
    confidenceThreshold = 0.5
    NMSThreshold = 0.3

    modelConfiguration = 'cfg/yolov3-tiny.cfg' #if args is None else args.cfg
    modelWeights = 'yolov3-tiny.weights' #if args is None else args.model #'yolov3.weights'

    labelsPath = 'coco.names'
    labels = open(labelsPath).read().strip().split('\n')

    np.random.seed(10)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

    outputLayer = net.getLayerNames()
    outputLayer = [outputLayer[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    st_time = time.time()

    video_capture = cv2.VideoCapture(0)
    # camera open check 
    # if not (video_capture.isOpened()):
    #     print("::::::::: WARNING ::::::::: Could not open video device")
    # else : 
    #     print(":::::::: CHEKED :::::::: Camera is opened ! ! ! ")

    # print(f"video capture value : {video_capture}")

    (W, H) = (None, None)

    prev_frame_time = 0
    next_frame_time = 0

    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        # print(f"check frame : {frame}") # value 
        # print(f"ret : {ret}") # True (or False)
        if W is None or H is None:
            (H,W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB = True, crop = False)
        net.setInput(blob)
        layersOutputs = net.forward(outputLayer)

        boxes = []
        confidences = []
        classIDs = []

        for output in layersOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confidenceThreshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY,  width, height) = box.astype('int')
                    x = int(centerX - (width/2))
                    y = int(centerY - (height/2))
                    right_below_x_point = int(centerX + (width/2))
                    right_below_y_point = int(centerY + (height/2))

                    # boxes.append([x, y, int(width), int(height)])
                    boxes.append([x, y, int(width), int(height), int(centerX), int(centerY), right_below_x_point, right_below_y_point])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        #Apply Non Maxima Suppression
        detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)
        if(len(detectionNMS) > 0):
            for i in detectionNMS.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # for text (-> to speech)
        outputs={}

        next_frame_time = time.time()
        fps = 1/(next_frame_time-prev_frame_time)
        prev_frame_time = next_frame_time

        fps = int(fps)

        # print(time.time())
        if int((time.time() - st_time ) % 15) == 0 : 
            # print(f"current FPS : {fps}")
            if(len(detectionNMS) > 0):
                outputs['detections'] = {}
                outputs['detections']['labels'] = []
                outputs['image_size'] = frame.shape[:2] #image.shape[:2]
                for i in detectionNMS.flatten():
                    detection = {}
                    detection['Label'] = labels[classIDs[i]]
                    detection['confidence'] = confidences[i]
                    detection['X'] = boxes[i][0]
                    detection['Y'] = boxes[i][1]
                    detection['Width'] = boxes[i][2]
                    detection['Height'] = boxes[i][3]
                    detection['centerX'] = boxes[i][4]
                    detection['centerY'] = boxes[i][5]
                    detection['right_below_x_point'] = boxes[i][6]
                    detection['right_below_y_point'] = boxes[i][7]
                    outputs['detections']['labels'].append(detection)
            
            else : 
                outputs['detections'] = 'No object detected'
            try : 
                info_dict = list(outputs['detections'].values())[0]
                image_size = outputs['image_size']
                info_dict = calculate_location_info(info_dict, image_size)
                info_dict = calculate_object_size(info_dict, image_size)
                info_dict = filteringObjects(info_dict)
                info_dict, counter = numberingObjects(info_dict)
                info_text = refined_text(info_dict)
            except : 
                info_text = outputs['detections']

            # pyttsx3 
            gender = 'Female'
            text_to_speech(info_text, gender)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            

        # cv2.imshow('Output', frame)
        # if(cv2.waitKey(1) & 0xFF == ord('q')):
        #     break


@app.route('/')
def index():
    return render_template('infer_webcam.html')

@app.route('/video')
def video(): 
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__" : 

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov3-tiny.weights', 
                    help='choose a model which will use')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny.cfg',
                    help='type a ppropriate cfg file for a model')
    parser.add_argument('--threshold', type=float, default='0.6',
                    help='confidence threshold')

    args = parser.parse_args()
    
    # app.run(debug=True)

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)