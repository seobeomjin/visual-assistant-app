import numpy as np
import cv2

import pyttsx3
# from gtts import gTTS
import os 

import argparse

def detectObjects(img_path, from_html=False) : 
    confidenceThreshold = 0.5
    NMSThreshold = 0.3

    modelConfiguration = 'cfg/yolov3.cfg'
    modelWeights = 'yolov3.weights'

    labelsPath = 'coco.names'
    labels = open(labelsPath).read().strip().split('\n')

    np.random.seed(10)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

    if from_html : 
        image = img_path
    else :
        image = cv2.imread(img_path)
    
    (H, W) = image.shape[:2]
    # print(f"image H and W : {H}, {W}")

    #Determine output layer names
    layerName = net.getLayerNames()
    layerName = [layerName[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False)
    net.setInput(blob)
    layersOutputs = net.forward(layerName)

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

                boxes.append([x, y, int(width), int(height), int(centerX), int(centerY), right_below_x_point, right_below_y_point])
                confidences.append(float(confidence))
                classIDs.append(classID)

    #Apply Non Maxima Suppression
    detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)

    if(len(detectionNMS) > 0):
        for i in detectionNMS.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            (centerX, centerY) = (boxes[i][4], boxes[i][5])
            (right_below_x_point, right_below_y_point) = (boxes[i][6], boxes[i][7])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # print(f"image format : {type(image)}") # <class 'numpy.ndarray'>
    outputs={}

    if(len(detectionNMS) > 0):
        outputs['detections'] = {}
        outputs['detections']['labels'] = []
        outputs['image_size'] = image.shape[:2]
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

    return outputs, image


def numberingObjects(info_dict):
    counter = dict()
    for d in info_dict : 
        if d['Label'] not in counter : 
            counter[d['Label']] = 1 
            d['Label'] = d['Label'] + ' ' + str(1)
        else : 
            counter[d['Label']] += 1 
            d['Label'] = d['Label'] + ' ' + str(counter[d['Label']])
    print("each object is numbered!")
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
    print("sizes, ratios and close-or-not information were recored!")
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
    # info_str = ""
    # for item in info : 
    #     info_str += item
    #     info_str += '\n'
    # print(info_str)
    print("location information is recorded!")
    # print(info_dict)
    return info_dict
        # image size : 960, 1280
        # {'Label': 'person', 'confidence': 0.9995384812355042, 'X': 996, 'Y': 374, 'Width': 186, 'Height': 493}
    # return None

def filteringObjects(info_dict) : 
    # low confidence OR too small image 
    before_filter = len(info_dict)
    for item in info_dict[:] : 
        if item['confidence'] < 0.6 or item['size ratio'] < 0.1 : 
            info_dict.remove(item)
    print(f"{before_filter-len(info_dict)} objects were deleted! ({before_filter}->{len(info_dict)})")
    return info_dict

def refined_text(info_dict) : 
    info = []
    # for idx, item in enumerate(info_dict): # item = {'Label' : ~, '': ~ 
    # {'Label': 'car1', 'confidence': 0.998502790927887, 'X': 101, 'Y': 555, 'Width': 409, 'Height': 253, 'centerX': 306, 
    # 'centerY': 682, 'right_below_x_point': 510, 'right_below_y_point': 555, 
    # 'location': 'top left', 'object size': 103477, 'size ratio': 8.42, 'close': True}

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


if __name__== "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='images/street_2.jpg', 
                    help='img path')
    args = parser.parse_args()

    img_path = args.path #'images/sample0.jpg'
    output, image = detectObjects(img_path)

    info_dict = list(output['detections'].values())[0]
    image_size = output['image_size']
    # print(f"image area  : {image_size[0]*image_size[1]}")
    
    # 1. object location - ☑
    # 2. object size based close object alert(size ratio 2.0 <) - 
    # 3. object numbering - ☑
    # 4. low confidence filtering (0.6 <) - 
    # 5. small size filtering (size ratio 0.1 <) - ☑
    
    info_dict = calculate_location_info(info_dict, image_size)
    info_dict = calculate_object_size(info_dict, image_size)
    info_dict = filteringObjects(info_dict)
    info_dict, counter = numberingObjects(info_dict)
    for item in info_dict : 
        print(item)
    info_text = refined_text(info_dict)

    print(f"image size : {image_size}")
    print(info_text)

    # pyttsx3 # Error
    gender = 'Female'
    text_to_speech(info_text, gender)

    #gtts # OK
    # tts = gTTS(text=info_text, lang='en')
    # tts.save("test.mp3")
    # print(f"mp3 file is saved!")

    # os.system("start test.mp3")

    # for item in info_dict : 
    #     print(item)
        
    cv2.imwrite('image_output.jpg', image)
    # cv2.imshow('Image', image)



