
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
        if item['X'] <= W/2 and item['Y'] > H/2 : # Quadrant 1 # top left
            if item['right_below_x_point'] <=  W/2 and item['right_below_y_point'] > H/2: 
                # top left 
                item['location'] = 'top left'
                # info.append(f"{idx+1 }. {item['Label']} is located on your top left side")
            else : 
                # middle
                item['location'] = 'middle'
                # info.append(f"{idx+1 }. {item['Label']} is located in front of you")
        elif item['X'] > W/2 and item['Y'] > H/2 : # Quadrant 2 # top right 
            if item['X'] >  W/2 and item['Y'] > H/2:
                # top right 
                item['location'] = 'top right'
                # info.append(f"{idx+1}. {item['Label']} is located on your top right side")
            else : 
                # middle 
                item['location'] = 'middle'
                # info.append(f"{idx+1 }. {item['Label']} is located in front of you")
        elif item['X'] <= W/2 and item['Y'] <= H/2 : # Quadrant 3 # bottom left 
            if item['right_below_x_point'] <=  W/2 and item['right_below_y_point'] <= H/2: 
                # bottom left 
                item['location'] = 'bottom left'
                # info.append(f"{idx+1 }. {item['Label']} is located on your bottom left side")
            else :
                # middle 
                item['location'] = 'middle'
                # info.append(f"{idx+1 }. {item['Label']} is located in front of you")
        else : # 4 #bottom right
            if item['X'] > W/2 and item['Y'] <= H/2:
                # bottom right 
                item['location'] = 'bottom right'
                # info.append(f"{idx+1 }. {item['Label']} is located on your bottom right side")
            else : 
                # middle
                item['location'] = 'middle'
                # info.append(f"{idx+1 }. {item['Label']} is located in front of you")
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
    return None

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
    {'Label': 'car1', 'confidence': 0.998502790927887, 'X': 101, 'Y': 555, 'Width': 409, 'Height': 253, 'centerX': 306, 
    'centerY': 682, 'right_below_x_point': 510, 'right_below_y_point': 555, 
    'location': 'top left', 'object size': 103477, 'size ratio': 8.42, 'close': True}

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