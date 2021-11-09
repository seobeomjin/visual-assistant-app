from flask import Flask, request, jsonify, render_template
# from torch_utils import transform_imgae, get_prediction
from yolo_detection_images import *
from PIL import Image
import numpy as np
import io
from base64 import b64encode
import cv2
import re 
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about(): 
    return render_template('about.html')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}
def allowed_file(filename):
    # xxx.png .jpg . jpeg
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    # rsplit('sep',maxsplit) 
    # rsplit은 오른쪽부터 나눔, 이건 사실 maxsplit을 쓰지 않으면 그냥 split과 매우 유사 
    # maxsplit은 sep를 기준으로 sep가 지나고나서 몇개가 남기를 원하는지를 묻는 것. 
    # 그래도 결과의 sep된 string은 왼쪽에서 오른쪽 순으로 나열됨.  

@app.route('/infer', methods=['POST'])
def predict():
    if request.method == 'POST': 
        f = request.files['file']
        if f is None or f.filename == "": 
            return jsonify({'error':'no file'})
        if not allowed_file(f.filename):
            return jsonify({'error':'format not supported'})
        
        # saveLocation = f.filename 
        # f.save(saveLocation)

        try : 
            img_bytes = f.read()
            # image = np.array(Image.open(io.BytesIO(img_bytes)))  #method1 binary -> numpy
            encoded_img = np.fromstring(img_bytes, dtype = np.uint8) #method2 bin->numpy
            image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR) # 1D np array -> 3D np array
            output, image = detectObjects(image, True)

            print(f"I wanna know image type : {type(image)}")

            info_dict = list(output['detections'].values())[0]
            image_size = output['image_size']

            info_dict = calculate_location_info(info_dict, image_size)
            info_dict = calculate_object_size(info_dict, image_size)
            info_dict = filteringObjects(info_dict)
            info_dict, counter = numberingObjects(info_dict)
            info_text = refined_text(info_dict)

            # to load an inferred image
            img = Image.fromarray(image.astype("uint8"))
            rawBytes = io.BytesIO()
            img.save(rawBytes, "JPEG")
            rawBytes.seek(0)
            img_base64 = b64encode(rawBytes.getvalue()).decode('ascii')
            mime = "image/jpeg"
            uri = "data:%s;base64,%s"%(mime, img_base64)

            print(f"passed here")

            # TTS 
            ###################### I have to fix this
            # gender = request.form['voices']
            # text_to_speech(info_text, gender)

            # make an output more informative
            # info_str = re.sub('\n','<br>',info_str)

            return render_template('inference.html',image=uri ,output=info_text)
        
        except Exception as e :
            return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True) 
    # app.debug = True
    # port = int(os.environ.get("PORT", 80))
    # app.run(host='0.0.0.0', port=port, debug=True)
    # app.run(debug=True, host='127.0.0.1', port=5000)


# @app.route('/predict',methods=['POST'])
# def predict():  # for predict 
#     if request.method == "POST":
#         file = request.files.get('file')
#         # later we pass a file to this post request and get this file
#         if file is None or file.filename=="" : 
#             return jsonify({'error':'no file'})
#         if not allowed_file(file.filename):
#             return jsonify({'error':'format not supported'})

#         try : 
#             img_bytes = file.read()
#             tensor = transform_imgae(img_bytes)
#             prediction = get_prediction(tensor)
#             data = {'prediction':prediction.item(), 'class_name':str(prediction.item())}
#             return jsonify(data)        
        
#         except:
#             return jsonify({'error':'error during prediction'})



    # 1 load image

    # 2 image -> tensor 
    # 3 prediction 
    # 4 return json dat 
    # but I want to return to img 
    # return jsonify({'result': 1})
    
    # terminal open key : ctrl j 
    # for do
    # set FLASK_APP=app.py
    # flask run 




# heroku 주요 명령어 
# https://www.a-mean-blog.com/ko/blog/%EB%8B%A8%ED%8E%B8%EA%B0%95%EC%A2%8C/_/Heroku-%ED%97%A4%EB%A1%9C%EC%BF%A0-%EA%B0%80%EC%9E%85-Heroku-CLI-%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C-%EA%B0%84%EB%8B%A8-%EC%82%AC%EC%9A%A9%EB%B2%95

