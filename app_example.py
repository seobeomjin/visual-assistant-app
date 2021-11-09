from flask import Flask, jsonify, request 
from yolo_detection_images import detectObjects

app = Flask(__name__)

@app.route('/myapp/detectObjects') 
def detect():
    img = request.args['Image']
    img_path = 'images/' + img 
    results = detectObjects(img_path)
    return jsonify(results)

app.run()

# have to give args to http address 
# http://127.0.0.1:5000/myapp/detectObjects?Image=person.jpg
# it prints out json format output 