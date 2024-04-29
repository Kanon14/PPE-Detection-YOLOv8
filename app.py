import sys, os
import cv2
from ppeDetection.pipeline.training_pipeline import TrainPipeline
from ppeDetection.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
from ppeDetection.constant.application import APP_HOST, APP_PORT


app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        

@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training Successfully!!"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)
        
        os.system(f"cd yolov8l_train && yolo task=detect mode=predict \
                    model='best.pt' \
                    imgsz=640 \
                    source='../data/inputImage.jpg' \
                    conf=0.5")
        
        opencodedbase64 = encodeImageIntoBase64("yolov8l_train/runs/detect/predict/inputImage.jpg")
        result = {"image": opencodedbase64.decode('utf-8')}
        
        os.system("rm -rf yolov8l_train/runs")
        
    except ValueError as val:
        print(val)
        return Response("Value not found inside json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        return Response("Something went wrong")
    
    return jsonify(result)
        
  
if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host=APP_HOST, port=APP_PORT, debug=True)