import io
from re import I
import flask
import requests
import os
import cv2
import numpy as np
import json
from PIL import Image
import base64

import gradcam

app = flask.Flask(__name__)
print(flask)
IMG_PATH = os.path.join("static", "image.jpg")

BASE = "http://tf-serve:8501/v1/models/dfu-maceration"


def process_image(image):
    """
    Takes the image file from the `image` parameter and returns an numpy array

    """
    image = Image.open(io.BytesIO(image))
    img_wd, img_hg = image.size
    img = np.asarray(image)

    app.logger.info("transformed image to np array")

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = img[np.newaxis, ...].astype(np.float32)

    return img


@app.route("/", methods=["GET"])
def index():
    return {"data": ["Here we serve the ZIEL maceration model."]}


"""
MODEL STATUS
"""


@app.route("/model/status", methods=["GET"])
def model_status():
    r = requests.get(url=BASE)
    return r.text


@app.route("/model/info", methods=["GET"])
def model_metainfo():
    r = requests.get(url=BASE + "/metadata")
    return r.text


"""
MACERATION MODEL 
"""


@app.route("/image/predict", methods=["POST"])
def image_test():

    app.logger.info("Starting prediction.")
    data = {"success": False}

    image = flask.request.files["image"].read()
    img = process_image(image)
    img = img.tolist()

    print(type(img))

    endpoint = "http://tf-serve:8501/v1/models/dfu-maceration:predict"

    json_data = {"instances": img}

    response = requests.post(endpoint, json=json_data)

    prediction = response.json()["predictions"]
    prediction_label = (
        "Maceration Present" if prediction[0][0] > 0.5 else "Maceration Absent"
    )

    data = {
        "inference": {"probability": prediction[0][0], "label": prediction_label},
    }

    return flask.jsonify(data)


@app.route("/image/gradcam", methods=["POST"])
def gradcam_maceration():

    r = {"success": True}

    # Read and process image from request
    raw_image = flask.request.files["image"].read()
    app.logger.info("Type raw image: " + str(type(raw_image)))

    img = process_image(raw_image)
    image = img.tolist()

    # Transfor to np array
    # Gradient Cam Function require an np array
    image = np.array(image)

    # get the last layer name
    last_layer_name = "conv_pw_13"
    model = gradcam.build_model()
    predicted_class_idx = 0

    # conduct the gradient cam and compute the heatmap
    icam = gradcam.GradCAM(model, predicted_class_idx, last_layer_name)
    heatmap = icam.compute_heatmap(image)
    app.logger.info("heatmap created")

    # Combine the original image (resize) with the heatmap
    heatmap, output = icam.overlay_heatmap(heatmap, img[-1], alpha=0.5)
    cv2.imwrite("gradient-02.jpg", output)

    r["size"] = output.shape

    image = output.copy(order="C")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, im_arr = cv2.imencode(
        ".jpg", image
    )  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes).decode("utf-8")
    r["image"] = im_b64

    return json.dumps(r)


@app.route("/image/info", methods=["POST", "GET"])
def image_info():

    if flask.request.method == "GET":
        return {"message": "no GET method available"}
    else:
        raw_image = flask.request.files["image"].read()
        img = process_image(raw_image)

        r = {
            "shape": [img.shape],
            "range": {"min": float(np.min(img)), "max": float(np.max(img))},
            "unit": "px",
        }

        r = json.dumps(r)
        return r


"""
Leila, from here on you can start and define the endpoints

@app.route("/classfication/prediction", methods=['POST'])
def fun_name1():
    pass

@app.route("/classfication/gradcam", methods=['POST'])
def fun_name2():
    pass

"""

if __name__ == "__main__":
    print("* Starting web service...")
    app.run(host="0.0.0.0", debug=True)
