from flask import Flask, request, jsonify
from markupsafe import escape
# from PIL import Image
from io import BytesIO
from fastai.vision.all import *

app = Flask(__name__)

@app.route("/")
def index():
  return "<p>Hello, World!</p>"

@app.route("/hello")
def hello():
  return "Good day!"

@app.route("/welcome/<username>")
def welcome(username):
  return f'User {escape(username)}'

@app.route('/post/<int:post_id>')
def show_post(post_id):
  return f'Post {post_id}'

categories = ('American Robin', 'Downy Woodpecker', 'Great Blue Heron',  'Hairy Woodpecker', 'House Sparrow')
def classify_img(img):
  learn = load_learner("model.pkl")
  pred,idx,probs = learn.predict(img)
  return dict(zip(categories, map(float,probs)))

@app.post('/predict-bird-image')
def predictBirdImage():
  print(request)
  print(request.form)
  print(request.form['yoyo'])
  print(request.files)
  print(request.files['image'])

  if 'image' not in request.files:
    return jsonify({"error": "No image file provided"}), 400
  
  image_file = request.files['image']

  try:
    # img = Image.open(BytesIO(image_file.read()))
    img = PILImage.create(BytesIO(image_file.read()))
    width, height = img.size
    result = classify_img(img)
    print(result)
    return jsonify(result)
  
  except Exception as e:
    return jsonify({"error": str(e)}), 500
