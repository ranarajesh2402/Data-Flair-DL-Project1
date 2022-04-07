import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:30]  # Top 30 results
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")


# CODE REFERENCE: https://github.com/matsui528/sis
#                 Video: https://www.youtube.com/watch?v=M0Y9_vBmYXU&ab_channel=YusukeMatsui


# STEPS:

# 1. Put some image( like 50 to 60 images) into the 'static/img/' folder.

# 2. Open Visual Studio Code in '8. Image Reverse Search Engine' folder.

# 3. Then, run the 'offline.py' as it'll create features in 'static/features/' folder with '.npy' extention.

# 4. Run the 'server.py'. Where in OUTPUT it'll tell you to click on the link. 

#    4.a. Copy the link and paste in any browser(eg., Google Chrome, Micosoft Edge, etc.)

#	 4.b. Choose the image by doble clicking the image from 'static/img/' folder. Which will be stored in 'static/uploaded/' folder.

#	 4.c. Then click on submit.

#	 5.d. It'll show you top 30 images which are similar to the uploaded image.
#		 5.d.i)  It'll show the similar images with the score (at which it's closest).
#		 5.d.ii) It'll only show only those images that are available in 'static/img/' folder.