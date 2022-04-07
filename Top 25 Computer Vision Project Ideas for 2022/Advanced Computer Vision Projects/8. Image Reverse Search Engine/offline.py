from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    fe = FeatureExtractor()

    for img_path in sorted(Path("./static/img").glob("*.jpg")):
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path))
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)


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