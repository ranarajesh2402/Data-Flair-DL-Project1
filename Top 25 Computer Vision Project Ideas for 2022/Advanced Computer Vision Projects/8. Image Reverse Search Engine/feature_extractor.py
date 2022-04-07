from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np

# See https://keras.io/api/applications/ for details

class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)

        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # Normalize
    
    
    
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

