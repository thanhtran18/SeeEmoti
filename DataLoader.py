from os.path import join
import numpy as np
import Constants as Constants
from sklearn.model_selection import train_test_split

class DataLoader:
    def load_data(self):
        images = np.load(join(Constants.DATA_DIR, Constants.DATA_IMAGE_FILE))
        images = images.reshape([-1, Constants.FACE_SIZE, Constants.FACE_SIZE, -1])
        labels = np.load(join(Constants.DATA_DIR, Constants.DATA_LABEL_FILE)).reshape([-1, len(Constants.EMOTIONS)])
        return train_test_split(images, labels, test_size=0.20, random_state=42)
