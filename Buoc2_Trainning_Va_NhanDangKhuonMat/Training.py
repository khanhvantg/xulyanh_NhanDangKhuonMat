# Nguyễn Đặng Khánh Văn - 19110312
# Hoàng Quốc Việt       - 19110315
# Nguồn tham khảo: Code thầy gửi trên trang dạy học số
from model import create_model
import numpy as np
import os.path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib

from align import AlignDlib
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.manifold import TSNE


class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)


def load_metadata(path):
    metadata = []
    for i in sorted(os.listdir(path)):
        for f in sorted(os.listdir(os.path.join(path, i))):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg' or ext == '.bmp':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[..., ::-1]


def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))



nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('Buoc2_Trainning_Va_NhanDangKhuonMat/weights/nn4.small2.v1.h5')


metadata = load_metadata('image')

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('Buoc2_Trainning_Va_NhanDangKhuonMat/models/shape_predictor_68_face_landmarks.dat')

# Load an image of a person
jc_orig = load_image(metadata[77].image_path())

# Detect face and return bounding box
bb = alignment.getLargestFaceBoundingBox(jc_orig)

# Transform image using specified face landmark indices and crop image to 96x96
jc_aligned = alignment.align(
    96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


embedded = np.zeros((metadata.shape[0], 128))
TTD_size = metadata.shape[0]

dem = 0

for i, m in enumerate(metadata):
    print(m.image_path())
    img = load_image(m.image_path())
    img = align_image(img)
    if img is not None:
        # scale RGB values to interval [0,1]
        img = (img / 255.).astype(np.float32)
        # obtain embedding vector for image
        embedded[i] = nn4_small2_pretrained.predict(
            np.expand_dims(img, axis=0))[0]
        dem = dem + 1

if dem < TTD_size:
    while True:
        mm, nn = embedded.shape
        flagThoat = True
        for i in range(0, mm):
            if np.sum(embedded[i]) == 0:
                embedded = np.delete(embedded, i, 0)
                metadata = np.delete(metadata, i, 0)
                flagThoat = False
                break
        if flagThoat == True:
            break


targets = np.array([m.name for m in metadata])

encoder = LabelEncoder()
encoder.fit(targets)

# Numerical encoding of identities
y = encoder.transform(targets)

train_idx = np.arange(metadata.shape[0]) % 2 != 0
test_idx = np.arange(metadata.shape[0]) % 2 == 0

# 50 train examples of 10 identities (5 examples each)
X_train = embedded[train_idx]
# 50 test examples of 10 identities (5 examples each)
X_test = embedded[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
svc = LinearSVC()

knn.fit(X_train, y_train)
svc.fit(X_train, y_train)


# Kết quả sẽ tạo ra model svc.pkl
joblib.dump(svc, 'Buoc2_Trainning_Va_NhanDangKhuonMat/svc.pkl')
