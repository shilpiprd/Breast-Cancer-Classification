from flask import Flask, render_template, request
import numpy as np
import io
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization, Flatten, Activation

app = Flask(__name__)

def create_model(): 
    #defining my_model architecture 
    base_model = ResNet50(input_shape=(100,100,3), weights='imagenet', include_top=False)
    my_model=Sequential()
    my_model.add(base_model)
    my_model.add(Dropout(0.2))
    my_model.add(Flatten())
    my_model.add(BatchNormalization())
    my_model.add(Dense(1024, kernel_regularizer=l2(0.001)))
    my_model.add(BatchNormalization())
    my_model.add(Activation('relu'))
    my_model.add(Dropout(0.2))
    my_model.add(Dense(1024,kernel_regularizer=l2(0.001)))
    my_model.add(BatchNormalization())
    my_model.add(Activation('relu'))
    my_model.add(Dropout(0.2))
    my_model.add(Dense(1024,kernel_regularizer= l2(0.001)))
    my_model.add(BatchNormalization())
    my_model.add(Activation('relu'))
    my_model.add(Dropout(0.2))
    my_model.add(Dense(1024,kernel_regularizer=l2(0.001)))
    my_model.add(BatchNormalization())
    my_model.add(Activation('relu'))
    my_model.add(Dropout(0.2))
    my_model.add(Dense(1, activation = 'sigmoid'))
    my_model.layers[0].trainable = False
    return my_model

# Load your model architecture and weights
my_new_model = create_model()  # Function to create your model architecture
my_new_model.load_weights('balanced_cancer_classification_weights.h5')


def create_cropped_rotated_images(img):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    # Defining crop dimensions (you can adjust these as needed)
    crop_size = min(width, height) // 2 

    # Cropping and rotating images
    images = []
    for i in range(5):
        # Cropping
        start_x = center[0] - crop_size + i * 10  # Adjust cropping logic as needed
        start_y = center[1] - crop_size + i * 10
        cropped_img = img[start_y:start_y + crop_size*2, start_x:start_x + crop_size*2]

        # Rotating
        rotation_matrix = cv2.getRotationMatrix2D(center, angle=i*10, scale=1)  # Rotate by i*10 degrees
        rotated_img = cv2.warpAffine(cropped_img, rotation_matrix, (width, height))

        images.append(rotated_img)

    return images

def predict_val(img):
    augmented_images = create_cropped_rotated_images(img)
    predictions = []

    for image in augmented_images:
        image_resized = cv2.resize(image, (100, 100))
        if len(image_resized.shape) == 2:
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
        image_batch = np.expand_dims(image_resized, axis=0)
        y_pred_temp = my_new_model.predict(image_batch)
        threshold = 0.2 
        prediction = (y_pred_temp > threshold).astype(int)
        predictions.append(prediction[0])

    return predictions

def get_majority_vote(predictions):
    count_malignant = sum(p == 1 for p in predictions)
    return 'malignant' if count_malignant > len(predictions) // 2 else 'benign'

    
@app.route('/')
def index(): 
    return render_template('index.html')
    
  
@app.route('/predict', methods=['POST'])
def predict():
# Get the image file from the request
    if request.method == 'POST':
        img_file = request.files['image']
        # Convert the file to an OpenCV image
        in_memory_file = io.BytesIO()
        img_file.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1  # 1 means cv2.IMREAD_COLOR
        img = cv2.imdecode(data, color_image_flag)
        predictions = predict_val(img)
        result = get_majority_vote(predictions)
    else:
        result = None

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

