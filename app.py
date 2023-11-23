from flask import Flask, render_template, request
import numpy as np
import io
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization, Flatten, Activation
import uuid
import os


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

def save_image(image, folder='static', extension='png'):
    """Saves an image to the static folder and returns the file path."""
    filename = f"{uuid.uuid4()}.{extension}"
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, image)
    return filename

# def create_cropped_rotated_images(img):
#     height, width = img.shape[:2]
#     center = (width // 2, height // 2)

#     # Defining crop dimensions (you can adjust these as needed)
#     crop_size = min(width, height) // 2 

#     # Cropping and rotating images
#     images = []
#     for i in range(5):
#         # Cropping
#         start_x = center[0] - crop_size + i * 10  # Adjust cropping logic as needed
#         start_y = center[1] - crop_size + i * 10
#         cropped_img = img[start_y:start_y + crop_size*2, start_x:start_x + crop_size*2]

#         # Rotating
#         rotation_matrix = cv2.getRotationMatrix2D(center, angle=i*10, scale=1)  # Rotate by i*10 degrees
#         rotated_img = cv2.warpAffine(cropped_img, rotation_matrix, (width, height))

#         images.append(rotated_img)

#     return images #returns 5 images.
def create_cropped_rotated_images(img):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    # The zoom factor for each image
    zoom_factors = [1, 1.2, 1.4, 1.6, 1.8]

    # Store the original and transformed images
    images = [img]  # Original image is the first one

    for i, zoom in enumerate(zoom_factors[1:], start=1):  # Skip the first factor as it's for the original image
        # Calculate new crop size based on zoom
        new_crop_size = int(min(width, height) / zoom / 2)

        # Define the crop box
        start_x = center[0] - new_crop_size
        start_y = center[1] - new_crop_size
        cropped_img = img[start_y:start_y + new_crop_size*2, start_x:start_x + new_crop_size*2]

        # Rotating by 90 degrees increments
        rotation_angle = 90 * i
        rotation_matrix = cv2.getRotationMatrix2D((new_crop_size, new_crop_size), rotation_angle, 1)
        rotated_img = cv2.warpAffine(cropped_img, rotation_matrix, (new_crop_size*2, new_crop_size*2))

        # Since rotation can introduce black borders, let's crop it out
        # Calculate the valid area after rotation
        valid_size = int(new_crop_size / (2**0.5))  # Diagonal length reduced after rotation
        start_valid = new_crop_size - valid_size
        valid_img = rotated_img[start_valid:start_valid + valid_size*2, start_valid:start_valid + valid_size*2]

        # Resize back to original size to maintain consistency
        resized_img = cv2.resize(valid_img, (width, height))

        images.append(resized_img)

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['image']
        # Convert the file to an OpenCV image
        in_memory_file = io.BytesIO()
        img_file.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1  # 1 means cv2.IMREAD_COLOR
        img = cv2.imdecode(data, color_image_flag)

        # Process the image and predict
        predictions = predict_val(img)
        final_prediction = get_majority_vote(predictions)

        # Save original and transformed images
        original_img_path = save_image(img)
        transformed_images = [(save_image(image), 'Malignant' if pred[0] == 1 else 'Benign') for image, pred in zip(create_cropped_rotated_images(img), predictions)]

        return render_template('index.html', original_image=original_img_path, transformed_images=transformed_images, final_prediction=final_prediction)

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
# Get the image file from the request
    if request.method == 'POST':
        img_file = request.files['image']

        # Convert the file to an OpenCV image
        in_memory_file = io.BytesIO()
        img_file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # Process image and predict
        predictions = predict_val(img)

        # # Save original and augmented images
        # img_filenames = []
        # cv2.imwrite(os.path.join('static', 'original.jpg'), img)
        # img_filenames.append('original.jpg')
        # for idx, aug_img in enumerate(create_cropped_rotated_images(img)):
        #     filename = f'augmented_{idx}.jpg'
        #     cv2.imwrite(os.path.join('static', filename), aug_img)
        #     img_filenames.append(filename)

        # result = get_majority_vote(predictions)
        # return render_template('index.html', result=result, img_filenames=img_filenames)             #old code stops here
         # Save original and transformed images and record their predictions
        img_info = []
        original_filename = 'original.jpg'
        cv2.imwrite(os.path.join('static', original_filename), img)
        img_info.append({
            'filename': original_filename,
            'prediction': 'Original'
        })
        transformed_images = create_cropped_rotated_images(img)

        for idx, (transformed_img, pred) in enumerate(zip(transformed_images, predictions)):
            filename = f'transformed_{idx}.jpg'
            cv2.imwrite(os.path.join('static', filename), transformed_img)
            img_info.append({
                'filename': filename,
                'prediction': 'Malignant' if pred[0] == 1 else 'Benign'
            })

        final_result = get_majority_vote(predictions)
        return render_template('index.html', final_result=final_result, img_info=img_info)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)