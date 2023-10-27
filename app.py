from flask import Flask, render_template, request, send_from_directory
import wget
import cv2
import os
import numpy as np


sharpening_kernel = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['RESULT_FOLDER'] = 'static/images/results'


@app.route('/')
def index():
    original_image = None
    modified_image = None
    return render_template('index.html', original_image=original_image, modified_image=modified_image)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file part"
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file"
    
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        modified_image = None

        return render_template('index.html', original_image=filename, modified_image=modified_image)
    





# Filter ------

def apply_filter(original_image, filter_type):
    image = cv2.imread(original_image)

    if filter_type == 'sepia':
        # Apply Sepia filter
        filtered_image = apply_sepia(image)
    elif filter_type == 'invert':
        # Apply Invert filter
        filtered_image = apply_invert(image)
    elif filter_type == 'grayscale':
        # Convert to grayscale
        filtered_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
    else:
        return "Invalid filter type"

    result_filename = os.path.join(app.config['RESULT_FOLDER'], 'filtered_image.jpg')
    cv2.imwrite(result_filename, filtered_image)
    return result_filename

def apply_sepia(image):
    
    from PIL import Image, ImageOps
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    sepia_image = ImageOps.colorize(image_pil.convert('L'), "#704214", "#C0A080")
    return cv2.cvtColor(np.array(sepia_image), cv2.COLOR_RGB2BGR)

def apply_invert(image):
    # Invert the colors
    inverted_image = cv2.bitwise_not(image)
    return inverted_image

#Image Enhancement -------

@app.route('/enhance', methods=['POST'])
def enhance_image():
    original_image = request.form['original_image']
    operation = request.form['operation']

    image = cv2.imread(original_image)

    if operation == 'reduce_noise':
        enhanced_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    elif operation == 'enhance_sharpness':
        enhanced_image = cv2.filter2D(image, -1, sharpening_kernel)
    else:
        return "Invalid operation"

    result_filename = os.path.join(app.config['RESULT_FOLDER'], 'enhanced_image.jpg')
    cv2.imwrite(result_filename, enhanced_image)
    return render_template('index.html', original_image=original_image, modified_image=result_filename)





 # Image Smoothing ----------    
    
@app.route('/smooth', methods=['POST'])
def smooth_image():
    original_image = request.form['original_image']
    smoothing_type = request.form['smoothing_type']

    image = cv2.imread(original_image)

    if smoothing_type == 'average':
        # Apply average smoothing
        smoothed_image = cv2.blur(image, (5, 5))
    elif smoothing_type == 'gaussian':
        # Apply Gaussian smoothing
        smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)
    elif smoothing_type == 'median':
        # Apply median smoothing
        smoothed_image = cv2.medianBlur(image, 5)
    else:
        return "Invalid smoothing type"

    result_filename = os.path.join(app.config['RESULT_FOLDER'], 'smoothed_image.jpg')
    cv2.imwrite(result_filename, smoothed_image)
    return render_template('index.html', original_image=original_image, modified_image=result_filename)
    



@app.route('/process', methods=['POST'])
def process_image():
    original_image = request.form['original_image']
    filter_type = request.form['filter']  # Get the selected filter

    modified_image = apply_image_operations(original_image)
    
    if filter_type != 'filter':
        modified_image = apply_filter(modified_image, filter_type)

    return render_template('index.html', original_image=original_image, modified_image=modified_image)




def apply_image_operations(original_image):
    color = request.form['color']
    rotation = int(request.form['rotation'])
    left = int(request.form['left'])
    top = int(request.form['top'])
    right = int(request.form['right'])
    bottom = int(request.form['bottom'])
    flip = request.form['flip']

    image = cv2.imread(original_image)

    if color == 'bw':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif color == 'grayscale':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if rotation > 0:
        image = rotate_image(image, rotation)

    if left > 0 or top > 0 or right > 0 or bottom > 0:
        image = crop_image(image, left, top, right, bottom)

    if flip == 'horizontal':
        image = cv2.flip(image, 1)
    elif flip == 'vertical':
        image = cv2.flip(image, 0)

    result_filename = os.path.join(app.config['RESULT_FOLDER'], 'modified_image.jpg')
    cv2.imwrite(result_filename, image)
    return result_filename

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def crop_image(image, left, top, right, bottom):
    return image[top:image.shape[0]-bottom, left:image.shape[1]-right]

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['RESULT_FOLDER']):
        os.makedirs(app.config['RESULT_FOLDER'])
    app.run(debug=True)