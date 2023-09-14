# Real-time Face Detection

This repository contains a real-time face detection system implemented in a Jupyter notebook. **The model was trained using images captured from a webcam, and the dataset was augmented using the 
*Albumentations* library. The entire dataset, consisting of approximately 5000 images, was meticulously curated by me.** The dataset was used for training, testing, and validation. Additionally, the initial images 
were annotated using the ***LabelMe*** library.

## Model Architecture

The face detection algorithm consists of two models:

1. **Classification Model**: This model is responsible for detecting whether a face is present in the image or not. It utilizes Binary Cross Entropy as the loss function.

2. **Regression Model**: This model is responsible for predicting the coordinates needed to draw a bounding box around the detected face. It employs Localization loss as the loss function.

The VGG16 model was used to implement these models, resulting in two lists of outputs. The first list contains a single value representing the classification probability of the face, while the second list 
contains coordinates for drawing the bounding box.

## Usage

To use this real-time face detection system, follow these steps:

1. **Clone this repository:**
   ```bash
   git clone https://github.com/a-mythh/Face-Detection.git
   ```

2. **Download required libraries**:
   ```bash
   pip install pandas numpy tensorflow opencv-python
   ```

3. **Execute the App File**:
   ```bash
   python face_tracker_app.py
   ```

The model if to be used is saved in the `.h5` file.

## Dataset Augmentation and Annotation

The dataset was augmented using the Albumentations library to enhance diversity in the training data. Additionally, the initial images were annotated using the LabelMe library to provide ground truth labels 
for the training process.

## Training and Testing

The model was trained on the augmented dataset, and performance was evaluated on the test set to ensure accurate face detection.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to reach out to with any questions or suggestions regarding this project.

Made by Amit Das.

[LinkedIn](https://www.linkedin.com/in/amit-das-work/) | [GitHub](https://github.com/a-mythh)
