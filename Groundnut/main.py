import streamlit as st
import cv2
import numpy as np
from tensorflow.keras import layers, models

# Constants
img_height = 64
img_width = 64 

# Define the model architecture
def create_crop_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))  # Adjust to 3 output neurons for 3 categories
    return model

# Load pre-trained model weights
model = create_crop_model()
model.load_weights('crop_classification_model.h5')

# Function to predict crop maturity
def predict_crop_maturity(img):
    img = cv2.resize(img, (img_height, img_width))
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    return prediction[0]

# Function to analyze crop maturity based on prediction
def analyze_crop_maturity(prediction):
    categories = ['mature', 'half_mature', 'immature']
    predicted_index = np.argmax(prediction)
    predicted_probability = prediction[predicted_index]
    
    # Print raw prediction probabilities for debugging
    print("Raw Prediction Probabilities:")
    for i, prob in enumerate(prediction):
        print(f"{categories[i]}: {prob}")
    
    predicted_category = categories[predicted_index]
    return f"The crop is predicted to be {predicted_category} with probability {predicted_probability:.4f}."


# Streamlit app
def main():
    st.title("Crop Maturity Analysis")

    choice = st.sidebar.selectbox("Choose an option", ["Upload Image", "Exit"])

    if choice == "Upload Image":
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            prediction = predict_crop_maturity(image)
            maturity_analysis = analyze_crop_maturity(prediction)
            
            st.subheader("Analysis Result:")
            st.write(maturity_analysis)

    elif choice == "Exit":
        st.write("Thank you for using the Crop Maturity Analysis App!")

if __name__ == "__main__":
    main()
