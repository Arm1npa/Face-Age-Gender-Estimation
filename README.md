# Face Age, Gender & Emotion Detection

A powerful web application built with Streamlit that uses DeepFace to detect and analyze faces in images. The application can identify age, gender, and emotions of people in uploaded images with high accuracy.

## Features

- 📸 Upload and analyze images
- 👤 Face detection using OpenCV
- 🔍 Multiple analysis capabilities:
  - Age estimation
  - Gender detection
  - Emotion recognition
- 🎨 Clean and user-friendly interface
- ⚡ Real-time processing
- 🖼️ Beautiful visualization with bounding boxes and labels

## Technologies Used

- Python
- Streamlit - For the web interface
- DeepFace - For face analysis
- OpenCV - For image processing and face detection
- PIL (Python Imaging Library) - For image handling
- NumPy - For numerical operations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Arm1npa/Face-Age-Gender-Estimation.git
cd Face-Age-Gender-Estimation.git
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Upload an image containing faces (supported formats: JPG, JPEG, PNG)

4. Click "Analyze Image" to process the image

5. View the results showing detected faces with their:
   - Age
   - Gender
   - Dominant emotion

## How It Works

The application uses a sophisticated pipeline for face analysis:

1. **Face Detection**:
   - Uses OpenCV for initial face detection
   - Supports multiple faces in a single image

2. **Face Analysis**:
   - Age estimation using DeepFace
   - Gender classification with confidence scores
   - Emotion recognition (identifies the dominant emotion)

3. **Visualization**:
   - Draws bounding boxes around detected faces
   - Displays age, gender, and emotion information
   - Automatically adjusts text size for better readability
   - Uses green bounding boxes and white text for clear visibility

## Project Structure

```
├── app.py           # Main Streamlit application
├── utils.py         # Utility functions for face analysis
├── requirements.txt # Project dependencies
└── README.md        # Project documentation
```

## Notes

- The first run might take longer as it loads the necessary models
- For best results, use clear images with well-lit faces
- The application supports JPG, JPEG, and PNG image formats
- Face detection works best with front-facing images
- The application can handle multiple faces in a single image

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 