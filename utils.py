import cv2
from deepface import DeepFace


def analyze_face(face_img):
    analysis = DeepFace.analyze(
        face_img, actions=['age', 'gender', 'emotion'], enforce_detection=False
    )[0]

    age = int(analysis['age'])

    # Get dominant gender
    gender_probs = analysis['gender']
    dominant_gender = max(gender_probs, key=gender_probs.get)

    # Get dominant emotion
    dominant_emotion = analysis['dominant_emotion']

    return age, dominant_gender, dominant_emotion


def draw_predictions(image, face_info, age, gender, emotion):
    x, y, w, h = face_info['box']
    label = f"{gender}, {age} yrs, {emotion}"

    img_h, img_w = image.shape[:2]

    # Find appropriate font_scale so that the text width matches the face box width
    font_scale = 1.0
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # Loop to adjust font_scale
    target_w = w - 10  # Slightly smaller than the box for aesthetics
    while text_w < target_w and font_scale < 10:
        font_scale += 0.1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    while text_w > target_w and font_scale > 0.1:
        font_scale -= 0.01
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    font_scale = max(0.1, font_scale)
    thickness = max(1, int(font_scale * 2))

    # Text coordinates – always above the face
    text_x = x
    text_y = y - 10
    if text_y - text_h < 0:
        text_y = y + h + text_h + 10
    if text_x + text_w > img_w:
        text_x = img_w - text_w - 10

    # Green background
    cv2.rectangle(image, (text_x, text_y - text_h - baseline),
                  (text_x + text_w + 10, text_y), (0, 255, 0), -1)

    # White text with thickness
    cv2.putText(image, label, (text_x + 5, text_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    # Face bounding box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image
