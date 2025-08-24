import cv2
import numpy as np

# --- 1. Model Paths ---
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
FACE_PROTO = "deploy.prototxt.txt"
AGE_MODEL = "age_net.caffemodel"
AGE_PROTO = "age_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# --- 2. Load Models ---
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

def recognize_age_gender(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return

    h, w = image.shape[:2]

    # Face Detection Blob
    face_blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                      (104.0, 177.0, 123.0),
                                      swapRB=False, crop=False)
    face_net.setInput(face_blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Clamp to image size
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w-1, endX), min(h-1, endY)

            face_roi = image[startY:endY, startX:endX]
            if face_roi.size == 0:  # Skip empty crops
                continue

            # Age & Gender Prediction Blob
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227),
                                         MODEL_MEAN_VALUES, swapRB=False)

            # Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]
            gender_conf = gender_preds[0].max()

            # Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGE_LIST[age_preds[0].argmax()]
            age_conf = age_preds[0].max()

            # Label
            label = f"{gender} ({gender_conf*100:.1f}%), Age: {age} ({age_conf*100:.1f}%)"
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Age and Gender Recognition", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- 3. Example Usage ---
image_to_analyze = "image.png"
recognize_age_gender(image_to_analyze)






#Live web cam page

# import cv2
# import numpy as np

# # --- Model Paths ---
# FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
# FACE_PROTO = "deploy.prototxt.txt"
# AGE_MODEL = "age_net.caffemodel"
# AGE_PROTO = "age_deploy.prototxt"
# GENDER_MODEL = "gender_net.caffemodel"
# GENDER_PROTO = "gender_deploy.prototxt"

# # Mean values & class labels
# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
#             '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# GENDER_LIST = ['Male', 'Female']


# # --- Load Models ---
# face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
# age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
# gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

# # --- Webcam Loop ---
# cap = cv2.VideoCapture(0)   # 0 = default webcam

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Failed to grab frame")
#         break

#     h, w = frame.shape[:2]

#     # Face Detection
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
#                                  (104.0, 177.0, 123.0),
#                                  swapRB=False, crop=False)
#     face_net.setInput(blob)
#     detections = face_net.forward()

#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.5:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")

#             # Clamp coordinates
#             startX, startY = max(0, startX), max(0, startY)
#             endX, endY = min(w - 1, endX), min(h - 1, endY)

#             face_roi = frame[startY:endY, startX:endX]
#             if face_roi.size == 0:
#                 continue

#             # Age & Gender Prediction
#             face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227),
#                                               MODEL_MEAN_VALUES, swapRB=False)
#             gender_net.setInput(face_blob)
#             gender_preds = gender_net.forward()
#             gender = GENDER_LIST[gender_preds[0].argmax()]
#             gender_conf = gender_preds[0].max()

#             age_net.setInput(face_blob)
#             age_preds = age_net.forward()
#             age = AGE_LIST[age_preds[0].argmax()]
#             age_conf = age_preds[0].max()

#             label = f"{gender} ({gender_conf*100:.1f}%), Age: {age} ({age_conf*100:.1f}%)"

#             # Draw on frame
#             cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
#             cv2.putText(frame, label, (startX, startY - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     cv2.imshow("Webcam Age & Gender Recognition", frame)

#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
