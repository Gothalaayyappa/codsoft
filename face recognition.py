import cv2
import numpy as np

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    print("Warning: 'face_recognition' module not found. Face recognition will not work.")
    FACE_RECOGNITION_AVAILABLE = False

def detect_faces(image_path, model='haar'):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if model == 'haar':
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_locations = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
        face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in face_locations]
    elif FACE_RECOGNITION_AVAILABLE:
        face_locations = face_recognition.face_locations(rgb_image)
    else:
        print("Face recognition model not available.")
        return
    
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def recognize_faces(known_faces, known_names, image_path):
    if not FACE_RECOGNITION_AVAILABLE:
        print("Face recognition model not available.")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"
        
        if True in matches:
            match_index = np.argmin(face_recognition.face_distance(known_faces, face_encoding))
            name = known_names[match_index]
        
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    known_faces = []
    known_names = []
    
    known_image_path = 'known_image.jpg'
    if FACE_RECOGNITION_AVAILABLE:
        try:
            known_image = face_recognition.load_image_file(known_image_path)
            encodings = face_recognition.face_encodings(known_image)
            if encodings:
                known_faces.append(encodings[0])
                known_names.append("Person 1")
            else:
                print("Warning: No faces found in known image.")
        except FileNotFoundError:
            print(f"Error: Known image file '{known_image_path}' not found.")
    
    test_image_path = 'test_image.jpg'
    detect_faces(test_image_path)
    recognize_faces(known_faces, known_names, test_image_path)
