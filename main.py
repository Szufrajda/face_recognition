import cv2 as cv
import numpy as np
import os

# Ustawienie minimalnego poziomu logowania dla TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import bibliotek
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

# Inicjalizacja modelu FaceNet
facenet = FaceNet()

# Wczytanie osadzeń twarzy oraz etykiet klas
faces_embeddings = np.load("faces_embeddings_done_4class.npz")
Y = faces_embeddings['arr_1']

# Inicjalizacja kodera etykiet
encoder = LabelEncoder()
encoder.fit(Y)

# Wczytanie kaskady Haarcascades do detekcji twarzy
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# Wczytanie wcześniej wytrenowanego modelu SVM za pomocą pickle
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# Otwarcie połączenia z kamerą (o numerze 0, co oznacza domyślną kamerę)
cap = cv.VideoCapture(0)

# Główna pętla programu
while cap.isOpened():
    # Wczytanie klatki obrazu z kamery
    _, frame = cap.read()

    # Konwersja klatki do przestrzeni kolorów RGB i odcieni szarości
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detekcja twarzy na klatce
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

    # Iteracja po każdej wykrytej twarzy
    for x, y, w, h in faces:
        # Wycinanie obszaru twarzy
        img = rgb_img[y:y+h, x:x+w]

        # Przeskalowanie obszaru twarzy do rozmiaru 160x160 pikseli
        img = cv.resize(img, (160, 160))

        # Dodanie wymiaru do osadzenia twarzy (1x160x160x3)
        img = np.expand_dims(img, axis=0)

        # Uzyskanie osadzenia twarzy za pomocą FaceNet
        ypred = facenet.embeddings(img)

        # Przewidywanie klasy twarzy przy użyciu modelu SVM
        face_name = model.predict(ypred)

        # Odkodowanie przewidzianej klasy na etykietę
        final_name = encoder.inverse_transform(face_name)[0]

        # Rysowanie ramki wokół twarzy
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)

        # Wyświetlenie etykiety nad ramką
        cv.putText(frame, str(final_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 3, cv.LINE_AA)

    # Wyświetlenie ramki z efektem rozpoznawania twarzy
    cv.imshow("Face Recognition:", frame)

    # Przerwanie pętli po naciśnięciu klawisza 'q'
    if cv.waitKey(1) & ord('q') == 27:
        break

# Zwolnienie zasobów kamery i zamknięcie wszystkich okien
cap.release()
cv.destroyAllWindows


