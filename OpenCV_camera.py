# -*- coding: utf-8 -*-
"""
Created on Wed May  9 13:45:26 2018

@author: Gleb
"""

import numpy as np
import cv2

from PIL import Image

from keras.models import model_from_json
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# Каталог с данными для проверки
val_dir = 'Marcel-Test'
# Каталог с данными для тестирования
#test_dir = 'test'
# Размеры изображения
img_width, img_height = 66, 76

batch_size = 32 # 32 изображения обрабатываются за раз

datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rescale=1. / 255)

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


# Загружаем данные об архитектуре сети из файла json
json_file = open("mnist_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
# Создаем модель на основе загруженных данных
loaded_model = model_from_json(loaded_model_json)
# Загружаем веса в модель
loaded_model.load_weights("mnist_model.h5")

# Компилируем модель
loaded_model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy
# Проверяем модель на тестовых данных
scores = loaded_model.evaluate_generator(generator = val_generator, steps = len(val_generator));

print("Точность модели на тестовых данных: %.2f%%" % (scores[1]*100))
#%%


cap = cv2.VideoCapture(0)

#predict_directory = 'Prediction'

temp = np.zeros([1, img_width, img_height, 3])

while(True): 
    ret, frame = cap.read()
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    frame = cv2.flip(frame, 1);
    
    img = Image.fromarray(frame);
    img_resized = img.resize((img_height, img_width), Image.ANTIALIAS);
    res = np.array(img_resized);
    temp[0] = res;
  
    
    print(loaded_model.predict(temp).argmax());
    
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


