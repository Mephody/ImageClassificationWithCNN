# -*- coding: utf-8 -*-
"""
Created on Fri May  4 19:24:55 2018

@author: Глеб
"""

from keras.callbacks import EarlyStopping


from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten


from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


#Каталог с данными для обучения
train_dir = 'Marcel-Train'
# Каталог с данными для проверки
val_dir = 'Marcel-Test'
# Каталог с данными для тестирования
#test_dir = 'test'
# Размеры изображения
img_width, img_height = 66, 76
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)

batch_size = 32 # 32 изображения обрабатываются за раз
num_epochs = 200 # 200 итераций обучения
kernel_size = 3 # ядра 3х3
pool_size = 2 # 2х2 pooling
conv_depth_1 = 32 #32 ядра на слой
conv_depth_2 = 64 # 64 ядра на слой
conv_depth_3 = 128 # 128 ядра на слой
drop_prob_1 = 0.25 
drop_prob_2 = 0.5 
hidden_size = 512 

num_classes = 6;

datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')





inp = Input(shape=input_shape) 
# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
#conv_1 = Convolution2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', activation='relu')(inp)
#conv_2 = Convolution2D(conv_depth_1, kernel_size, kernel_size, border_mode='same', activation='relu')(conv_1)

conv_1 = Conv2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Conv2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)


pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_3 = Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
conv_4 = Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)

# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_5 = Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_2)
conv_6 = Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_5)
pool_3 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_6)
drop_3 = Dropout(drop_prob_1)(pool_3)


# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
flat = Flatten()(drop_3)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)

model = Model(inputs=inp, outputs=out)

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer=opt, # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

early_stopping = EarlyStopping(monitor='val_loss', patience=25)

hist = model.fit_generator(train_generator,                     
                    epochs=num_epochs, 
                    steps_per_epoch=len(train_generator),
                    validation_data = val_generator,
                    validation_steps = 200,
                    callbacks=[early_stopping],
                    verbose = 1
                    
                    )

#%%

score = model.evaluate_generator(generator = val_generator, steps = len(val_generator));
print(score[1]);



#%%

# Генерируем описание модели в формате json
model_json = model.to_json()
# Записываем модель в файл
json_file = open("mnist_model.json", "w")
json_file.write(model_json)
json_file.close()

model.save_weights("mnist_model.h5")

#print('Сохраняем сеть')
#model.save("NNE.h5")
#print("Сохранение завершено!")

