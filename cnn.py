#importing libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#creating session
import keras
import tensorflow as tf
config = tf.ConfigProto(device_count = {'GPU':1, 'CPU':2})
sess = tf.Session(config = config)
keras.backend.set_session(sess)

def model_training():
    #defining network
    classifier = Sequential()

    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    classifier.add(Flatten())

    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))


    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    #initializing image datagenerator
    from keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    #loading training and testing sets from their respective folders
    training_set = train_datagen.flow_from_directory('training_set', target_size = (64, 64), batch_size = 64, class_mode = 'binary')

    test_set = test_datagen.flow_from_directory('test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')


    classifier.fit_generator(training_set, steps_per_epoch = 8000, epochs = 2, validation_data = test_set, validation_steps = 2000)


    from keras.models import model_from_json
    model_json = classifier.to_json()
    with open("model.json", "w") as file:
        file.write(model_json)

    classifier.save_weights("model.h5")
    
    print('model saved')
    

#model_training()

#loading model
from keras.models import model_from_json
json_file = open('model.json','r')
loaded = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded)
loaded_model.load_weights('model.h5')


#loading image and getting prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('file_destination_here', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'


print(prediction)