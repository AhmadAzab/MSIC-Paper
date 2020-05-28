from keras.models import Sequential
from keras.layers import Conv2D,Activation,MaxPooling2D,Dense,Flatten,Dropout
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import keras
from keras.layers import BatchNormalization
from keras.optimizers import Adam
import pickle
from keras.models import load_model

#CNN 1 layer parameters definition
classifier = Sequential()
classifier.add(Conv2D(64,(3,3),input_shape=(200,200,3)))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))
classifier.add(Flatten())
classifier.add(Dense(128))
classifier.add(Dropout(0.5))
classifier.add(Activation('relu'))
classifier.add(Dense(11))#change according to the number of classes
classifier.add(Activation('softmax'))
classifier.summary()
classifier.compile(optimizer =keras.optimizers.Adam(lr=0.001),
                   loss ='categorical_crossentropy',
                   metrics =['accuracy'])

#CNN 2 layers parameters definition
classifier1 = Sequential()
classifier1.add(Conv2D(64,(3,3),input_shape=(200,200,3)))
classifier1.add(Activation('relu'))
classifier1.add(MaxPooling2D(pool_size =(2,2)))
classifier1.add(Conv2D(64,(3,3),input_shape=(200,200,3)))
classifier1.add(Activation('relu'))
classifier1.add(MaxPooling2D(pool_size =(2,2)))
classifier1.add(Flatten())
classifier1.add(Dense(128))
classifier1.add(Dropout(0.5))
classifier1.add(Activation('relu'))
classifier1.add(Dense(11))#change according to the number of classes
classifier1.add(Activation('softmax'))
classifier1.summary()
classifier1.compile(optimizer =keras.optimizers.Adam(lr=0.001),
                   loss ='categorical_crossentropy',
                   metrics =['accuracy'])

#CNN 3 layers parameters definition
classifier2 = Sequential()
classifier2.add(Conv2D(64,(3,3),input_shape=(200,200,3)))
classifier2.add(Activation('relu'))
classifier2.add(MaxPooling2D(pool_size =(2,2)))
classifier2.add(Conv2D(64,(3,3)))
classifier2.add(Activation('relu'))
classifier2.add(MaxPooling2D(pool_size =(2,2)))
classifier2.add(Conv2D(64,(3,3)))
classifier2.add(Activation('relu'))
classifier2.add(MaxPooling2D(pool_size =(2,2)))
classifier2.add(Flatten())
classifier2.add(Dense(128))
classifier2.add(Dropout(0.5))
classifier2.add(Activation('relu'))
classifier2.add(Dense(11))#change according to the number of classes
classifier2.add(Activation('softmax'))
classifier2.summary()
classifier2.compile(optimizer =keras.optimizers.Adam(lr=0.001),
                   loss ='categorical_crossentropy',
                   metrics =['accuracy'])


train_datagen = ImageDataGenerator(rescale =1./255,
                                   shear_range =0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip =True)
test_datagen = ImageDataGenerator(rescale = 1./255)

batchsize=10
training_set = train_datagen.flow_from_directory('Insert the training folder here', 
                                                target_size=(200,200),
                                                batch_size= batchsize,
                                                class_mode='categorical')

test_set = test_datagen.flow_from_directory('Insert the validation folder here', 
                                           target_size = (200,200),
                                           batch_size = batchsize,
					   shuffle=False,
                                           class_mode ='categorical')


#Build 1 layer CNN model with 100 epochs
history=classifier.fit_generator(training_set,
                        steps_per_epoch = 'insert the number of training samples without quotation' // batchsize,
                        epochs = 100,
                        validation_data =test_set,
                        validation_steps = 'insert the number of the validation samples without quotation' // batchsize)

#Build 2 layers CNN model with 100 epochs
history1=classifier1.fit_generator(training_set,
                        steps_per_epoch = 'insert the number of training samples without quotation' // batchsize,
                        epochs = 100,
                        validation_data =test_set,
                        validation_steps = 'insert the number of the validation samples without quotation' // batchsize)

#Build 3 layers CNN model with 100 epochs
history2=classifier2.fit_generator(training_set,
                        steps_per_epoch = 'insert the number of training samples without quotation' // batchsize,
                        epochs = 100,
                        validation_data =test_set,
                        validation_steps = 'insert the number of the validation samples without quotation' // batchsize)




#print confusion matrix and accuracy results for 1 layer CNN
Y_pred = classifier.predict_generator(test_set, steps= 'number of validation samples without quotation' // batchsize+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix 3 layers')
print(confusion_matrix(test_set.classes, y_pred))
print('Classification Report')
target_names = test_set.classes
class_labels = list(test_set.class_indices.keys()) 
#insert the labels of the classes in the training/testing folders in target_names below
target_names = ['label1','label2']
report = classification_report(test_set.classes, y_pred, target_names=class_labels)
print(report)

#print confusion matrix and accuracy results for 2 layers CNN 
Y_pred = classifier1.predict_generator(test_set, steps= 'number of validation samples without quotation' // batchsize+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix 3 layers')
print(confusion_matrix(test_set.classes, y_pred))
print('Classification Report')
target_names = test_set.classes
class_labels = list(test_set.class_indices.keys()) 
#insert the labels of the classes in the training/testing folders in target_names below
target_names = ['label1','label2']
report = classification_report(test_set.classes, y_pred, target_names=class_labels)
print(report)


#print confusion matrix and accuracy results for 3 layers CNN
Y_pred = classifier2.predict_generator(test_set, steps= 'number of validation samples without quotation' // batchsize+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix 3 layers')
print(confusion_matrix(test_set.classes, y_pred))
print('Classification Report')
target_names = test_set.classes
class_labels = list(test_set.class_indices.keys()) 
#insert the labels of the classes in the training/testing folders in target_names below
target_names = ['label1','label2']
report = classification_report(test_set.classes, y_pred, target_names=class_labels)
print(report) 

# summarize history for accuracy for the three built models through 100 epochs
plt.plot(history['val_accuracy'])
plt.plot(history1['val_accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('model accuracy using 64 filters, dropout and .001 Adam learning rate')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['CNN_1', 'CNN_2', 'CNN_3'], loc='upper left')
plt.show()
# summarize history for loss the three built models through 100 epochs
plt.plot(history['val_loss'])
plt.plot(history1['val_loss'])
plt.plot(history2.history['val_loss'])
plt.title('model loss using 64 filters, dropout and .001 Adam learning rate')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['CNN_1', 'CNN_2', 'CNN_3'], loc='upper left')
plt.show()


