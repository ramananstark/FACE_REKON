
import numpy as np
from keras import layers
from keras import models
from keras import regularizers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir='/home/raman/dataset/data_UMD/cache1/color/train/'
test_dir='/home/raman/dataset/data_UMD/cache1/color/test/'



max_count=100
reg_val=[]
lr_val=[]
test_loss=[]
test_acc=[]

for i in range(max_count):

	print ("*"*30)
	print (str(i+1)+"/"+str(max_count))
	print ("*"*30)
	reg=10**(np.random.uniform(-4,0))
	lr=10**(np.random.uniform(-3,-4))

	model=models.Sequential()

	model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(60,60,3)))

	model.add(layers.MaxPooling2D((2,2)))

	model.add(layers.Conv2D(64,(3,3),activation='relu'))

	model.add(layers.MaxPooling2D(2,2))

	model.add(layers.Conv2D(128,(3,3),activation='relu'))

	model.add(layers.MaxPooling2D((2,2)))

	model.add(layers.Conv2D(128,(3,3),activation='relu'))

	model.add(layers.MaxPooling2D((2,2)))

	model.add(layers.Flatten())

	model.add(layers.Dense(512,activation='relu',kernel_regularizer=regularizers.l2(reg)))

	model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(reg)))




	model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
             metrics=['acc'])
	

	train_datagen=ImageDataGenerator(rescale=1./255)
	test_datagen=ImageDataGenerator(rescale=1./255)

	train_generator=train_datagen.flow_from_directory(
                train_dir,
                target_size=(60,60),
                batch_size=20,
                class_mode='binary')
	test_generator=test_datagen.flow_from_directory(
                test_dir,
                target_size=(60,60),
                batch_size=20,
                class_mode='binary'
                )

	history=model.fit_generator(
        	train_generator,
        	steps_per_epoch=100,
        	epochs=5,
        	validation_data=test_generator,
        	validation_steps=50)

	reg_val.append(reg)
	lr_val.append(lr)
	test_loss.append(history.history['val_loss'])
	test_acc.append(history.history['val_acc'])


'''
acc=history.history['acc']
test_acc=history.history['val_acc']
loss=history.history['loss']
test_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='TRAINING ACCURACY')
plt.plot(epochs,test_acc,'b',label='TEST ACCURACY')
plt.title('TRAINING AND TEST ACCURACY')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='TRAINING LOSS')
plt.plot(epochs,test_loss,'b',label='TEST LOSS')
plt.title('TRAINING AND TESTING LOSS')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
'''

print ("*"*30)
print ("Finding the highest Test Accuracy and lowest Test Loss...")

index1=0
index2=0
max_test_acc=max(test_acc[0])
min_test_loss=min(test_loss[0])
for i in range(max_count):
	temp1=max(test_acc[i])
    max_test_acc=temp1
	index1=i
    temp2=min(test_loss[i])
	if(temp2<min_test_loss):
		min_test_loss=temp2
		index2=i	  

print ('Maximum Testing Accuracy:',max_test_acc)
print ('Minimum Testing Loss:',min_test_loss)
print ('Value of optimum learning rate :',lr_val[index1])
print ('Value of optimum regularization:',reg_val[index2])
