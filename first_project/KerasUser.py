from keras.layers import Conv2D,Dense,Flatten
from keras.models import Sequential
from keras.initializers import glorot_uniform
from keras.datasets import mnist
from keras.utils import np_utils
from keras.activations import softmax




(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape)
print(y_train.shape)

y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)
print(y_train.shape)
model = Sequential()
#model.add(Conv2D(filters=2,kernel_size=(2,2),strides=(1,1),padding='same',kernel_initializer=glorot_uniform(),input_shape=(5,5,3)))
model.add(Flatten(input_shape=(28,28)))
print(model.output_shape)
model.add(Dense(units=10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train,batch_size=32, nb_epoch=10, verbose=1)
acc = model.evaluate(x_test, y_test, verbose=0)

print('The accuracy is ',acc)



