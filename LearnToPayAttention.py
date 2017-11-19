import numpy as np
import tensorflow as tf
import keras
import os
from keras import backend as K
from keras.models import Model
from keras.engine.topology import Layer
from keras.layers import Input
from keras.layers.core import Dense, Lambda, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.merge import Concatenate, Add
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, LambdaCallback
class AttentionVGG:
    def __init__(self, att='att1', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=100, weight_decay=0.005):


        inp=Input(shape=(height,width,channels))

        regularizer=keras.regularizers.l2(weight_decay)

        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_regularizer=regularizer)(inp)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_regularizer=regularizer)(x) 

        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_regularizer=regularizer)(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_regularizer=regularizer)(x)

        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_regularizer=regularizer)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_regularizer=regularizer)(x)
        local1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_regularizer=regularizer)(x) #batch*x*y*channel 
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(local1)  

        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_regularizer=regularizer)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_regularizer=regularizer)(x)
        local2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_regularizer=regularizer)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(local2) 

        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_regularizer=regularizer)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_regularizer=regularizer)(x)
        local3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_regularizer=regularizer)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(local3) 

        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1', kernel_regularizer=regularizer)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2', kernel_regularizer=regularizer)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool2')(x)
        x = Flatten(name='flatten')(x)
        g = Dense(512, activation='relu', name='fc1', kernel_regularizer=regularizer)(x) #batch*512

        l1=Dense(512, kernel_regularizer=regularizer)(local1) #batch*x*y*512
        c1=ParametrisedCompatibility(kernel_regularizer=regularizer)([l1,g]) #batch*x*y
        if compatibilityfunction=='dp':
            c1=Lambda(lambda lam: K.squeeze(K.map_fn(lambda xy: K.dot(xy[0], xy[1]),elems=(lam[0],K.expand_dims(lam[1],-1)), dtype=tf.float32),3))([l1,g])  #batch*x*y    
        flatc1=Flatten()(c1) #batch*xy
        a1=Activation('softmax')(flatc1) #batch*xy
        reshaped1=Lambda(lambda la: K.map_fn(lambda lam: K.reshape(lam,[-1,512]),elems=[la],dtype=tf.float32))(l1) #batch*xy*512. 
        g1=Lambda(lambda lam: K.squeeze(K.batch_dot(K.expand_dims(lam[0],1),lam[1]),1), name="g1")([a1,reshaped1]) #batch*512.


        height=height//2
        width=width//2
        l2=local2
        c2=ParametrisedCompatibility(kernel_regularizer=regularizer)([l2,g])
        if compatibilityfunction=='dp':
            c2=Lambda(lambda lam: K.squeeze(K.map_fn(lambda xy: K.dot(xy[0], xy[1]),elems=(lam[0],K.expand_dims(lam[1],-1)), dtype=tf.float32),3))([l2,g])
        flatc2=Flatten()(c2)
        a2=Activation('softmax')(flatc2) 
        reshaped2=Lambda(lambda la: K.map_fn(lambda lam: K.reshape(lam,[-1,512]),elems=[la],dtype=tf.float32))(l2)
        g2=Lambda(lambda lam: K.squeeze(K.batch_dot(K.expand_dims(lam[0],1),lam[1]),1))([a2,reshaped2])

        height=height//2
        width=width//2
        l3=local3
        c3=ParametrisedCompatibility(kernel_regularizer=regularizer)([l3,g])
        if compatibilityfunction=='dp':
            c3=Lambda(lambda lam: K.squeeze(K.map_fn(lambda xy: K.dot(xy[0], xy[1]),elems=(lam[0],K.expand_dims(lam[1],-1)), dtype=tf.float32),3))([l3,g])    
        flatc3=Flatten()(c3)
        a3=Activation('softmax')(flatc3) 
        reshaped3=Lambda(lambda la: K.map_fn(lambda lam: K.reshape(lam,[-1,512]),elems=[la],dtype=tf.float32))(l3) 
        g3=Lambda(lambda lam: K.squeeze(K.batch_dot(K.expand_dims(lam[0],1),lam[1]),1))([a3,reshaped3])

        out=''
        if gmode=='concat':
            glist=[g3]
            if att=='att2':
                glist.append(g2)
            if att=='att3':
                glist.append(g2)
                glist.append(g1)    
            predictedG=g1
            if att!='att1' and att!='att':
                predictedG=Concatenate(axis=1)(glist)    
            x=Dense(outputclasses, kernel_regularizer=regularizer)(predictedG)
            out=Activation("softmax")(x)
        else:
            gd3=Dense(outputclasses, activation='softmax')(g3)
            if att=='att' or att=='att1':
                out=gd3
            elif att=='att2':
                gd2=Dense(outputclasses, activation='sotfmax', kernel_regularizer=regularizer)(g2)
                out=Add()([gd3,gd2])
                out=Lambda(lambda lam: lam/2)(out)
            else:
                gd2=Dense(outputclasses, activation='softmax', kernel_regularizer=regularizer)(g2)
                gd1=Dense(outputclasses, activation='softmax', kernel_regularizer=regularizer)(g1)
                out=Add()([gd1,gd2,gd3])
                out=Lambda(lambda lam: lam/3)(out)

        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer=keras.optimizers.SGD(lr=1, momentum=0.9, decay=0.0000001, nesterov=False),loss='categorical_crossentropy',metrics=['accuracy'])
        name=("(VGG-"+att+")-"+gmode+"-"+compatibilityfunction).replace('att)','att1)')
        print("Generated "+name)
        self.name=name
        self.model=model
    def StandardFit(self, dataset, X, Y):
        startingepoch=0
        pastepochs=list(map(int,[x.replace(".hdf5","").replace(self.name+"-"+dataset,"").replace(" ","") for x in os.listdir("weights") if self.name+"-"+dataset in x]))
        if pastepochs:
            self.model.load_weights("weights/"+self.name+"-"+dataset+" "+str(max(pastepochs))+".hdf5")
            startingepoch=max(pastepochs)            
        scaler=LearningRateScaler(25,0.5)
        checkpoint=ModelCheckpoint("weights/"+self.name+"-"+dataset+" {epoch}.hdf5",save_weights_only=True)
        epochprint=LambdaCallback(on_epoch_end=lambda epoch, logs:print("Passed epoch "+str(epoch)))
        callbackslist=[scaler,checkpoint,epochprint]
        self.model.fit(X,Y,128,300,callbacks=callbackslist,initial_epoch=startingepoch)
class ParametrisedCompatibility(Layer):

    def __init__(self, kernel_regularizer=None, **kwargs):
        super(ParametrisedCompatibility, self).__init__(**kwargs)
        self.regularizer=kernel_regularizer

    def build(self, input_shape):
        self.u = self.add_weight(name='u', shape=(512, 1), initializer='uniform', regularizer=self.regularizer, trainable=True)
        super(ParametrisedCompatibility, self).build(input_shape)

    def call(self, x): #add l and g together with map_fn. Dot the sum with u.
        return K.dot(K.map_fn(lambda lam: lam[0]+lam[1],elems=(x),dtype=tf.float32), self.u)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2])
class LearningRateScaler(keras.callbacks.Callback):
    def __init__(self, epochs, multiplier):
        self.multiplier=multiplier
        self.epochs=epochs
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        oldrate=K.get_value(self.model.optimizer.lr)
        lr=oldrate*self.multiplier
        if epoch>0 and epoch%self.epochs==0:
            K.set_value(self.model.optimizer.lr, lr)
            print("Updated learning rate from "+str(oldrate)+" to "+str(lr)+" on epoch "+str(epoch))



if __name__ == "__main__":
    a=AttentionVGG(att='att', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=10)
    X=np.ones((100,32,32,3))
    Y=3*np.ones((100,10))
    a.StandardFit("randomset",X,Y)
