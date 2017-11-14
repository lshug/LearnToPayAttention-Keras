import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.engine.topology import Layer
from keras.layers import Input
from keras.layers.core import Dense, Lambda, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.merge import Concatenate, Add
from keras.layers.pooling import MaxPooling2D
def AttentionVGG(att='att1', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=100):
    
    
    inp=Input(shape=(height,width,channels))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inp)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x) 

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x) #batch*x*y*channel  

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(pool1)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x) 

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(pool2)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x) 

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1')(pool3)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool2')(x)
    x = Flatten(name='flatten')(x)
    g = Dense(512, activation='relu', name='fc1')(x) #batch*512

    l1=Dense(512)(pool1) #batch*x*y*512
    c1=ParametrisedCompatibility()([l1,g]) #batch*x*y
    if compatibilityfunction=='dp':
        c1=Lambda(lambda lam: K.squeeze(tf.map_fn(lambda xy: K.dot(xy[0], xy[1]),elems=(lam[0],K.expand_dims(lam[1],-1)), dtype=tf.float32),3))([l1,g])  #batch*x*y    
    flatc1=Flatten()(c1) #batch*xy
    a1=Activation('softmax')(flatc1) #batch*xy
    reshaped1=Lambda(lambda la: K.reshape(K.map_fn(lambda xy: K.map_fn(lambda lam: K.reshape(lam,[-1]),elems=(xy),dtype=tf.float32),elems=(K.reshape(la,[-1,512,height//2,width//2])),dtype=tf.float32),[-1,(height//2)*(width//2),512]))(l1) #batch*xy*512. 
    g1=Lambda(lambda lam: K.squeeze(K.batch_dot(K.expand_dims(lam[0],1),lam[1]),1), name="g1")([a1,reshaped1]) #batch*512.
            
    
    l2=Dense(512)(pool2)
    c2=ParametrisedCompatibility()([l2,g])
    if compatibilityfunction=='dp':
        c2=Lambda(lambda lam: K.squeeze(tf.map_fn(lambda xy: K.dot(xy[0], xy[1]),elems=(lam[0],K.expand_dims(lam[1],-1)), dtype=tf.float32),3))([l2,g])
    flatc2=Flatten()(c2)
    a2=Activation('softmax')(flatc2) 
    reshaped2=Lambda(lambda la: K.reshape(K.map_fn(lambda xy: K.map_fn(lambda lam: K.reshape(lam,[-1]),elems=(xy),dtype=tf.float32),elems=(K.reshape(la,[-1,512,height//4,width//4])),dtype=tf.float32),[-1,(height//4)*(width//4),512]))(l2)
    g2=Lambda(lambda lam: K.squeeze(K.batch_dot(K.expand_dims(lam[0],1),lam[1]),1))([a2,reshaped2])
    
    l3=Dense(512)(pool3)
    c3=ParametrisedCompatibility()([l3,g])
    if compatibilityfunction=='dp':
        c3=Lambda(lambda lam: K.squeeze(tf.map_fn(lambda xy: K.dot(xy[0], xy[1]),elems=(lam[0],K.expand_dims(lam[1],-1)), dtype=tf.float32),3))([l3,g])    
    flatc3=Flatten()(c3)
    a3=Activation('softmax')(flatc3) 
    reshaped3=Lambda(lambda la: K.reshape(K.map_fn(lambda xy: K.map_fn(lambda lam: K.reshape(lam,[-1]),elems=(xy),dtype=tf.float32),elems=(K.reshape(la,[-1,512,height//8,width//8])),dtype=tf.float32),[-1,(height//8)*(width//8),512]))(l3) 
    g3=Lambda(lambda lam: K.squeeze(K.batch_dot(K.expand_dims(lam[0],1),lam[1]),1))([a3,reshaped3])
    
    out=''
    if gmode=='concat':
        glist=[g1]
        if att=='att2':
            glist.append(g2)
        if att=='att3':
            glist.append(g2)
            glist.append(g3)    
        predictedG=Concatenate(axis=1)([g1,g2,g3])    
        x=Dense(outputclasses)(predictedG)
        out=Activation("relu")(x)
    else:
        gd1=Dense(outputclasses, activation='relu')(g1)
        if att=='att' or att=='att1':
            out=gd1
        elif att=='att2':
            gd2=Dense(outputclasses, activatin='relu')(gd2)
            out=Add()([gd1,gd2])
            out=Lambda(lambda lam: lam/2)(out)
        else:
            gd2=Dense(outputclasses, activatin='relu')(gd2)
            gd3=Dense(outputclasses, activatin='relu')(gd3)
            out=Add()([gd1,gd2,gd3])
            out=Lambda(lambda lam: lam/3)(out)

    out=Activation("relu")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
class ParametrisedCompatibility(Layer):
    
    def __init__(self, **kwargs):
        super(ParametrisedCompatibility, self).__init__(**kwargs)

    def build(self, input_shape):
        self.u = self.add_weight(name='u', shape=(512, 1), initializer='uniform', trainable=True)
        super(ParametrisedCompatibility, self).build(input_shape)

    def call(self, x): #add l and g together with map_fn. Dot the sum with u.
        return K.dot(K.map_fn(lambda lam: lam[0]+lam[1],elems=(x),dtype=tf.float32), self.u)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2])

if __name__ == "__main__":
    a=AttentionVGG(att='att1', gmode='concat', compatibilityfunction='pc', height=32, width=32, channels=3, outputclasses=100)
