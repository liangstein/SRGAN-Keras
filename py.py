from keras.models import Model;
from keras.regularizers import  l2,l1,l1_l2;
from keras.optimizers import rmsprop,adam,adagrad,SGD;
from keras.layers import Input,Dense,merge,Dropout,BatchNormalization,\
    Activation,Conv2D,MaxPooling1D,MaxPooling2D,AveragePooling2D,Reshape,Flatten,UpSampling2D,Conv2DTranspose;
from keras.layers.advanced_activations import PReLU,LeakyReLU;
import time;import os;import pickle;import random;
from keras.models import Sequential,load_model;import numpy as np;import json;
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau;
from keras.utils.vis_utils import plot_model;from tqdm import tqdm;
DIR=os.getcwd();

def generator_model(small_image_input):
    y0=Conv2DTranspose(kernel_size=4,filters=64,strides=(1,1),padding="same",name="generate_conv_1")(small_image_input);
    y0=BatchNormalization(name="generate_b1")(y0);
    y0 = LeakyReLU(alpha=0.2, name="generate_act1")(y0);
    y1=Conv2DTranspose(kernel_size=4,filters=128,strides=(1,1),padding="same",name="generate_conv_2")(y0);
    y1=BatchNormalization(name="generate_b2")(y1);
    y1 =LeakyReLU(alpha=0.2, name="generate_act2")(y1);
    y2=Conv2DTranspose(kernel_size=4,filters=128,strides=(1,1),padding="same",name="generate_conv_3")(y1);
    y2=BatchNormalization()(y2);
    y2=LeakyReLU(alpha=0.2,name="generate_act3")(y2);
    y2=UpSampling2D(size=(2,2),name="generate_ups1")(y2);
    y3=Conv2DTranspose(kernel_size=4,filters=128,strides=(1,1),padding="same",name="generate_conv_4")(y2);
    y3=BatchNormalization(name="generate_b3")(y3);
    y3=LeakyReLU(alpha=0.2,name="generate_act4")(y3);
    #y4=UpSampling2D(size=(2,2),name="generate_ups2")(y3);
    #y5=Conv2D(kernel_size=3,filters=128,strides=(1,1),padding="same",name="generate_conv_5")(y4);
    #y5 = BatchNormalization(name="generate_b4")(y5);
    #y5=LeakyReLU(alpha=0.2,name="generate_act6")(y5)
    y6=Conv2DTranspose(filters=3,kernel_size=1,strides=(1,1),padding="same",name="generate_conv_6")(y3);
    y_final=Activation("tanh",name="generate_act7")(y6);
    return Model(inputs=small_image_input,outputs=y_final);

def discriminator_model(large_image_input,nearest_image_input):
    image_input=merge([large_image_input,nearest_image_input],mode="concat",concat_axis=-1)
    y0=Conv2D(kernel_size=4,filters=32,strides=(1,1),padding="valid",name="discriminator_conv_1")(image_input);
    y0=LeakyReLU(alpha=0.2,name="discriminator_act_1")(y0);
    y0=Conv2D(kernel_size=4,filters=32,strides=(2,2),padding="valid",name="discriminator_conv_2")(y0);
    y0=LeakyReLU(alpha=0.2,name="discriminator_act_2")(y0);
    y1=Conv2D(kernel_size=4,filters=64,strides=(1,1),padding="valid",name="discriminator_conv_3")(y0);
    y1=LeakyReLU(alpha=0.2,name="discriminator_act_3")(y1);
    y1=Conv2D(kernel_size=4,filters=64,strides=(2,2),padding="valid",name="discriminator_conv_4")(y1);
    y1=LeakyReLU(alpha=0.2,name="discriminator_act_4")(y1);
    #y2=Conv2D(kernel_size=4,filters=256,strides=(1,1),padding="valid",name="discriminator_conv_5")(y1);
    #y2=Activation("tanh",name="discriminator_act_5")(y2);
    #y2=Conv2D(kernel_size=4,filters=256,strides=(2,2),padding="valid",name="discriminator_conv_6")(y2);
    #y2=Activation("tanh",name="discriminator_act_6")(y2);
    y5=Flatten(name="discriminator_30")(y1);
    y5=Dense(512,name="discriminator_31")(y5);
    y5=LeakyReLU(alpha=0.2,name="discriminator_32")(y5);
    #y5=Dropout(0.2,name="discriminator_drop1")(y5);
    y_final=Dense(1,activation="sigmoid",name="discriminator_33")(y5);
    return Model(inputs=[large_image_input,nearest_image_input],outputs=y_final);

def set_trainable(model, key_word, value=True):
    layers_list = [layer for layer in model.layers if key_word in layer.name]
    for layer in layers_list:
        layer.trainable = value

def g_d_together(g_model,d_model,small_image_input,nearest_image_input):
    g_output=g_model(small_image_input);
    output=d_model([g_output,nearest_image_input]);
    return Model(inputs=[small_image_input,nearest_image_input],outputs=output);

#prepare dataset
'''from PIL import Image,ImageDraw,ImageFont;
import random;import os;
import numpy as np;
DIR=os.getcwd();N=96
for count in np.arange(0,1):
    img=Image.new("RGB",size=(N,N),color="white");
    draw=ImageDraw.Draw(img);
    color_list=["blue","red","green"];
    object_count=3;
    x0=random.randrange(0,N-10);y0=random.randrange(0,N-10);
    x1=random.randrange(x0+10,N);y1=random.randrange(y0+10,N);
    draw.ellipse([x0,y0,x1,y1],fill=color_list[0]);
    x0 = random.randrange(0, N-10);
    y0 = random.randrange(0, N-10);
    x1 = random.randrange(x0 + 10, N);
    y1 = random.randrange(y0 + 10, N);
    draw.rectangle([x0,y0,x1,y1],fill=color_list[1])
    x0 = random.randrange(0, N-10);
    y0 = random.randrange(0, N-10);
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
    draw.text([x0,y0],"Hello World",font=fnt,fill=color_list[2]);
    img.save(DIR+"/"+str(count)+".png");
    img1=img.resize((24,24));
    img1.save(DIR+"/small/"+str(count)+".png");
    if count%1000==0:print("{} completed".format(str(count*10000**-1)));

from PIL import Image,ImageDraw,ImageFont;
import random;import os;
import numpy as np;import pickle;
DIR=os.getcwd();
all_large_image_matrix=np.zeros((10000,96,96,3),dtype=np.uint8);
all_small_image_matrix=np.zeros((10000,24,24,3),dtype=np.uint8);
for i in np.arange(0,10000):
    img=Image.open(DIR+"/"+str(i)+".png");
    img=np.array(img);
    all_large_image_matrix[i]=img;
    img1=Image.open(DIR+"/small/"+str(i)+".png");
    img1=np.array(img1);
    all_small_image_matrix[i]=img1;
    if i%1000==0:print("{} completed".format(str(i*10000**-1)));

with open(DIR+"/all_large_image_matrix","wb") as f:
    pickle.dump(all_large_image_matrix,f,protocol=pickle.HIGHEST_PROTOCOL);

with open(DIR+"/all_small_image_matrix","wb") as f:
    pickle.dump(all_small_image_matrix,f,protocol=pickle.HIGHEST_PROTOCOL);'''

# loading dataset
'''import os;import pickle;
import numpy as np;
from PIL import Image;
DIR=os.getcwd();
with open(DIR+"/all_large_image_matrix","rb") as f:
    all_large_image_matrix=pickle.load(f);

with open(DIR+"/all_small_image_matrix","rb") as f:
    all_small_image_matrix=pickle.load(f);'''

# start training
#def train():
small_image_input = Input(shape=(16, 16, 3));
g_model= generator_model(small_image_input);

large_image_input = Input(shape=(32, 32, 3));
nearest_image_input=Input(shape=(32,32,3));
d_model= discriminator_model(large_image_input,nearest_image_input);

g_d_model = g_d_together(g_model, d_model, small_image_input,nearest_image_input);

d_optim = adam(lr=0.0001)
g_optim = adam(lr=0.00002);
#g_model.compile(loss="mse",optimizer=adam(lr=0.001));
d_model.compile(loss='binary_crossentropy', optimizer=d_optim)
g_d_model.compile(
    loss='binary_crossentropy', optimizer=g_optim)
#plot_model(g_d_model, to_file="g_d_model.png", show_shapes=True);

'''from keras.datasets import mnist
from PIL import Image;
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28,28, 1);
X_train_small=np.zeros((X_train.shape[0],14,14,1));
for i in np.arange(0,len(X_train)):
    img=Image.fromarray(X_train[i].reshape((28,28)));
    img=img.resize((14,14));
    X_train_small[i]=np.array(img).reshape((14,14,1));'''

from keras.datasets import cifar10
from PIL import Image;import numpy as np;
from tqdm import tqdm;
(x_train, y_train), (x_test, y_test) = cifar10.load_data();
x_train=x_train.reshape(x_train.shape[0],32,32,3);
x_train_small=np.zeros((x_train.shape[0],16,16,3));
x_train_nearest=np.zeros((x_train.shape[0],32,32,3));
for i in np.arange(0,len(x_train)):
    img_array=x_train[i];
    img=Image.fromarray(img_array);
    img1=img.resize((16,16),resample=Image.BICUBIC);
    x_train_small[i]=np.array(img1);
    img2=img1.resize((32,32),resample=Image.NEAREST);
    x_train_nearest[i]=np.array(img2);

#pretrain generator
#g_model.compile(loss="mse",optimizer=g_optim);
#g_model.fit(x=x_train_small*255**-1,y=x_train*255**-1,batch_size=200,epochs=2);


#pretrain discriminator
'''batch=200;
for epoch in range(1):
    initial_loss=[];
    all_labels=np.arange(0,len(x_train));np.random.shuffle(all_labels);
    batch_labels=np.array_split(all_labels,int(len(x_train)*batch**-1));
    for t in tqdm(range(len(batch_labels))):
        batch_large_input=np.zeros((batch,32,32,3),dtype=np.float32);
        batch_nearest_input=np.zeros((batch,32,32,3),dtype=np.float32);
        batch_small_input=np.zeros((batch,16,16,3),dtype=np.float32);
        for i,ele in enumerate(batch_labels[t]):
            batch_large_input[i]=(x_train[ele]-127.5)*127.5**-1;
            batch_small_input[i]=x_train_small[ele]*255**-1;
            batch_nearest_input[i]=(x_train_nearest[ele]-127.5)*127.5**-1;
        batch_predict=g_model.predict(batch_small_input);
        X=np.concatenate((batch_large_input,batch_predict));
        X1=np.concatenate((batch_nearest_input,batch_nearest_input));
        y = [random.uniform(0.7,1) for _ in range(batch)]+\
                [random.uniform(0,0.3) for _ in range(batch)];
        initial_loss.append(d_model.train_on_batch([X,X1],y));
    with open(DIR+"/log","a") as f:
        f.write("initial_loss: "+str(np.mean(initial_loss))+"\n");'''

g_model.load_weights(DIR+"/g_weights");
d_model.load_weights(DIR+"/d_weights");
batch_size=200;total_loss={};d_loss_average=8;g_loss_average=8;
for epoch in range(60):
    total_loss["d_loss"] = [];
    total_loss["g_loss"] = [];
    all_image_labels=np.arange(0,len(x_train));
    np.random.shuffle(all_image_labels);
    batch_labels=np.array_split(all_image_labels,int(len(x_train)/batch_size));
    for j in tqdm(range(len(batch_labels))):
        batch_large_image_matrix = np.zeros((batch_size, 32, 32, 3), dtype=np.float32);
        batch_nearest_image_matrix = np.zeros((batch_size, 32, 32, 3), dtype=np.float32);
        batch_small_image_matrix = np.zeros((batch_size, 16, 16, 3), dtype=np.float32);
        for i,ele in enumerate(batch_labels[j]):
            batch_large_image_matrix[i]=(x_train[ele]-127.5)*127.5**-1;
            batch_nearest_image_matrix[i]=(x_train_nearest[ele]-127.5)*127.5**-1;
            batch_small_image_matrix[i]=x_train_small[ele]*255**-1;
        generated_images = g_model.predict(batch_small_image_matrix, verbose=0);
        X = np.concatenate((batch_large_image_matrix, generated_images));
        X1=np.concatenate((batch_nearest_image_matrix,batch_nearest_image_matrix));
        y = [random.uniform(0.7,1) for _ in range(batch_size)]+\
            [random.uniform(0,0.3) for _ in range(batch_size)];
        set_trainable(d_model, "discriminator", True);
        d_loss = d_model.train_on_batch([X,X1], y);
        total_loss["d_loss"].append(d_loss);
        set_trainable(d_model, "discriminator", False);
        g_loss = g_d_model.train_on_batch([batch_small_image_matrix,batch_nearest_image_matrix],
                                          [random.uniform(0.7,1) for _ in range(batch_size)])
        total_loss["g_loss"].append(g_loss);
    d_loss_average=np.mean(total_loss["d_loss"]);
    g_loss_average=np.mean(total_loss["g_loss"]);
    with open(DIR+"/log","a") as f:
        f.write(str(epoch)+"\t"+str(d_loss_average)+"\t"+str(g_loss_average)+"\n");
    g_model.save_weights(DIR+"/g_weights");
    d_model.save_weights(DIR+"/d_weights");


