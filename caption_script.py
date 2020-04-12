#!/usr/bin/env python
# coding: utf-8

# In[75]:


import numpy as np
from keras.preprocessing import image
from keras.preprocessing import sequence
from keras.layers import Embedding,SimpleRNN,Dense
from keras.models import *
from keras.callbacks import ModelCheckpoint # save the best model, fight overfiitting
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical


# In[76]:


from keras.layers import Dropout,LSTM,add


# In[77]:


from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions


# In[78]:


from keras.preprocessing.sequence import pad_sequences


# In[79]:


model = load_model("model_weights/model_8.h5")
model._make_predict_function()


# ## text and image processing using transfer learning

# In[80]:


model_temp = ResNet50(weights="imagenet",input_shape=(224,224,3))
model.summary()
#resnet will be used for just feature extraction


# In[81]:


model_resnet = Model(model_temp.input,model_temp.layers[-2].output)
model_resnet._make_predict_function()

# In[82]:


def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0) 
    #bcz img may b needed to b feed in batches .so its dim will become(n,224,224,3).so expand_im is used
    img = preprocess_input(img)#predefined func of keras
    return img


# In[83]:


def encode_img(img):
    img = preprocess_img(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape((1,feature_vector.shape[1]))
    return feature_vector


# In[84]:


import pickle


# In[86]:


with open("model_weights/wrd2idx.pkl","rb") as w2i:
    word2idx=pickle.load(w2i)
with open("model_weights/idx2wrd.pkl","rb") as i2w:
    idx2word=pickle.load(i2w)


# In[87]:


max_len = 38
def predict_caption(photo):
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word2idx[w] for w in in_text.split() if w in word2idx]
        sequence = pad_sequences([sequence],maxlen = max_len,padding = "post")
        #print(sequence)
        ypred = model.predict([photo,sequence])
        #print("ypred = ",str(ypred))
        ypred = ypred.argmax()
        #print("ypred = ",str(ypred))
        word = idx2word[ypred]
        in_text+=(" " + word)
        if word == "endseq":
            break
    final_caption = in_text.split()[1:-1]#remove startseq nd endseq
    final_caption = " ".join(final_caption)
        
    return final_caption


# In[95]:


def caption_this_img(imgnm):    
    img = encode_img(imgnm)
    caption = predict_caption(img)
    return caption
    


# In[ ]:




