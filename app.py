import streamlit as st
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('/content/saved_models/my_model1.hdf5')
  model=tf.keras.models.load_model('/content/saved_models/my_model2.hdf5')
  model=tf.keras.models.load_model('/content/saved_models/my_model3.hdf5')
  model=tf.keras.models.load_model('/content/saved_models/my_model4.hdf5')
  return model
 model=load_model()
 st.write("""
         #Crack Detection
         """
         )

file = st.file_uploader("please upload an crack image", type=["jpg", "png"])
import cv2
import PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):

size = (180,180)
image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
img = np.asarray(image)
img_reshape = img[np.newaxis,...]
prediction = model.predict(img_reshape)

return prediction

if file is None:
st.text("please upload an image file")
else:
image = Image.open(file)
st.image(image,use_column_width=True)
predictions = import_and_predict(image, model)
class names['diagonal','horizontal','structural','vertical']
string="This image most likely is: "+class_names[np.argmax(predictions)]
st.success(string)

