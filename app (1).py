!pip install streamlit
st.markdown('<h1 style="color:black;">Vgg 16 Image classification model</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">The image classification model classifies image into following categories:</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;"> Diagonal,  Horizontal, Structural, Vertical,</h3>', unsafe_allow_html=True)
!streamlit run filename.py
# background image to streamlit

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('/content/background.webp')
upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
c1, c2= st.columns(2)
if upload is not None:
  im= Image.open(upload)
  img= np.asarray(im)
  image= cv2.resize(img,(224, 224))
  img= preprocess_input(image)
  img= np.expand_dims(img, 0)
  c1.header('Input Image')
  c1.image(im)
  c1.write(img.shape)
  #load weights of the trained model.
  input_shape = (224, 224, 3)
  optim_1 = Adam(learning_rate=0.0001)
  n_classes=4
  vgg_model = model(input_shape, n_classes, optim_1, fine_tune=2)
  vgg_model.load_weights('/content/drive/MyDrive/vgg/tune_model19.weights.best.hdf5')

  # prediction on model
  vgg_preds = vgg_model.predict(img)
  vgg_pred_classes = np.argmax(vgg_preds, axis=1)
  c2.header('Output')
  c2.subheader('Predicted class :')
  c2.write(classes[vgg_pred_classes[0]] )
