import streamlit as st
import tensorflow as tf
import streamlit as st
from keras.preprocessing.image import img_to_array
import cv2
from PIL import Image, ImageOps
import numpy as np
from pipreqs import pipreqs


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('Cnn+6_firefly.h5')
    return model


with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         ## Firefly algorithm and deep learning application to Covid19
        ###  By : Bouigrouane Hamza && Ameur Khadija
        #### encadred : Bouzaachane Khadija
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png", "jpeg"])
print(file)


st.set_option('deprecation.showfileUploaderEncoding', False)


def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = image.load_img("val/normal/" + i, target_size=(224, 224))
    # img = image.img_to_array(image)
    new_array = cv2.resize(img, (224, 224), 3)
    new_array = new_array.reshape(-1, 224, 224, 3)
    # img = np.expand_dims(img, axis=0)
    p = model.predict(new_array)
    # img = image.img_to_array(image)
    # img = np.expand_dims(image, axis=0)
    # img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

    # img_reshape = image[np.newaxis,...]
    # prediction = model.predict(image)

    # prediction = model.predict(img_reshape)

    return p


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    # score = tf.nn.softmax(predictions[0])
    Category = ["Covid", "Normal"]
    st.write(Category[int(predictions[0][0])])
# st.write(score)
# print(
#  "This image most likely belongs to {} with a {:.2f} percent confidence."
#  .format(class_names[np.argmax(score)], 100 * np.max(score))
# )