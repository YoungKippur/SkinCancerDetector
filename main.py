import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import cv2 as cv
import datetime

vid = cv.VideoCapture(1)
# cap = cv.imread("saved/23-2-2023 32218.png")

optimizer = "rmsprop"

module_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
m = tf.keras.Sequential([
    hub.KerasLayer(module_url, output_shape=[2048], trainable=False),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

m.build([None, 299, 299, 3])
m.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
m.summary()

# load the weights with the least loss
m.load_weights("benign-vs-malignant_64_rmsprop_0.384.h5")

def predict_image_class_path(img_path, model, threshold=0.5):
  img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = tf.expand_dims(img, 0) # Create a batch
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  img = tf.image.convert_image_dtype(img, tf.float32)
  predictions = model.predict(img)
  score = predictions.squeeze()
  if score >= threshold:
    text = f"This image is {100 * score:.2f}% malignant."
  else:
    text = f"This image is {100 * (1 - score):.2f}% benign."
  plt.imshow(img[0])
  plt.axis('off')
  plt.suptitle(text)
  plt.show()

while True:
  ret, frame = vid.read()
  cv.imshow('Camera', frame)

  if cv.waitKey(1) & 0xFF == ord('q'):
    break

  if cv.waitKey(1) & 0xFF == ord('c'):
    img_name = "saved/{}.png".format("%s-%s-%s %s%s%s" % (datetime.datetime.now().day, datetime.datetime.now().month, datetime.datetime.now().year, datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second))
    cv.imwrite(img_name, frame)
    print(img_name)
    predict_image_class_path(img_name, m, 0.2)


vid.release()
cv.destroyAllWindows()