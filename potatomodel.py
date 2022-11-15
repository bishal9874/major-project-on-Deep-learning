import tensorflow as tf
from tensorflow.python.keras import models,layers
import matplotlib.pyplot as plt
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "potato disease/disease",
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size= BATCH_SIZE

)
class_name = dataset.class_names
# print(class_name)
print(len(dataset)*32)
plt.figure(figsize=(10,10))
for image_batch,label_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.axis("off")
        plt.title(class_name[label_batch[i]])
        # plt.show()
train_size=0.8
dataset.take(10)
print(len(dataset)*train_size)
train_ds = dataset.take(54)
print(len(train_ds))
test_ds = dataset.skip(54)
print(len(test_ds))
val_size= 0.1
print(len(dataset)*val_size)
val_ds = test_ds.take(6)
print(len(val_ds))
test_ds = test_ds.skip(6)
print(len(test_ds))


    # print(image_batch[0])
    # print(label_batch.numpy())