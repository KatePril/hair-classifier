import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet101


tf.random.set_seed(42)
base_path = './data'

train_generator = ImageDataGenerator()
train_ds = train_generator.flow_from_directory(
    f'{base_path}/train',
    target_size=(200, 200),
    batch_size=20
)

val_generator = ImageDataGenerator()
val_ds = val_generator.flow_from_directory(
    f'{base_path}/val',
    target_size=(200, 200),
    batch_size=20
)

test_generator = ImageDataGenerator()
test_ds = test_generator.flow_from_directory(
    f'{base_path}/test',
    target_size=(200, 200),
    batch_size=20,
    shuffle=False
)

base_model = ResNet101(
    include_top=False,
    input_shape=(200, 200, 3)
)
base_model.trainable = False

inputs = keras.Input(shape=(200, 200, 3))
base = base_model(inputs, training=False)
vectors = layers.GlobalAveragePooling2D()(base)
outputs = layers.Dense(5, activation="softmax")(vectors)

model = keras.Model(inputs, outputs)
print(model.summary())

optimizer = keras.optimizers.SGD(learning_rate=0.02, momentum=0.8)
loss = keras.losses.CategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.fit(train_ds, epochs=10, validation_data=val_ds)

print(model.evaluate(test_ds))
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converted_model = converter.convert()

with open('hair-classifier.tflite', 'wb') as f:
    f.write(converted_model)