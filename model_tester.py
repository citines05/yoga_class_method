import os
import numpy as np
import tensorflow as tf
from absl import logging
import pathlib
import time
from keras import layers

# Configure TensorFlow to suppress informational messages and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
logging.set_verbosity(logging.ERROR)

# Transformer Encoder definition
init = tf.keras.initializers.HeUniform(seed=42)
reg1 = tf.keras.regularizers.L2(1e-4)
reg2 = tf.keras.regularizers.L2(1e-5)

@tf.keras.utils.register_keras_serializable()
class TransformerEncoder(layers.Layer):
    def __init__(self, num_heads, conv_dim, name, **kwargs):
        super(TransformerEncoder, self).__init__(name=name, **kwargs)

        self.num_heads = num_heads
        self.conv_dim = conv_dim

        self.layer_norm = layers.LayerNormalization()
        self.mha = layers.MultiHeadAttention(self.num_heads, self.conv_dim, kernel_initializer=init)
        self.conv_proj = tf.keras.Sequential([
            layers.Dense(self.conv_dim, activation='relu', kernel_regularizer=reg1, bias_regularizer=reg1, activity_regularizer=reg2),
            layers.Dense(self.conv_dim, activation='relu', kernel_regularizer=reg1, bias_regularizer=reg1, activity_regularizer=reg2),
            layers.Dense(self.conv_dim, kernel_regularizer=reg1, bias_regularizer=reg1, activity_regularizer=reg2)
        ])

    def build(self, input_shape):
        super(TransformerEncoder, self).build(input_shape)

    def call(self, inputs):
        attention_output = self.mha(query=inputs, value=inputs, key=inputs)
        proj_input = self.layer_norm(inputs + attention_output)
        proj_output = self.conv_proj(proj_input)
        return self.layer_norm(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.mha.num_heads,
            "conv_dim": self.mha.key_dim,
        })
        return config

# For keypoints
dataset_path = pathlib.Path('kp_ds_aug_2')

# Function to measure prediction latency
def measure_prediction_latency(model, dataset):
    start_time = time.time()
    predictions = model.predict(dataset)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
    return elapsed_time, predictions

# Load models
model_transformer = tf.keras.models.load_model("experiments/experiment_kp_ds_aug_2/kp_ds_aug_2_model_1_tranformer_cnn.keras",
                                               custom_objects={"TransformerEncoder": TransformerEncoder})
model_lstm = tf.keras.models.load_model("baselines/baseline-cnn-lstm.keras")
mobilenet = tf.keras.models.load_model("baselines/baseline-mobilenetv2.keras")

### Create dataset for keypoints
test_ds_keypoints = tf.data.Dataset.list_files(str(dataset_path/'test/*/*.npy'), shuffle=False)

# Load class names
class_names = np.array(sorted([item.name for item in dataset_path.glob('test/*')]))

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.argmax(one_hot)

def decode_npy(file_path):
    def load_npy(path):
        return np.load(path.decode('utf-8')).astype(np.float32)
    npy_data = tf.numpy_function(load_npy, [file_path], tf.float32)
    npy_data.set_shape([33, 4])
    return npy_data

def process_path(file_path, label=None):
    label = get_label(file_path)
    data = decode_npy(file_path)
    return data, label

test_ds_keypoints = test_ds_keypoints.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

def configure_for_performance(ds, shuffle=False):
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(32)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

test_ds_keypoints = configure_for_performance(test_ds_keypoints, shuffle=False)

### Create image dataset
batch_size = 32
img_height = 224
img_width = 224
rescale = 1./255
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=rescale)

test_ds_images = datagen.flow_from_directory(
    directory='dataset/test',
    target_size=(img_height, img_width),
    class_mode='sparse',
    color_mode='rgb',
    seed=123,
    shuffle=False,
    batch_size=batch_size)

# Measure prediction latency for each model
latency_transformer, transformer_pred = measure_prediction_latency(model_transformer, test_ds_keypoints)
latency_lstm, lstm_pred = measure_prediction_latency(model_lstm, test_ds_keypoints)
latency_mobilenet, mobilenet_pred = measure_prediction_latency(mobilenet, test_ds_images)

# Save results to a text file
output_file = "prediction_latency.txt"
with open(output_file, "w") as f:
    f.write(f"Latency (ms) - Transformer: {latency_transformer:.4f} ms\n")
    f.write(f"Latency (ms) - LSTM: {latency_lstm:.4f} ms\n")
    f.write(f"Latency (ms) - MobileNetV2: {latency_mobilenet:.4f} ms\n")

print(f"Results saved to {output_file}")
