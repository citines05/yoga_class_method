import tensorflow as tf
from keras import layers
from tensorflow.keras.regularizers import L2
from tensorflow.keras.initializers import HeUniform

# Customized Transformer Encoder
init = HeUniform(seed=42)
reg1 = L2(1e-4)
reg2 = L2(1e-5)

@tf.keras.utils.register_keras_serializable()
class TransformerEncoder(layers.Layer):
  def __init__(self, num_heads, conv_dim, name, **kwargs):
    super(TransformerEncoder, self).__init__(name=name, **kwargs)

    self.num_heads = num_heads
    self.conv_dim = conv_dim

    self.layer_norm = layers.LayerNormalization()
    self.mha = layers.MultiHeadAttention(self.num_heads, self.conv_dim,
                                         kernel_initializer=init)
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

model = tf.keras.models.load_model("kp_ds_aug_2_model_1_tranformer_cnn.keras",
                   custom_objects={"TransformerEnconder": TransformerEncoder})

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("yoga_model_lite.tflite", "wb") as f:
    f.write(tflite_model)

print("modelo salvo com sucesso")