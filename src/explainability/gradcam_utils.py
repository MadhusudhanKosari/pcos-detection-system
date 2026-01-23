import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")


def generate_gradcam(image_path, output_path):

    model = tf.keras.models.load_model("models/image_cnn_classifier.h5")

    # Load image
    img_pil = Image.open(image_path).convert("RGB").resize((224,224))
    img_array = np.array(img_pil) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # ----------------------------
    # AUTO FIND LAST CONV LAYER
    # ----------------------------
    last_conv_layer = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No convolution layer found for GradCAM")

    # ----------------------------
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.45 + img_array * 255

    cv2.imwrite(output_path, superimposed_img)
