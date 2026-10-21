import base64
import io
import json
from pathlib import Path

import matplotlib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

TEMPLATE_PATH = Path(__file__).with_name("network_view.html")
LAYER_NAMES = ["conv_block1", "conv_block2", "dense_features", "output"]
FEATURE_MAPS = 8
DENSE_NEURONS = 12
LETTERS = [chr(ord("A") + index) for index in range(26)]


@st.cache_resource
def activation_model(_model):
    """Model that returns the activations of the key layers."""
    outputs = [_model.get_layer(name).output for name in LAYER_NAMES]
    return tf.keras.Model(_model.inputs[0], outputs)


def to_data_uri(array, colormap="magma", size=56):
    array = array - array.min()
    peak = array.max()
    if peak > 0:
        array = array / peak

    rgba = (matplotlib.colormaps[colormap](array) * 255).astype(np.uint8)
    image = Image.fromarray(rgba).resize((size, size), Image.Resampling.NEAREST)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def strongest(values, count):
    top = np.argsort(values)[::-1][:count]
    return top, values[top] / max(float(values.max()), 1e-8)


def links(matrix):
    peak = max(float(np.abs(matrix).max()), 1e-8)
    rows, columns = matrix.shape
    return [
        [i, j, round(float(matrix[i, j]) / peak, 3)]
        for i in range(rows)
        for j in range(columns)
    ]


def network_data(model, input_image):
    conv1, conv2, dense, probabilities = activation_model(model)(
        input_image, training=False
    )
    conv1, conv2 = conv1.numpy()[0], conv2.numpy()[0]
    dense, probabilities = dense.numpy()[0], probabilities.numpy()[0]

    conv1_top, conv1_values = strongest(conv1.mean(axis=(0, 1)), FEATURE_MAPS)
    conv2_top, conv2_values = strongest(conv2.mean(axis=(0, 1)), FEATURE_MAPS)
    dense_top, dense_values = strongest(dense, DENSE_NEURONS)

    height, width, channels = conv2.shape
    pooled = conv2.reshape(height // 2, 2, width // 2, 2, channels).max(axis=(1, 3))
    hidden_layer = next(
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dense)
    )
    hidden_kernel = hidden_layer.get_weights()[0].reshape(*pooled.shape, -1)
    conv2_to_dense = np.einsum("hwc,hwcn->cn", pooled, hidden_kernel)

    output_kernel = model.get_layer("output").get_weights()[0]
    dense_to_output = dense[dense_top, np.newaxis] * output_kernel[dense_top]

    return {
        "layers": [
            {
                "title": "Entrada",
                "subtitle": "28x28 pixels",
                "caption": "A letra vira 784 números",
                "nodes": [{
                    "label": "Imagem 28x28",
                    "value": 1.0,
                    "image": to_data_uri(input_image[0, ..., 0], "gray", 112),
                }],
            },
            {
                "title": "Bordas e traços",
                "subtitle": f"Convolução 1 · {FEATURE_MAPS} de {conv1.shape[-1]} filtros",
                "caption": "Filtros acham bordas e curvas",
                "nodes": [
                    {
                        "label": f"Filtro #{index}",
                        "value": round(float(value), 3),
                        "image": to_data_uri(conv1[..., index]),
                    }
                    for index, value in zip(conv1_top, conv1_values)
                ],
            },
            {
                "title": "Formas",
                "subtitle": f"Convolução 2 · {FEATURE_MAPS} de {channels} filtros",
                "caption": "Traços viram partes da letra",
                "nodes": [
                    {
                        "label": f"Filtro #{index}",
                        "value": round(float(value), 3),
                        "image": to_data_uri(conv2[..., index]),
                    }
                    for index, value in zip(conv2_top, conv2_values)
                ],
            },
            {
                "title": "Combinação",
                "subtitle": f"Densa · {DENSE_NEURONS} de {dense.shape[0]} neurônios",
                "caption": "Neurônios juntam as pistas",
                "nodes": [
                    {"label": f"Neurônio #{index}", "value": round(float(value), 3)}
                    for index, value in zip(dense_top, dense_values)
                ],
            },
            {
                "title": "Decisão",
                "subtitle": "Softmax · 26 letras",
                "caption": "Votos viram probabilidades",
                "nodes": [
                    {"label": letter, "value": round(float(probability), 4)}
                    for letter, probability in zip(LETTERS, probabilities)
                ],
            },
        ],
        "links": [
            {"real": False, "items": links(conv1_values[np.newaxis, :])},
            {"real": False, "items": links(np.outer(conv1_values, conv2_values))},
            {"real": True, "items": links(conv2_to_dense[np.ix_(conv2_top, dense_top)])},
            {"real": True, "items": links(dense_to_output)},
        ],
        "winner": int(probabilities.argmax()),
    }


def render_network(model, input_image=None, caption="", run=None, reserve=0, height=800):
    """Render the animated network view, or its idle state when there is no input.

    When the animation ends, the parent page body gets ``data-animated`` set to ``run``.
    ``reserve`` is the width in pixels kept free on the right for the result card.
    """
    idle = input_image is None
    if idle:
        input_image = np.zeros((1, 28, 28, 1), dtype=np.float32)

    data = network_data(model, input_image)
    data["idle"] = idle
    data["caption"] = caption
    data["run"] = run
    data["reserve"] = reserve
    html = TEMPLATE_PATH.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(data)
    )
    st.iframe(html, height=height)
