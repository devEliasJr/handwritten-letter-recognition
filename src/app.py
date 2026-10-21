
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

MODEL_PATH = Path("models/handwritten_letter_cnn.keras")
IMAGE_SIZE = 28
CANVAS_SIZE = 280


st.set_page_config(
    page_title="Handwritten Letter Recognition",
    page_icon="✍️",
    layout="centered",
)


@st.cache_resource
def load_model():
    """Load the trained CNN."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run src/train.py first."
        )

    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_canvas(image_data):
    """Crop, resize, and center a drawing for CNN inference."""
    image = Image.fromarray(image_data.astype(np.uint8)).convert("L")
    image_array = np.asarray(image)

    foreground = image_array > 30

    if not np.any(foreground):
        return None

    rows, columns = np.where(foreground)

    left = int(columns.min())
    right = int(columns.max()) + 1
    top = int(rows.min())
    bottom = int(rows.max()) + 1

    cropped = image.crop((left, top, right, bottom))
    cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)

    processed_image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)

    x_offset = (IMAGE_SIZE - cropped.width) // 2
    y_offset = (IMAGE_SIZE - cropped.height) // 2

    processed_image.paste(cropped, (x_offset, y_offset))

    image_array = np.asarray(processed_image, dtype=np.float32) / 255.0
    image_array = image_array[..., np.newaxis]

    return np.expand_dims(image_array, axis=0)


def main():
    st.title("Handwritten Letter Recognition")
    st.write(
        "Draw a single uppercase letter from A to Z, "
        "then click the button to recognize it."
    )

    try:
        model = load_model()
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(str(error))
        st.stop()

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=12,
        stroke_color="#FFFFFF",
        background_color="#000000",
        drawing_mode="freedraw",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        update_streamlit=True,
        return_image_data=True,
        key="letter_canvas",
    )

    if st.button("Recognize letter", type="primary"):
        if canvas_result.image_data is None:
            st.warning("Draw a letter first.")
            st.stop()

        input_image = preprocess_canvas(canvas_result.image_data)

        if input_image is None:
            st.warning("Draw a letter first.")
            st.stop()

        probabilities = model.predict(input_image, verbose=0)[0]

        predicted_index = int(np.argmax(probabilities))
        predicted_letter = chr(ord("A") + predicted_index)
        confidence = float(probabilities[predicted_index])

        st.subheader(f"Prediction: {predicted_letter}")
        st.metric("Confidence", f"{confidence:.2%}")

        st.caption(
            "Confidence is the model's estimated probability, "
            "not a guarantee that the prediction is correct."
        )

        st.subheader("Top 3 predictions")

        top_indices = np.argsort(probabilities)[-3:][::-1]

        for index in top_indices:
            letter = chr(ord("A") + int(index))
            probability = float(probabilities[index])

            st.write(f"**{letter}**: {probability:.2%}")
            st.progress(probability)


if __name__ == "__main__":
    main()
