import hashlib
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageFilter
from streamlit_drawable_canvas import st_canvas

from network_view import render_network, to_data_uri

MODEL_PATH = Path("models/handwritten_letter_cnn.keras")
REPORT_PATH = Path("outputs/classification_report.txt")
IMAGE_SIZE = 28
LETTER_SIZE = 24
CANVAS_SIZE = 214
DOUBT_MARGIN = 0.20
RECENT_ATTEMPTS = 3
RESULT_CARD_RESERVE = 360
LETTERS = [chr(ord("A") + index) for index in range(26)]
SIMILAR_LETTERS = [{"I", "L"}, {"G", "Q"}, {"U", "V"}, {"O", "D"}]


st.set_page_config(
    page_title="LETRA · Leitor de Escrita Treinado em Redes Artificiais",
    page_icon="✍️",
    layout="wide",
)

st.markdown(
    """
    <style>
    header[data-testid="stHeader"] { display: none; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    .st-key-network { position: fixed; inset: 0; z-index: 1; }
    .st-key-network iframe { height: 100vh !important; display: block; }

    .st-key-panel {
        position: fixed; top: 12px; left: 12px; z-index: 1000;
        width: 236px !important; gap: 0.4rem !important;
        padding: 10px 10px 10px; border-radius: 14px;
        background: rgba(17, 22, 42, 0.94); border: 1px solid #2a3150;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
        max-height: calc(100vh - 24px); overflow-y: auto;
    }
    .st-key-panel [data-testid="stHorizontalBlock"] { gap: 0.4rem; }
    .st-key-panel_actions { margin-top: -0.45rem; }
    .st-key-panel_actions [data-testid="stHorizontalBlock"] { gap: 0.25rem; }
    .st-key-panel p, .st-key-panel label { font-size: 0.78rem !important; }
    .st-key-panel button { min-height: 1.9rem; padding: 0.15rem 0.5rem; }
    .st-key-panel [data-testid="stSlider"] { padding: 0 0.3rem; }
    .st-key-panel [data-testid="stSliderTickBar"] { display: none; }

    .st-key-scoreboard {
        margin-top: 0.2rem; padding-top: 0.6rem !important;
        border-top: 1px solid #262d4a; gap: 0.45rem !important;
    }
    .score-bar {
        display: flex; height: 6px; margin: 0 0 0.4rem;
        border-radius: 999px; overflow: hidden; background: #232a42;
    }
    .score-bar .hit { background: linear-gradient(90deg, #10b981, #34d399); }
    .score-bar .miss { background: linear-gradient(90deg, #f87171, #ef4444); }
    .score-stats {
        display: flex; gap: 0.35rem; margin-bottom: 0.5rem;
        font-size: 0.72rem; color: #8b93a7;
    }
    .score-stats .hit b { color: #34d399; }
    .score-stats .miss b { color: #f87171; }
    .score-stats .rate b { color: #e6e8ef; }
    .st-key-score_row [data-testid="stMarkdownContainer"],
    .st-key-score_row [data-testid="stElementContainer"] { margin: 0 !important; }
    .score-recent { display: flex; align-items: center; gap: 0.3rem; }
    .score-chip {
        display: inline-grid; place-items: center; width: 1.45rem; height: 1.45rem;
        border-radius: 7px; font-size: 0.75rem; font-weight: 700;
    }
    .score-chip.hit { color: #34d399; background: rgba(52, 211, 153, 0.12); border: 1px solid rgba(52, 211, 153, 0.4); }
    .score-chip.miss { color: #f87171; background: rgba(248, 113, 113, 0.12); border: 1px solid rgba(248, 113, 113, 0.4); }
    .st-key-scoreboard button { min-height: 1.6rem !important; padding: 0 0.3rem !important; }

    .st-key-result_card {
        position: fixed; z-index: 1000; right: 20px; top: 50%;
        transform: translateY(-50%);
        width: 320px !important; max-height: calc(100vh - 120px); overflow-y: auto;
        gap: 0 !important; padding: 18px 18px 16px; border-radius: 18px;
        background: linear-gradient(160deg, rgba(31, 37, 80, 0.92), rgba(13, 17, 33, 0.95));
        border: 1px solid rgba(124, 92, 255, 0.35);
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.5), 0 0 40px rgba(124, 92, 255, 0.15);
        backdrop-filter: blur(6px);
    }
    .st-key-result_card p { font-size: 0.82rem; }
    .result-kicker {
        color: #8b93a7; font-size: 0.68rem; font-weight: 600;
        letter-spacing: 0.1em; text-transform: uppercase;
    }
    .result-head { display: flex; align-items: center; gap: 0.9rem; }
    .result-letter {
        display: grid; place-items: center; flex: none;
        width: 64px; height: 64px; border-radius: 16px;
        font-size: 2.3rem; font-weight: 800; line-height: 1;
        color: var(--tone); background: rgba(10, 14, 28, 0.6);
        border: 1px solid #2f3760; overflow: hidden;
        transition: border-color 0.35s, box-shadow 0.35s;
    }
    .result-letter .when-pending, .result-letter .skeleton { width: 100%; height: 100%; border-radius: 0; }
    .result-confidence { margin-top: 0.2rem; font-size: 1.45rem; font-weight: 700; line-height: 1.1; }
    .result-confidence span { display: block; color: #8b93a7; font-size: 0.72rem; font-weight: 500; }
    .result-confidence-skeleton { width: 110px; height: 1.5rem; margin-top: 0.3rem; }
    .result-section {
        margin: 1.15rem 0 0.55rem; color: #8b93a7; font-size: 0.68rem; font-weight: 600;
        letter-spacing: 0.1em; text-transform: uppercase;
    }
    .doubt {
        margin-top: 0.75rem; padding: 0.45rem 0.6rem; border-radius: 10px;
        color: #fbbf24; font-size: 0.74rem; background: rgba(251, 191, 36, 0.08);
        border: 1px solid rgba(251, 191, 36, 0.25);
    }
    .bar-row { display: flex; align-items: center; gap: 0.55rem; margin: 0 0 0.5rem; }
    .bar-row:last-child { margin-bottom: 0; }
    .bar-letter { width: 1rem; font-weight: 700; font-size: 0.8rem; }
    .bar-track { flex: 1; height: 7px; background: #232a42; border-radius: 999px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 999px; }
    .bar-value { width: 3rem; text-align: right; color: #c9cede; font-size: 0.75rem; }
    .model-view { display: flex; align-items: center; justify-content: center; gap: 0.6rem; }
    .model-view figure { margin: 0; text-align: center; }
    .model-view img {
        display: block; width: 104px; height: 104px; border-radius: 12px;
        background: #000; border: 1px solid #2f3760;
    }
    .model-view img.pixelated { image-rendering: pixelated; border-color: rgba(34, 211, 238, 0.5); }
    .model-view figcaption { margin-top: 0.35rem; color: #8b93a7; font-size: 0.7rem; }
    .model-view .arrow { color: #7c5cff; font-size: 1.1rem; margin-bottom: 1.2rem; }
    .model-note { margin-top: 0.55rem; color: #6b7390; font-size: 0.68rem; text-align: center; line-height: 1.35; }
    .result-wait { margin-top: 1.1rem; color: #8b93a7; font-size: 0.75rem; text-align: center; }
    .st-key-result_feedback { display: none; gap: 0.55rem !important; }
    .st-key-result_feedback button { min-height: 2.2rem; border-radius: 10px; }
    .st-key-feedback_yes button:hover { border-color: #34d399; color: #34d399; }
    .st-key-feedback_no button:hover { border-color: #f87171; color: #f87171; }
    .feedback-done {
        margin: 0.2rem 0 0.5rem; padding: 0.55rem 0.7rem; border-radius: 10px;
        font-size: 0.78rem; text-align: center;
    }
    .feedback-done.hit { color: #34d399; background: rgba(52, 211, 153, 0.1); border: 1px solid rgba(52, 211, 153, 0.3); }
    .feedback-done.miss { color: #f87171; background: rgba(248, 113, 113, 0.1); border: 1px solid rgba(248, 113, 113, 0.3); }

    .when-ready { display: none; }
    .skeleton {
        border-radius: 6px;
        background: linear-gradient(90deg, #232a42 25%, #343c5e 50%, #232a42 75%);
        background-size: 200% 100%; animation: shimmer 1.2s linear infinite;
    }
    .bar-row .skeleton { height: 7px; }
    .bar-letter.skeleton { height: 0.8rem; }
    .bar-value.skeleton { height: 0.8rem; width: 2.4rem; margin-left: 0.6rem; }
    @keyframes shimmer {
        from { background-position: 200% 0; }
        to { background-position: -200% 0; }
    }
    @keyframes rise-in {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .app-loading {
        position: fixed; inset: 0; z-index: 100000;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        gap: 1rem; transition: opacity 0.45s ease, visibility 0.45s ease;
        background: radial-gradient(ellipse at 55% 0%, #1f2550 0%, #11162a 55%, #0b0f1d 100%);
    }
    body[data-ready] .app-loading { opacity: 0; visibility: hidden; pointer-events: none; }
    body:has([data-testid="stAlert"]) .app-loading { display: none; }
    .app-loading-ring {
        width: 54px; height: 54px; border-radius: 50%;
        border: 3px solid rgba(124, 92, 255, 0.2); border-top-color: #7c5cff;
        box-shadow: 0 0 30px rgba(124, 92, 255, 0.3);
        animation: spin 0.9s linear infinite;
    }
    .app-loading-title { color: #e6e8ef; font-size: 1.3rem; font-weight: 800; letter-spacing: 0.2em; }
    .app-loading-text { color: #8b93a7; font-size: 0.85rem; margin-top: -0.6rem; }
    @keyframes spin { to { transform: rotate(360deg); } }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-loading"><div class="app-loading-ring"></div>'
    '<div class="app-loading-title">LETRA</div>'
    '<div class="app-loading-text">Carregando a rede neural…</div></div>',
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the trained CNN."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_PATH}. Execute src/train.py primeiro."
        )

    return tf.keras.models.load_model(MODEL_PATH)


def reported_accuracy():
    if not REPORT_PATH.exists():
        return None
    match = re.search(r"^Accuracy: ([\d.]+)%", REPORT_PATH.read_text(), re.MULTILINE)
    return float(match.group(1)) if match else None


def preprocess_canvas(image_data):
    """Crop, resize, and center a drawing the way EMNIST images are framed."""
    image = Image.fromarray(image_data.astype(np.uint8)).convert("L")
    rows, columns = np.where(np.asarray(image) > 30)

    if rows.size == 0:
        return None

    cropped = image.crop(
        (columns.min(), rows.min(), columns.max() + 1, rows.max() + 1)
    ).filter(ImageFilter.GaussianBlur(radius=1.5))

    scale = LETTER_SIZE / max(cropped.size)
    new_size = (
        max(1, round(cropped.width * scale)),
        max(1, round(cropped.height * scale)),
    )
    cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)

    framed = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
    framed.paste(
        cropped,
        ((IMAGE_SIZE - cropped.width) // 2, (IMAGE_SIZE - cropped.height) // 2),
    )

    pixels = np.asarray(framed, dtype=np.float32) / 255.0
    return pixels[np.newaxis, ..., np.newaxis]


def drawing_hash(input_image):
    return hashlib.md5(input_image.tobytes()).hexdigest()


def confidence_color(confidence):
    if confidence >= 0.8:
        return "#34d399"
    if confidence >= 0.5:
        return "#fbbf24"
    return "#f87171"


def format_percent(value, digits=0):
    return f"{value:.{digits}%}".replace(".", ",")


def init_state():
    st.session_state.setdefault("canvas_key", 0)
    st.session_state.setdefault("stroke_width", 11)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("ask_letter", False)
    st.session_state.setdefault("runs", 0)


def clear_canvas():
    st.session_state.canvas_key += 1
    st.session_state.result = None
    st.session_state.ask_letter = False


def detect(model, input_image, canvas_image):
    probabilities = model.predict(input_image, verbose=0)[0]
    st.session_state.runs += 1
    st.session_state.result = {
        "run": st.session_state.runs,
        "image": input_image,
        "canvas": canvas_image,
        "probabilities": probabilities,
        "hash": drawing_hash(input_image),
        "time": datetime.now().strftime("%H:%M:%S"),
        "feedback": None,
    }


def register_feedback(correct, true_letter=None):
    result = st.session_state.result
    st.session_state.ask_letter = False
    if result is None or result["feedback"] is not None:
        return

    probabilities = result["probabilities"]
    order = np.argsort(probabilities)[::-1]
    st.session_state.history.append({
        "time": result["time"],
        "image": result["image"],
        "predicted": LETTERS[order[0]],
        "confidence": float(probabilities[order[0]]),
        "top3": [(LETTERS[index], float(probabilities[index])) for index in order[:3]],
        "correct": correct,
        "true_letter": true_letter,
    })
    result["feedback"] = {"correct": correct, "true_letter": true_letter}


def request_correction():
    st.session_state.ask_letter = True


def cancel_correction():
    st.session_state.ask_letter = False


@st.dialog("Qual era a letra?", on_dismiss=cancel_correction)
def correction_dialog():
    probabilities = st.session_state.result["probabilities"]
    predicted = LETTERS[int(np.argmax(probabilities))]

    st.caption(f"O modelo respondeu **{predicted}**. Qual letra você desenhou?")
    choice = st.pills(
        "Letra correta",
        [letter for letter in LETTERS if letter != predicted],
        label_visibility="collapsed",
    )
    if st.button("Confirmar", type="primary", width="stretch", disabled=choice is None):
        register_feedback(False, choice)
        st.rerun()


def mean_confidence(attempts):
    if not attempts:
        return "-"
    return format_percent(np.mean([attempt["confidence"] for attempt in attempts]))


@st.dialog("Detalhes da sessão", width="large")
def session_dialog():
    history = st.session_state.history
    hits = [attempt for attempt in history if attempt["correct"]]
    misses = [attempt for attempt in history if not attempt["correct"]]

    columns = st.columns(4)
    columns[0].metric("Tentativas", len(history))
    columns[1].metric("Acurácia", format_percent(len(hits) / len(history)))
    columns[2].metric("Confiança nos acertos", mean_confidence(hits))
    columns[3].metric("Confiança nos erros", mean_confidence(misses))

    rows = [
        {
            "#": number,
            "Hora": attempt["time"],
            "Desenho": to_data_uri(attempt["image"][0, ..., 0], "gray", 56),
            "Prevista": attempt["predicted"],
            "Confiança": attempt["confidence"],
            "Top 3": " · ".join(
                f"{letter} {format_percent(probability)}"
                for letter, probability in attempt["top3"]
            ),
            "Resultado": "Acertou" if attempt["correct"] else "Errou",
            "Correta": attempt["true_letter"] or "",
        }
        for number, attempt in reversed(list(enumerate(history, start=1)))
    ]

    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        row_height=60,
        column_config={
            "Desenho": st.column_config.ImageColumn("Desenho", width="small"),
            "Confiança": st.column_config.ProgressColumn(
                "Confiança", format="percent", min_value=0.0, max_value=1.0
            ),
        },
    )

    if misses:
        (true_letter, predicted), count = Counter(
            (attempt["true_letter"], attempt["predicted"]) for attempt in misses
        ).most_common(1)[0]
        st.caption(f"Mais confundida: era {true_letter}, previu {predicted} ({count}x)")


def network_caption():
    accuracy = reported_accuracy()
    if accuracy is None:
        return "CNN treinada do zero com o EMNIST Letters"
    return f"CNN treinada do zero com o EMNIST Letters · acurácia no teste {format_percent(accuracy / 100, 2)}"


def pending(content):
    return f'<div class="when-pending">{content}</div>'


def ready(content):
    return f'<div class="when-ready">{content}</div>'


def top5_bars(probabilities):
    order = np.argsort(probabilities)[::-1][:5]
    color = confidence_color(float(probabilities[order[0]]))
    return "".join(
        f'<div class="bar-row">'
        f'<div class="bar-letter">{LETTERS[index]}</div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{probabilities[index]:.1%};'
        f'background:{color if rank == 0 else "#7c5cff"}"></div></div>'
        f'<div class="bar-value">{format_percent(probabilities[index], 1)}</div>'
        f"</div>"
        for rank, index in enumerate(order)
    )


def bars_skeleton():
    return (
        '<div class="bar-row"><div class="bar-letter skeleton"></div>'
        '<div class="bar-track skeleton"></div><div class="bar-value skeleton"></div></div>'
    ) * 5


def doubt_note(probabilities):
    first, second = np.argsort(probabilities)[::-1][:2]
    if probabilities[first] - probabilities[second] >= DOUBT_MARGIN:
        return ""

    pair = {LETTERS[first], LETTERS[second]}
    note = ", letras parecidas no EMNIST" if pair in SIMILAR_LETTERS else ""
    return (
        f'<div class="doubt">Em dúvida entre {LETTERS[first]} e '
        f"{LETTERS[second]}{note}.</div>"
    )


def result_card_html(result):
    probabilities = result["probabilities"]
    winner = int(np.argmax(probabilities))
    confidence = float(probabilities[winner])
    color = confidence_color(confidence)
    drawing = np.asarray(Image.fromarray(result["canvas"].astype(np.uint8)).convert("L"))
    drawing_uri = to_data_uri(drawing / 255.0, "gray", CANVAS_SIZE)
    input_uri = to_data_uri(result["image"][0, ..., 0], "gray", 112)
    letter_skeleton = '<div class="skeleton"></div>'
    confidence_skeleton = '<div class="skeleton result-confidence-skeleton"></div>'
    confidence_html = (
        f'<div class="result-confidence" style="color:{color}">'
        f"{format_percent(confidence, 1)}<span>de confiança</span></div>"
    )
    wait_html = '<div class="result-wait">Aguardando a decisão da rede…</div>'

    return (
        '<div class="result-kicker">Decisão da rede</div>'
        '<div class="result-head" style="margin-top:0.6rem">'
        f'<div class="result-letter" style="--tone:{color}">'
        f"{pending(letter_skeleton)}{ready(LETTERS[winner])}</div>"
        f"<div>{pending(confidence_skeleton)}{ready(confidence_html)}</div>"
        "</div>"
        f"{ready(doubt_note(probabilities))}"
        '<div class="result-section">Mais prováveis</div>'
        f"{pending(bars_skeleton())}{ready(top5_bars(probabilities))}"
        '<div class="result-section">O que o modelo vê</div>'
        '<div class="model-view">'
        f'<figure><img src="{drawing_uri}"><figcaption>Seu desenho</figcaption></figure>'
        '<div class="arrow">→</div>'
        f'<figure><img class="pixelated" src="{input_uri}">'
        "<figcaption>Entrada 28×28</figcaption></figure>"
        "</div>"
        '<div class="model-note">Recortado, suavizado e centralizado num quadro de 28×28, '
        "como no EMNIST.</div>"
        f"{pending(wait_html)}"
    )


def render_feedback(result):
    feedback = result["feedback"]
    with st.container(key="result_feedback"):
        st.markdown('<div class="result-section">O modelo acertou?</div>', unsafe_allow_html=True)
        if feedback is None:
            with st.container(horizontal=True, gap="small"):
                st.button(
                    "SIM", icon=":material/thumb_up:", key="feedback_yes", width="stretch",
                    on_click=register_feedback, args=(True,),
                )
                st.button(
                    "NÃO", icon=":material/thumb_down:", key="feedback_no", width="stretch",
                    on_click=request_correction,
                )
        elif feedback["correct"]:
            st.markdown(
                '<div class="feedback-done hit">✓ Acerto registrado no placar</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="feedback-done miss">✕ Erro registrado · era '
                f'<b>{feedback["true_letter"]}</b></div>',
                unsafe_allow_html=True,
            )


def render_result_card():
    result = st.session_state.result
    if result is None:
        return

    with st.container(key="result_card"):
        st.markdown(result_card_html(result), unsafe_allow_html=True)
        render_feedback(result)


def render_animation_style():
    result = st.session_state.result
    done = f'body[data-animated="{"" if result is None else result["run"]}"]'
    st.html(
        f"<style>"
        f"{done} .when-pending {{ display: none; }}"
        f"{done} .when-ready {{ display: block; animation: rise-in 0.35s ease-out; }}"
        f"{done} .st-key-result_feedback {{ display: flex; animation: rise-in 0.35s ease-out; }}"
        f"{done} .result-letter {{ border-color: var(--tone);"
        f" box-shadow: 0 0 24px color-mix(in srgb, var(--tone) 35%, transparent); }}"
        f"</style>"
    )


def render_scoreboard():
    history = st.session_state.history
    if not history:
        return

    hits = sum(attempt["correct"] for attempt in history)
    misses = len(history) - hits
    chips = "".join(
        f'<span class="score-chip {"hit" if attempt["correct"] else "miss"}">'
        f'{attempt["predicted"]}</span>'
        for attempt in history[-RECENT_ATTEMPTS:]
    )

    with st.container(key="scoreboard"):
        st.markdown(
            f'<div class="score-bar"><div class="hit" style="width:{hits / len(history):.1%}"></div>'
            f'<div class="miss" style="width:{misses / len(history):.1%}"></div></div>'
            f'<div class="score-stats"><span class="hit"><b>{hits}</b> acertos</span>·'
            f'<span class="miss"><b>{misses}</b> erros</span>·'
            f'<span class="rate"><b>{format_percent(hits / len(history))}</b></span></div>',
            unsafe_allow_html=True,
        )
        with st.container(
            key="score_row", horizontal=True, horizontal_alignment="distribute",
            vertical_alignment="center", wrap=False,
        ):
            st.markdown(f'<div class="score-recent">{chips}</div>', unsafe_allow_html=True, width="content")
            if st.button("Ver Mais", type="tertiary", icon=":material/visibility:", key="details"):
                session_dialog()


def render_panel(model):
    with st.container(key="panel"):
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=st.session_state.stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            drawing_mode="freedraw",
            height=CANVAS_SIZE,
            width=CANVAS_SIZE,
            update_streamlit=True,
            return_image_data=True,
            key=f"canvas_{st.session_state.canvas_key}",
        )
        st.slider(
            "Espessura do traço", 8, 32, key="stroke_width", label_visibility="collapsed"
        )

        input_image = None
        if canvas_result.image_data is not None:
            input_image = preprocess_canvas(canvas_result.image_data)

        with st.container(key="panel_actions"):
            clear_column, detect_column = st.columns(2)
            clear_column.button(
                "Limpar", icon=":material/delete:", width="stretch", on_click=clear_canvas
            )
            if detect_column.button(
                "Detectar", type="primary", icon=":material/neurology:",
                width="stretch", disabled=input_image is None,
            ):
                detect(model, input_image, canvas_result.image_data)

        result = st.session_state.result
        if (
            result is not None
            and input_image is not None
            and drawing_hash(input_image) != result["hash"]
        ):
            st.caption(":orange[Desenho alterado. Clique em Detectar.]")

        render_scoreboard()


def main():
    init_state()

    try:
        model = load_model()
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(str(error))
        st.stop()

    render_panel(model)
    render_animation_style()

    with st.container(key="network"):
        result = st.session_state.result
        if result is None:
            render_network(model, caption=network_caption())
        else:
            render_network(
                model, result["image"], network_caption(), result["run"],
                RESULT_CARD_RESERVE,
            )

    render_result_card()

    if st.session_state.ask_letter and st.session_state.result is not None:
        correction_dialog()


if __name__ == "__main__":
    main()
