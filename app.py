"""
Taller NASA SEEC 2027 - Analizador de Respuestas Guardianes de Meztli
App de Gradio para Hugging Face Spaces.
"""

import json
from collections import defaultdict

import cv2
import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import spaces
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Configuración y carga del modelo (se ejecuta una sola vez al iniciar)
# ---------------------------------------------------------------------------

MODEL_NAME = "trpakov/vit-face-expression"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FRAMES = 60  # límite de frames analizados por video, para no tardar demasiado

print(f"Cargando modelo de emociones ViT en {DEVICE}...")
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model = ViTForImageClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
ID2LABEL = model.config.id2label
print("Modelo cargado correctamente.")

# Detector de rostros: el modelo de emociones funciona mucho mejor si le
# damos solo la cara recortada, no el frame completo.
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

LABEL_DISPLAY = {
    "Neutral": "Neutral",
    "Alegria_Disfrute": "Alegría/Disfrute",
    "Curiosidad_Sorpresa": "Curiosidad/Sorpresa",
    "Disconfort": "Disconfort",
    "Ansiedad_Frustracion": "Ansiedad/Frustración",
    "Ansiedad_Miedo": "Ansiedad/Miedo",
    "Rechazo_Sensorial": "Rechazo Sensorial",
}

EMOTION_COLORS = {
    "Neutral": "#808080",
    "Alegría/Disfrute": "#4CAF50",
    "Curiosidad/Sorpresa": "#2196F3",
    "Disconfort": "#FF9800",
    "Ansiedad/Frustración": "#F44336",
    "Ansiedad/Miedo": "#9C27B0",
    "Rechazo Sensorial": "#795548",
}

EMOTION_SCALE = {
    "Ansiedad/Miedo": -3,
    "Ansiedad/Frustración": -2,
    "Rechazo Sensorial": -1,
    "Disconfort": -0.5,
    "Neutral": 0,
    "Curiosidad/Sorpresa": 1,
    "Alegría/Disfrute": 2,
}


# ---------------------------------------------------------------------------
# Lógica de análisis
# ---------------------------------------------------------------------------

def detect_face(frame_bgr):
    """Devuelve el recorte de la cara más grande encontrada, o None si no hay."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    # Nos quedamos con la cara de mayor área (la más cercana a cámara)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return frame_bgr[y : y + h, x : x + w]


def analyze_frame(frame_bgr):
    """Clasifica la emoción de un frame. Devuelve None si no se detectó rostro."""
    face = detect_face(frame_bgr)
    if face is None:
        return None

    rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_face)

    inputs = processor(images=pil_image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

    top_idx = int(torch.argmax(probs, dim=-1).item())
    raw_label = ID2LABEL[top_idx]
    return {
        "emotion": LABEL_DISPLAY.get(raw_label, raw_label),
        "confidence": round(probs[0, top_idx].item(), 4),
    }


def build_plot(timeline, stimulus, temperature):
    fig, ax = plt.subplots(figsize=(10, 4))
    timestamps = [t["timestamp"] for t in timeline]
    emotions = [t["emotion"] for t in timeline]
    y_values = [EMOTION_SCALE.get(e, 0) for e in emotions]
    colors = [EMOTION_COLORS.get(e, "#808080") for e in emotions]

    ax.plot(timestamps, y_values, "-", color="#999999", alpha=0.5, zorder=1)
    ax.scatter(timestamps, y_values, c=colors, s=90, alpha=0.9, zorder=2, edgecolors="white")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)

    ax.set_yticks(list(EMOTION_SCALE.values()))
    ax.set_yticklabels(list(EMOTION_SCALE.keys()))
    ax.set_title(f"Respuesta emocional — {stimulus} ({temperature}°C)")
    ax.set_xlabel("Tiempo (s)")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    return fig


def build_report(timeline, stimulus, temperature, student_id, skipped_frames):
    counts = defaultdict(int)
    for t in timeline:
        counts[t["emotion"]] += 1
    dominant = max(counts, key=counts.get)

    emotions = [t["emotion"] for t in timeline]
    changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i - 1])
    variability = changes / (len(emotions) - 1) if len(emotions) > 1 else 0.0
    avg_confidence = sum(t["confidence"] for t in timeline) / len(timeline)

    lines = [
        "REPORTE DE SESIÓN",
        "━" * 32,
        f"Estudiante: {student_id}",
        f"Estímulo: {stimulus}",
        f"Temperatura: {temperature}°C",
        f"Duración analizada: {timeline[-1]['timestamp']:.1f}s",
        f"Frames analizados: {len(timeline)}  |  Frames sin rostro detectado: {skipped_frames}",
        "━" * 32,
        f"Emoción dominante: {dominant} ({counts[dominant]}/{len(timeline)} frames)",
        f"Confianza promedio del modelo: {avg_confidence:.2%}",
        f"Variabilidad emocional: {variability:.2f}",
        "",
        "INTERPRETACIÓN:",
    ]

    if variability < 0.3:
        lines.append("• Respuesta estable. Baja reactividad sensorial.")
    else:
        lines.append("• Alta variabilidad. Posible sobreestimulación.")

    if "Alegría" in dominant or "Curiosidad" in dominant:
        lines.append("• Experiencia predominantemente positiva.")
    elif "Ansiedad" in dominant or "Rechazo" in dominant or "Disconfort" in dominant:
        lines.append("• Experiencia desafiante. Considerar reducir exposición.")
    else:
        lines.append("• Respuesta neutral, sin señales fuertes en ninguna dirección.")

    if skipped_frames > len(timeline):
        lines.append(
            "• Aviso: se detectó rostro en pocos frames; verifica iluminación/encuadre del video."
        )

    return "\n".join(lines)


@spaces.GPU(duration=120)
def process_video(video_path, stimulus, temperature, student_id, progress=gr.Progress()):
    if not video_path:
        return None, "Sube un video para analizar.", None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "No se pudo abrir el video. Verifica el formato (mp4 recomendado).", None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    sample_rate = max(1, int(fps))  # ~1 muestra por segundo

    timeline = []
    skipped_frames = 0
    frame_count = 0

    progress(0, desc="Analizando video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:
            result = analyze_frame(frame)
            if result is not None:
                result["timestamp"] = round(frame_count / fps, 2)
                timeline.append(result)
            else:
                skipped_frames += 1

            progress(min(frame_count / total_frames, 1.0), desc="Analizando video...")

        frame_count += 1
        if len(timeline) >= MAX_FRAMES:
            break

    cap.release()

    if not timeline:
        return (
            None,
            "No se detectó ningún rostro en el video. Verifica encuadre e iluminación.",
            None,
        )

    fig = build_plot(timeline, stimulus, temperature)
    report = build_report(timeline, stimulus, temperature, student_id, skipped_frames)
    data = json.dumps(
        {
            "student": student_id,
            "stimulus": stimulus,
            "temperature": temperature,
            "skipped_frames": skipped_frames,
            "timeline": timeline,
        },
        indent=2,
        ensure_ascii=False,
    )

    return fig, report, data


# ---------------------------------------------------------------------------
# Interfaz de Gradio
# ---------------------------------------------------------------------------

with gr.Blocks(title="Guardianes de Meztli - Analizador de Emociones") as demo:
    gr.Markdown(
        "# 🚀 Taller Táctil NASA SEEC 2027\n"
        "### Analizador de Respuestas — Guardianes de Meztli"
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="📹 Video", sources=["upload"])
            student_id = gr.Textbox(label="👤 ID del estudiante", value="STU001")
            stimulus = gr.Dropdown(
                choices=[
                    "Regolito Lunar",
                    "Regolito Marciano",
                    "Arena Caliente",
                    "Arena Fría",
                ],
                value="Regolito Lunar",
                label="🎯 Estímulo",
            )
            temperature = gr.Slider(-10, 60, value=25, step=1, label="🌡️ Temperatura (°C)")
            btn = gr.Button("🔬 Analizar", variant="primary")

        with gr.Column():
            plot_output = gr.Plot(label="📈 Línea de tiempo emocional")
            report_output = gr.Textbox(label="📝 Reporte", lines=12)
            json_output = gr.JSON(label="💾 Datos JSON")

    btn.click(
        fn=process_video,
        inputs=[video_input, stimulus, temperature, student_id],
        outputs=[plot_output, report_output, json_output],
    )

if __name__ == "__main__":
    demo.launch()
