import os
import joblib
import gradio as gr

MODEL_PATH = "fake_news_detector.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Не найден fake_news_detector.joblib. "
        "Добавьте файл обученной модели в корень Space."
    )

model = joblib.load(MODEL_PATH)

def predict_news(text):
    if not text or not text.strip():
        return "Введите текст новости для проверки."

    prediction = int(model.predict([text])[0])
    score = float(model.decision_function([text])[0])

    if prediction == 0:
        result = "FAKE — новость классифицирована как фейковая"
    else:
        result = "REAL — новость классифицирована как реальная"

    return (
        "РЕЗУЛЬТАТ ПРОВЕРКИ\n\n"
        f"{result}\n\n"
        f"Оценка классификатора: {score:.4f}\n\n"
        "Положительное значение оценки соответствует REAL,\n"
        "отрицательное — FAKE.\n"
        "Оценка не является вероятностью."
    )

demo = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(
        lines=12,
        placeholder="Вставьте текст новости сюда...",
        label="Текст новости"
    ),
    outputs=gr.Textbox(label="Результат анализа", lines=7),
    title="Fake News Detector",
    description=(
        "Интеллектуальная система распознавания фейковых новостей "
        "на основе машинного обучения. Введите текст новости "
        "и нажмите кнопку проверки."
    ),
    submit_btn="Проверить новость",
    clear_btn="Очистить"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
