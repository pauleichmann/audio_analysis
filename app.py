from flask import Flask, request, jsonify
from flask_cors import CORS
from speechbrain.inference import foreign_class, EncoderDecoderASR
from transformers import pipeline
from pydub import AudioSegment
import os
import uuid
import logging
import threading

# 禁止日志信息干扰
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# 初始化 Flask 应用并启用 CORS
app = Flask(__name__)
CORS(app)

# 用于存储任务结果的全局字典
RESULTS = {}

# 初始化语音情感识别模型
speech_emotion_classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier"
)

# 初始化语音转文字模型
asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-transformer-transformerlm-librispeech",
    savedir="pretrained_models/asr-transformer-transformerlm-librispeech"
)

# 初始化文本情感分类器
text_emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# 定义情感映射
emotion_mapping = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "sad",
    "surprise": "neutral",
    "disgust": "angry",
}

def convert_to_python_type(value):
    """
    将可能的 Tensor 或复杂对象转换为 Python 基础类型。
    """
    if hasattr(value, 'numpy'):
        return value.numpy().tolist()  # 如果是 Tensor，转换为 numpy，再转为 Python 列表
    elif hasattr(value, 'item'):
        return value.item()  # 如果是标量 Tensor，直接转换为基础类型
    elif isinstance(value, (list, dict)):
        return value  # 如果已经是基础类型，直接返回
    else:
        return value  # 默认直接返回


def process_audio(file_path, task_id):
    """处理音频文件，提取语音情感、转录文本并分析文本情感"""
    try:
        # 验证并转换为 PCM 编码的 WAV 格式
        audio = AudioSegment.from_file(file_path)
        pcm_wav_path = f"{task_id}_processed.wav"
        audio.export(pcm_wav_path, format="wav")

        # 使用模型分析
        out_prob, score, index, text_lab = speech_emotion_classifier.classify_file(pcm_wav_path)
        speech_emotion = text_lab[0]
        speech_confidence = convert_to_python_type(score[0])  # 转换为 float

        transcription = asr_model.transcribe_file(pcm_wav_path)

        text_emotion_result = text_emotion_classifier(transcription)
        text_emotion_label = text_emotion_result[0]['label']
        mapped_emotion = emotion_mapping.get(text_emotion_label, "neutral")
        text_confidence = convert_to_python_type(text_emotion_result[0]['score'])  # 转换为 float

        # 决定最终情感
        if speech_confidence >= text_confidence:
            final_emotion = speech_emotion
            final_confidence = speech_confidence
        else:
            final_emotion = mapped_emotion
            final_confidence = text_confidence

        # 存储结果
        RESULTS[task_id] = {
            "status": "completed",
            "result": {
                "transcription": transcription,
                "final_emotion": final_emotion,
                "final_confidence": final_confidence
            }
        }
    except Exception as e:
        RESULTS[task_id] = {"status": "error", "error": str(e)}
    finally:
        # 清理文件
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(pcm_wav_path):
            os.remove(pcm_wav_path)


@app.route('/upload', methods=['POST'])
def upload_audio():
    """上传音频接口，处理音频文件并启动后台任务"""
    if 'audioFile' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audioFile']

    task_id = str(uuid.uuid4())  # 生成唯一任务 ID
    RESULTS[task_id] = {"status": "processing"}

    # 保存上传的音频文件
    file_path = f"{task_id}.wav"
    audio_file.save(file_path)

    # 启动后台线程处理音频文件
    threading.Thread(target=process_audio, args=(file_path, task_id)).start()

    return jsonify({"task_id": task_id})


@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    """查询任务结果接口"""
    if task_id not in RESULTS:
        return jsonify({"error": "Invalid task_id"}), 404
    return jsonify(RESULTS[task_id])


@app.route('/')
def index():
    """API 主页"""
    return jsonify({
        "message": "Welcome to the Speech Processing API",
        "endpoints": {
            "upload": "/upload (POST)",
            "result": "/result/<task_id> (GET)"
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
