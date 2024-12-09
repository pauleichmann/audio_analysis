from flask import Flask, request, jsonify
from flask_cors import CORS
from speechbrain.inference import foreign_class, EncoderDecoderASR
from transformers import pipeline
from pydub import AudioSegment
import os
import uuid
import logging
import threading

# Prohibit log information interference
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Initialize Flask application and enable CORS
app = Flask(__name__)
CORS(app)

# Global dictionary used for storing task results
RESULTS = {}

# Initialize the voice emotion recognition model
speech_emotion_classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier"
)

# Initialize the voice-to-text model
asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-transformer-transformerlm-librispeech",
    savedir="pretrained_models/asr-transformer-transformerlm-librispeech"
)

# Initialize text sentiment classifier
text_emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# Define emotional mapping
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
    Convert possible Tensors or complex objects to Python basic types.
    """
    if hasattr(value, 'numpy'):
        return value.numpy().tolist()  # If it is a Tensor, convert it to numpy, and then to a Python list.
    elif hasattr(value, 'item'):
        return value.item()  # If it is a scalar Tensor, directly convert it to a basic type.
    elif isinstance(value, (list, dict)):
        return value  # If it is already a basic type, return directly
    else:
        return value  # Default direct return


def process_audio(file_path, task_id):
    """Process audio files, extract speech emotion, transcribe text, and analyze text emotion."""
    try:
        # Verify and convert to PCM-encoded WAV format
        audio = AudioSegment.from_file(file_path)
        pcm_wav_path = f"{task_id}_processed.wav"
        audio.export(pcm_wav_path, format="wav")

        # Using model analysis
        out_prob, score, index, text_lab = speech_emotion_classifier.classify_file(pcm_wav_path)
        speech_emotion = text_lab[0]
        speech_confidence = convert_to_python_type(score[0])  # transfer to float

        transcription = asr_model.transcribe_file(pcm_wav_path)

        text_emotion_result = text_emotion_classifier(transcription)
        text_emotion_label = text_emotion_result[0]['label']
        mapped_emotion = emotion_mapping.get(text_emotion_label, "neutral")
        text_confidence = convert_to_python_type(text_emotion_result[0]['score'])  # transfer to float

        # Determine the final emotion
        if speech_confidence >= text_confidence:
            final_emotion = speech_emotion
            final_confidence = speech_confidence
        else:
            final_emotion = mapped_emotion
            final_confidence = text_confidence

        # Store the results
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
        # Clear files
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(pcm_wav_path):
            os.remove(pcm_wav_path)


@app.route('/upload', methods=['POST'])
def upload_audio():
    """Upload audio interface, process audio files and start background tasks"""
    if 'audioFile' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audioFile']

    task_id = str(uuid.uuid4())  # Generate a unique task ID
    RESULTS[task_id] = {"status": "processing"}

    # Save uploaded audio file
    file_path = f"{task_id}.wav"
    audio_file.save(file_path)

    # Start a background thread to process audio files
    threading.Thread(target=process_audio, args=(file_path, task_id)).start()

    return jsonify({"task_id": task_id})


@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    """Query task result interface"""
    if task_id not in RESULTS:
        return jsonify({"error": "Invalid task_id"}), 404
    return jsonify(RESULTS[task_id])


@app.route('/')
def index():
    """API homepage"""
    return jsonify({
        "message": "Welcome to the Speech Processing API",
        "endpoints": {
            "upload": "/upload (POST)",
            "result": "/result/<task_id> (GET)"
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
