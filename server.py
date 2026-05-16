from flask import Flask, request, jsonify
from flask_cors import CORS
import speech_recognition as sr
import json
import wave
import tempfile
import os
import numpy as np
import librosa
from markers import *
from datetime import datetime
import urllib.request
import zipfile
from supabase import create_client, Client
from collections import Counter
import torch
import torch.nn as nn
import joblib
from pydub import AudioSegment

app = Flask(__name__)
CORS(app)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


class RiskClassifier(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=32, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


NEURAL_MODEL_PATH = "fraud_risk_model.pth"
SCALER_PATH = "scaler.pkl"

neural_model = None
scaler = None

if os.path.exists(NEURAL_MODEL_PATH) and os.path.exists(SCALER_PATH):
    neural_model = RiskClassifier()
    neural_model.load_state_dict(torch.load(NEURAL_MODEL_PATH, map_location='cpu'))
    neural_model.eval()
    scaler = joblib.load(SCALER_PATH)


def extract_neural_features(text):
    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)

    hack_count = sum(1 for m in hacking_markers if m in text_lower)
    bank_count = sum(1 for m in bank_markers if m in text_lower)
    employee_count = sum(1 for m in employee_markers if m in text_lower)
    person_count = sum(1 for m in person_markers if m in text_lower)
    accident_count = sum(1 for m in accident_markers if m in text_lower)
    urgent_count = sum(1 for m in urgent_markers if m in text_lower)
    code_count = sum(1 for m in code_markers if m in text_lower)

    total_markers = hack_count + bank_count + employee_count + person_count + accident_count + urgent_count + code_count
    markers_density = (total_markers / word_count * 100) if word_count > 0 else 0

    categories = [hack_count, bank_count, employee_count, person_count, accident_count, urgent_count, code_count]
    has_multiple_categories = sum(1 for c in categories if c > 0)

    all_markers = hacking_markers + bank_markers + employee_markers + person_markers + accident_markers + urgent_markers + code_markers
    unique_markers = len(set(m for m in all_markers if m in text_lower))
    unique_ratio = unique_markers / total_markers if total_markers > 0 else 0

    first_marker_pos = word_count
    for i, word in enumerate(words):
        if any(m in word for m in all_markers):
            first_marker_pos = i
            break

    bigram_hack_bank = hack_count * bank_count
    bigram_code_urgent = code_count * urgent_count
    bigram_employee_hack = employee_count * hack_count
    bigram_person_accident = person_count * accident_count
    bigram_code_bank = code_count * bank_count
    bigram_urgent_bank = urgent_count * bank_count
    bigram_employee_code = employee_count * code_count

    bigrams_sum = (bigram_hack_bank + bigram_code_urgent + bigram_employee_hack +
                   bigram_person_accident + bigram_code_bank + bigram_urgent_bank +
                   bigram_employee_code)
    high_frequency_bigrams = bigrams_sum / 7 if total_markers > 0 else 0

    sentence_count = text.count('.') + text.count('!') + text.count('?') + 1
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else word_count
    exclamation_count = text.count('!')

    word_freq = Counter(words)
    repetition_rate = sum(1 for c in word_freq.values() if c > 2) / max(1, len(word_freq))

    return np.array([
        hack_count, bank_count, employee_count, person_count, accident_count, urgent_count, code_count,
        markers_density, has_multiple_categories, total_markers, unique_ratio, first_marker_pos,
        bigram_hack_bank, bigram_code_urgent, bigram_employee_hack, bigram_person_accident,
        bigram_code_bank, bigram_urgent_bank, bigram_employee_code, high_frequency_bigrams,
        word_count, avg_sentence_length, exclamation_count, repetition_rate
    ], dtype=np.float32)


def predict_risk_by_neural(text):
    if neural_model is None or scaler is None:
        return "Средний", [0.33, 0.34, 0.33]

    features = extract_neural_features(text)
    features_scaled = scaler.transform([features])
    features_tensor = torch.FloatTensor(features_scaled)

    with torch.no_grad():
        outputs = neural_model(features_tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        predicted_class = np.argmax(probs)

    risk_names = {0: "Низкий", 1: "Средний", 2: "Высокий"}
    return risk_names[predicted_class], probs.tolist()


def to_wav_if_needed(input_path):
    if str(input_path).endswith('.wav'):
        return str(input_path)

    wav_path = str(input_path).rsplit('.', 1)[0] + '_temp.wav'
    audio = AudioSegment.from_file(str(input_path))
    audio.export(wav_path, format='wav')
    return wav_path


def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    wav_path = to_wav_if_needed(audio_path)

    try:
        with sr.AudioFile(wav_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language="ru-RU")
            return text.strip()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"Ошибка API: {e}")
        return ""
    except Exception as e:
        print(f"Ошибка: {e}")
        return ""
    finally:
        if wav_path != str(audio_path) and os.path.exists(wav_path):
            os.remove(wav_path)


def get_duration(file_path):
    with wave.open(file_path, 'rb') as w:
        return w.getnframes() / w.getframerate()


def prosodic_characteristics(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1).tolist()
    mfcc_std = np.std(mfcc, axis=1).tolist()

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    spectral_contrast = np.mean(contrast, axis=1).tolist()

    zcr = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate = np.mean(zcr).tolist()

    return {
        'mfcc_mean': mfcc_mean,
        'mfcc_std': mfcc_std,
        'spectral_contrast': spectral_contrast,
        'zero_crossing_rate': zero_crossing_rate
    }


def comparison(profile):
    response = supabase.table('Просодический_профиль').select('*').execute()
    fraudsters = response.data

    best_match = None
    best_similarity = 0

    for fraudster in fraudsters:
        fraudster_profile = {
            'mfcc_mean': np.array(fraudster['mfcc_mean']),
            'mfcc_std': np.array(fraudster['mfcc_std']),
            'spectral_contrast': np.array(fraudster.get('spectral_constrast', fraudster.get('spectral_contrast', []))),
            'zero_crossing_rate': np.array(fraudster['zero_crossing_rate'])
        }

        dist1 = np.linalg.norm(np.array(profile['mfcc_mean']) - fraudster_profile['mfcc_mean'])
        dist2 = np.linalg.norm(np.array(profile['mfcc_std']) - fraudster_profile['mfcc_std'])
        dist3 = np.linalg.norm(np.array(profile['spectral_contrast']) - fraudster_profile['spectral_contrast'])
        dist4 = np.linalg.norm(np.array(profile['zero_crossing_rate']) - fraudster_profile['zero_crossing_rate'])

        total_distance = dist1 + dist2 + dist3 + dist4
        similarity = max(0, min(100, 100 - (total_distance / 200 * 100)))

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = fraudster.get('Мошенник')

    return best_match, best_similarity


def generate_recommendation(risk_level):
    if risk_level == "Высокий":
        return '''[color=ff0000][b]ВЫСОКИЙ РИСК МОШЕННИЧЕСТВА![/b][/color]
[color=000000][b]Что делать прямо сейчас:[/b]
1. [color=ff0000][b]НЕЗАМЕДЛИТЕЛЬНО[/b][/color] прекратите разговор и положите трубку.
2. [color=ff0000][b]НЕ[/b][/color] называйте никакие коды из SMS.
3. [color=ff0000][b]НЕ[/b][/color] переводите деньги на какие-либо счета.
4. Перезвоните в банк по официальному номеру.[/color]'''

    elif risk_level == "Средний":
        return '''[color=ffd700][b]СРЕДНИЙ РИСК МОШЕННИЧЕСТВА[/b][/color]
[color=000000][b]Рекомендации:[/b]
1. Будьте бдительны, [color=ffd700][b]не называйте[/b][/color] коды из SMS.
2. Если собеседник торопит и угрожает — это мошенник.
3. Прервите разговор и перезвоните в банк сами.[/color]'''

    else:
        return '''[color=008000][b]НИЗКИЙ РИСК МОШЕННИЧЕСТВА[/b][/color]
[color=000000][b]Правила безопасности:[/b]
1. Никогда [color=008000][b]не называйте[/b][/color] коды из SMS.
2. [color=008000][b]Не переводите деньги[/b][/color] незнакомцам.
3. При сомнении - положите трубку и перезвоните в банк по официальному номеру.[/color]'''


def keep_supabase_awake():
    try:
        supabase.table('Аудиофайл').select('count').execute()
        print("Supabase keep-alive successful")
    except Exception as e:
        print(f"Supabase keep-alive error: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    keep_supabase_awake()
    return jsonify({'status': 'ok'})


@app.route('/keepalive', methods=['GET'])
def keepalive():
    keep_supabase_awake()
    return jsonify({'status': 'supabase pinged'})


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        text = transcribe_audio(tmp_path)

        if not text.strip():
            os.unlink(tmp_path)
            return jsonify({'error': 'Не удалось распознать текст'}), 400

        risk_level, probabilities = predict_risk_by_neural(text)

        text_lower = text.lower()
        word_count = len(text_lower.split())

        hack_count = sum(1 for m in hacking_markers if m in text_lower)
        bank_count = sum(1 for m in bank_markers if m in text_lower)
        employee_count = sum(1 for m in employee_markers if m in text_lower)
        person_count = sum(1 for m in person_markers if m in text_lower)
        accident_count = sum(1 for m in accident_markers if m in text_lower)
        urgent_count = sum(1 for m in urgent_markers if m in text_lower)
        code_count = sum(1 for m in code_markers if m in text_lower)

        markers_count = hack_count + bank_count + employee_count + person_count + accident_count + urgent_count + code_count
        markers_density = int((markers_count / word_count * 100)) if word_count > 0 else 0

        duration = int(get_duration(tmp_path))
        filename = audio_file.filename
        analysis_date = datetime.now().isoformat()

        recommendation = generate_recommendation(risk_level)

        if supabase is not None:
            try:
                keep_supabase_awake()

                audio_response = supabase.table('Аудиофайл').insert({
                    'Имя_файла': filename,
                    'Длительность': duration,
                    'Дата_анализа': analysis_date
                }).execute()

                audio_id = audio_response.data[0]['ИН_аудиофайла']

                supabase.table('Текст').insert({
                    'Объем_текста': word_count,
                    'Количество_маркеров': markers_count,
                    'Плотность_маркеров': markers_density,
                    'Риск_мошенничества': risk_level,
                    'Аудиофайл': audio_id
                }).execute()

                profile = prosodic_characteristics(tmp_path)

                profile_response = supabase.table('Просодический_профиль').insert({
                    'mfcc_mean': profile['mfcc_mean'],
                    'mfcc_std': profile['mfcc_std'],
                    'spectral_constrast': profile['spectral_contrast'],
                    'zero_crossing_rate': profile['zero_crossing_rate'],
                    'Аудиофайл': audio_id
                }).execute()

                profile_id = profile_response.data[0]['ИН_профиля']

                fraudster_id, similarity = comparison(profile)

                if fraudster_id and similarity >= 60:
                    supabase.table('Совпадение').insert({
                        'Процент_совпадения': int(similarity),
                        'Аудиофайл': audio_id,
                        'Просодический_профиль': profile_id,
                        'Мошенник': fraudster_id
                    }).execute()

                    supabase.table('Мошенник').update({
                        'Количество_обращений':
                            supabase.table('Мошенник').select('Количество_обращений').eq('ИН_мошенника',
                                                                                         fraudster_id).execute().data[
                                0]['Количество_обращений'] + 1,
                        'Последнее_обращение': analysis_date
                    }).eq('ИН_мошенника', fraudster_id).execute()

                    recommendation += f'\n\n[size=14][color=ff0000]Совпадение с мошенником: {similarity:.1f}%[/color][/size]'
                else:
                    recommendation += f'\n\n[size=14][color=008000]Совпадений с базой мошенников не обнаружено[/color][/size]'

            except Exception as db_error:
                print(f"Database error: {db_error}")
                recommendation += f'\n\n[size=14][color=ffd700]Примечание: не удалось сохранить данные в базу[/color][/size]'

        return jsonify({
            'success': True,
            'text': text,
            'result_content': recommendation,
            'risk_level': risk_level,
            'neural_probabilities': {
                'low': probabilities[0],
                'medium': probabilities[1],
                'high': probabilities[2]
            },
            'word_count': word_count,
            'markers_count': markers_count,
            'markers_density': markers_density
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
