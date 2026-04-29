from flask import Flask, request, jsonify
from flask_cors import CORS
import vosk
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

app = Flask(__name__)
CORS(app)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_PATH = "vosk-model-small-ru-0.22"
model = None


def vosk_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            urllib.request.urlretrieve(
                "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip", "model.zip")
            with zipfile.ZipFile("model.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove("model.zip")
        model = vosk.Model(MODEL_PATH)
    return model


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
            'spectral_contrast': np.array(fraudster['spectral_contrast']),
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


def text_analysis(current_audio):
    try:
        wf = wave.open(current_audio, 'rb')
        model = vosk_model()
        rec = vosk.KaldiRecognizer(model, wf.getframerate())

        text = ""
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text += result.get('text', '') + " "

        final = json.loads(rec.FinalResult())
        text += final.get('text', '')
        wf.close()

        if not text.strip():
            return None, 'Не удалось распознать текст.', None, 0, 0, 0

    except Exception as e:
        return None, f'Ошибка: {e}', None, 0, 0, 0

    text_lower = text.lower()
    word_count = len(text_lower.split())

    hack_danger = sum(m in text_lower for m in hacking_markers)
    bank_danger = sum(m in text_lower for m in bank_markers)
    employee_danger = sum(m in text_lower for m in employee_markers)
    person_danger = sum(m in text_lower for m in person_markers)
    urgent_danger = sum(m in text_lower for m in urgent_markers)
    code_danger = sum(m in text_lower for m in code_markers)
    accident_danger = sum(m in text_lower for m in accident_markers)

    markers_count = hack_danger + bank_danger + employee_danger + person_danger + urgent_danger + code_danger + accident_danger
    markers_density = int((markers_count / word_count * 100)) if word_count > 0 else 0

    if code_danger >= 1:
        risk_level = 'Высокий'
        title = '[color=ff0000][b]ВЫСОКИЙ РИСК МОШЕННИЧЕСТВА![/b][/color]'
        content = f'''{title}\n
[color=000000][b]Система распознала сценарий "Сообщите код".\nЧто делать прямо сейчас:[/b]
1. [color=ff0000][b]НЕЗАМЕДЛИТЕЛЬНО[/b][/color] прекратите разговор и положите трубку.
2. [color=ff0000][b]НЕ[/b][/color] называйте никакие коды из SMS или push-уведомлений.
3. Если вы [color=ff0000][b]УЖЕ СООБЩИЛИ[/b][/color] код - заблокируйте карту через мобильное приложение.
4. Перезвоните в ваш банк по официальному номеру, указанному на обороте карты.[/color]'''

    elif (employee_danger >= 1 and hack_danger >= 1 and bank_danger >= 1 and urgent_danger >= 1) or (
            hack_danger >= 1 and bank_danger >= 1 and urgent_danger >= 1) or (
            employee_danger >= 1 and (hack_danger >= 1 or bank_danger >= 1)) or (hack_danger >= 1 and bank_danger >= 1):
        risk_level = 'Высокий'
        title = '[color=ff0000][b]ВЫСОКИЙ РИСК МОШЕННИЧЕСТВА![/b][/color]'
        content = f'''{title}\n
[color=000000][b]Система распознала сценарий "Финансы под угрозой".\nЧто делать прямо сейчас:[/b]
1. [color=ff0000][b]НЕЗАМЕДЛИТЕЛЬНО[/b][/color] прекратите разговор и положите трубку.
2. [color=ff0000][b]НЕ[/b][/color] переводите деньги на какие-либо счета.
3. Если вы [color=ff0000][b]УЖЕ ПЕРЕВЕЛИ[/b][/color] деньги - обратитесь в полицию.
4. Перезвоните в ваш банк по официальному номеру, указанному на обороте карты.[/color]'''

    elif (person_danger >= 1 and accident_danger >= 1 and bank_danger >= 1 and urgent_danger >= 1) or (
            employee_danger >= 1 and person_danger >= 1 and accident_danger >= 1 and bank_danger >= 1 and urgent_danger >= 1):
        risk_level = 'Высокий'
        title = '[color=ff0000][b]ВЫСОКИЙ РИСК МОШЕННИЧЕСТВА![/b][/color]'
        content = f'''{title}\n
[color=000000][b]Система распознала сценарий "Родственник в беде".\nЧто делать прямо сейчас:[/b]
1. [color=ff0000][b]Прервите[/b][/color] разговор и [color=ff0000][b]перезвоните[/b][/color] родственнику на его личный номер.
2. [color=ff0000][b]Не переводите деньги[/b][/color] незнакомцам, даже если представляются родственниками.
3. Помните: сотрудник полиции/СК [color=ff0000][b]никогда не потребует[/b][/color] перевода денег для освобождения вашего родственника.'''

    elif bank_danger >= 1 or hack_danger >= 1:
        risk_level = 'Средний'
        title = '[color=ffd700][b]СРЕДНИЙ РИСК МОШЕННИЧЕСТВА![/b][/color]'
        content = f'''{title}\n
[color=000000][b]Рекомендации:[/b]
1. Будьте бдительны и [color=ffd700][b]не называйте[/b][/color] никакие коды из SMS.
2. Если собеседник [color=ffd700][b]торопит и угрожает блокировкой[/b][/color] - это мошенник.
3. Если сомневаетесь, прервите разговор и перезвоните в банк самостоятельно по официальному номеру.'''

    else:
        risk_level = 'Низкий'
        title = '[color=008000][b]НИЗКИЙ РИСК МОШЕННИЧЕСТВА[/b][/color]'
        content = f'''{title}\n
[color=000000][b]Правила безопасности при телефонном разговоре:[/b]
1. Никогда [color=008000][b]не называйте[/b][/color] коды из SMS.
2. [color=008000][b]Не переводите деньги[/b][/color] незнакомцам, даже если представляются родственниками.
3. При любом сомнении - [color=008000][b]положите трубку[/b][/color] и перезвоните в банк сами.'''

    return text, content, risk_level, word_count, markers_count, markers_density


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

        text, result_content, risk_level, word_count, markers_count, markers_density = text_analysis(tmp_path)

        if text is None:
            os.unlink(tmp_path)
            return jsonify({'error': result_content}), 400

        duration = int(get_duration(tmp_path))
        filename = audio_file.filename
        analysis_date = datetime.now().isoformat()

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

                    result_content += f'\n\n[size=14][color=ff0000]Совпадение с мошенником: {similarity:.1f}%[/color][/size]'
                else:
                    result_content += f'\n\n[size=14][color=008000]Совпадений с базой мошенников не обнаружено[/color][/size]'

            except Exception as db_error:
                import traceback
                print(f"Database error: {db_error}")
                print(f"Traceback: {traceback.format_exc()}")
                result_content += f'\n\n[size=14][color=ffd700]Примечание: не удалось сохранить данные в базу ({str(db_error)})[/color][/size]'

        return jsonify({
            'success': True,
            'text': text,
            'result_content': result_content,
            'risk_level': risk_level,
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
