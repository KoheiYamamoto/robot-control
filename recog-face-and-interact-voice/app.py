# webカメラの映像から顔を検出し、面積を計算。顔を検出する際は、azure face api を用いる。
# 5フレームの間、すべての面積が特定の値X以上であれば、音声認識を起動。音声認識は azure speech services を用いる。
# 音声認識が起動している間も、裏では顔認識は動かし続ける。ただし、音声認識が複数呼び出されることは防ぐ。
import cv2, requests, json, threading, time, azure.cognitiveservices.speech as speechsdk, openai

def chatGPT(input_txt):
    response = openai.ChatCompletion.create(
        engine=DEPLOYMENT_NAME, # engine = "deployment_name"
        messages = [
            {"role":"system","content":"あなたは友達と話すような返答を行ってください。\n構文の特徴は以下の通りです。\n- 一切、敬語を使わない\n- タメ口で話す\n"},
            {"role":"user","content":"明日何しようかな"},
            {"role":"assistant","content":"お天気もいいし、公園でピクニックでもしない？ \nそれとも、映画を観に行くとかどう？"},
            {"role": "user", "content": input_txt},
        ],
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    return response['choices'][0]['message']['content']

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_recognition_language="ja-JP"
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
def stt(): # STT 実行
    speech_recognition_result = speech_recognizer.recognize_once_async().get()
    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")

speech_config.speech_synthesis_voice_name='ja-JP-NanamiNeural'
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
def tts(answer_txt): # TTS 実行
    speech_synthesis_result = speech_synthesizer.speak_text_async(answer_txt).get()
    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        pass
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")

def speech_recognition_thread():
    command = stt()
    if command is None:
        print('no command')        
    else:
        print(command)
        answer = chatGPT(command)
        tts(answer)
    global voice_recognition_flag
    voice_recognition_flag = False
    time.sleep(5)    

cap = cv2.VideoCapture(0) # 0はカメラ情報（内蔵カメラ）
voice_recognition_flag = False
areas = []
while True:
    if cv2.waitKey(1) == 27: # ESC
        break
    ret, frame = cap.read() 
    _, img_encoded = cv2.imencode('.jpg', frame) 
    response = requests.post(face_api_url, params={}, headers=headers, data=img_encoded.tobytes())   
    faces = json.loads(response.content) 
   
    for face in faces:
        left, top, right, bottom = face['faceRectangle']['left'], face['faceRectangle']['top'], face['faceRectangle']['width'], face['faceRectangle']['height']
        area = face['faceRectangle']['width'] * face['faceRectangle']['height']            
        cv2.rectangle(frame, (left, top), (left + right, top + bottom), (0, 255,0), 3) 
        if area > 10000:
            cv2.rectangle(frame, (left, top), (left + right, top + bottom), (0, 0, 255), 3) 
        cv2.putText(frame, 'area: '+ str(area), (left, top - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255,0), 2, cv2.LINE_AA) 
        if voice_recognition_flag: 
            cv2.putText(frame, 'Face Detected!', (int(frame.shape[1]/2) - 100, int(frame.shape[0]/4)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA) 
            cv2.putText(frame, 'Say something to AI', (int(frame.shape[1]/2) - 100, int(frame.shape[0]/4)+50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA) 
            
        areas.append(area)
        if len(areas) > 5:
            areas.pop(0)
        
        
        if len(areas) == 5 and all([a > 10000 for a in areas]):
            if not voice_recognition_flag:
                voice_recognition_flag = True
                print('start voice recognition')
                speech_thread = threading.Thread(target=speech_recognition_thread)
                speech_thread.start()

    cv2.imshow('frame', frame) 
cap.release() 
cv2.destroyAllWindows() 
