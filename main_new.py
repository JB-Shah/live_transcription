# import queue
# import sys
# import gradio as gr
# import threading
# import io
# from google.cloud import speech, translate_v2 as translate
# import pyaudio
# import wave
# from pyannote.audio import Pipeline
# from pydub import AudioSegment

# # Audio recording parameters
# RATE = 16000
# CHUNK = int(RATE / 10)  # 100ms

# # Initialize Google Cloud Translate client
# translate_client = translate.Client()

# # Global flag to control stop function
# stop_recording = False

# # MicrophoneStream class to stream and save audio
# class MicrophoneStream:
#     def __init__(self, rate=RATE, chunk=CHUNK):
#         self._rate = rate
#         self._chunk = chunk
#         self._buff = queue.Queue()
#         self.closed = True
#         self.frames = []

#     def __enter__(self):
#         self._audio_interface = pyaudio.PyAudio()
#         self._audio_stream = self._audio_interface.open(
#             format=pyaudio.paInt16,
#             channels=1,
#             rate=self._rate,
#             input=True,
#             frames_per_buffer=self._chunk,
#             stream_callback=self._fill_buffer,
#         )
#         self.closed = False
#         return self

#     def __exit__(self, type, value, traceback):
#         self._audio_stream.stop_stream()
#         self._audio_stream.close()
#         self.closed = True
#         self._buff.put(None)
#         self._audio_interface.terminate()

#         # Save to file
#         wf = wave.open("recorded_audio.wav", 'wb')
#         wf.setnchannels(1)
#         wf.setsampwidth(self._audio_interface.get_sample_size(pyaudio.paInt16))
#         wf.setframerate(self._rate)
#         wf.writeframes(b''.join(self.frames))
#         wf.close()

#     def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
#         self._buff.put(in_data)
#         self.frames.append(in_data)
#         return None, pyaudio.paContinue

#     def generator(self):
#         while not self.closed:
#             if stop_recording:
#                 return
#             chunk = self._buff.get()
#             if chunk is None:
#                 return
#             data = [chunk]
#             while True:
#                 try:
#                     chunk = self._buff.get(block=False)
#                     if chunk is None:
#                         return
#                     data.append(chunk)
#                 except queue.Empty:
#                     break
#             yield b"".join(data)

# # Translator helper
# def translate_text(text, target_language="mr"):
#     if not text.strip():
#         return ""
#     translation = translate_client.translate(text, target_language=target_language)
#     return translation["translatedText"]

# # Real-time transcription
# def live_transcription(input_language):
#     global stop_recording
#     stop_recording = False

#     language_map = {"Marathi": "mr-IN", "Hindi": "hi-IN", "English": "en-US"}
#     selected_language = language_map.get(input_language, "mr-IN")

#     client = speech.SpeechClient()
#     config = speech.RecognitionConfig(
#         encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#         sample_rate_hertz=RATE,
#         language_code=selected_language,
#     )

#     streaming_config = speech.StreamingRecognitionConfig(
#         config=config, interim_results=True
#     )

#     with MicrophoneStream(RATE, CHUNK) as stream:
#         audio_generator = stream.generator()
#         requests = (
#             speech.StreamingRecognizeRequest(audio_content=content)
#             for content in audio_generator
#         )
#         responses = client.streaming_recognize(streaming_config, requests)

#         for response in responses:
#             if stop_recording:
#                 break

#             if not response.results:
#                 continue

#             result = response.results[0]
#             if not result.alternatives:
#                 continue

#             transcript = result.alternatives[0].transcript
#             translated_text = translate_text(transcript)
#             yield translated_text

# # Diarization with transcription + translation
# def diarize_audio(file_path="recorded_audio.wav", target_language="mr"):
#     """Performs diarization + transcription + translation for each speaker segment."""
#     pipeline = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization",
#         use_auth_token=""
#     )
#     diarization = pipeline(file_path)

#     audio = AudioSegment.from_wav(file_path)git
#     client = speech.SpeechClient()
#     diarized_output = ""

#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#         start_ms = int(turn.start * 1000)
#         end_ms = int(turn.end * 1000)

#         segment = audio[start_ms:end_ms]
#         audio_chunk = io.BytesIO()
#         segment.export(audio_chunk, format="wav")
#         audio_chunk.seek(0)

#         audio_content = audio_chunk.read()
#         audio_sample = speech.RecognitionAudio(content=audio_content)
#         config = speech.RecognitionConfig(
#             encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#             sample_rate_hertz=16000,
#             language_code="en-US",  # Assuming transcription in English
#         )

#         response = client.recognize(config=config, audio=audio_sample)
#         transcript = ""
#         for result in response.results:
#             transcript += result.alternatives[0].transcript

#         # Translate the transcript — log if it's empty
#         if transcript.strip():
#             translated = translate_text(transcript, target_language)
#         else:
#             translated = "[No speech detected]"

#         # Logging to help debug
#         print(f"[{round(turn.start, 1)}s - {round(turn.end, 1)}s] {speaker}")
#         print(f"Transcript: {transcript}")
#         print(f"Translated: {translated}\n")

#         diarized_output += f"[{round(turn.start, 1)}s - {round(turn.end, 1)}s] {speaker}: {translated}\n"

#     return diarized_output

# # Stop button handler
# def stop_transcription():
#     global stop_recording
#     stop_recording = True
#     return (
#         gr.update(value="✅ Recording stopped. Now you can generate diarization.", interactive=True),
#         gr.update(visible=True),  # Show diarization button
#         gr.update(visible=False)  # Hide save until diarization
#     )

# # Diarization button handler
# def generate_diarization(language):
#     language_map = {"Marathi": "mr", "Hindi": "hi", "English": "en"}
#     target_lang = language_map.get(language, "mr")
#     diarized_result = diarize_audio("recorded_audio.wav", target_language=target_lang)
#     return diarized_result, gr.update(visible=True)

# # Save button handler
# def save_to_file(text):
#     with open("translated_text.txt", "w", encoding="utf-8") as file:
#         file.write(text)
#     return "✅ Text saved to 'translated_text.txt'!"

# # Gradio UI
# with gr.Blocks() as app:
#     gr.Markdown("## 🎤 Live Speech Transcription & Translation to Marathi with Speaker Diarization")

#     language_selector = gr.Dropdown(
#         ["Marathi", "Hindi", "English"], label="Select Input Language", value="Marathi"
#     )

#     output_text = gr.Textbox(
#         label="Translated Text with Speaker Diarization",
#         interactive=False,
#         lines=10,
#         max_lines=20,
#         show_copy_button=True,
#     )

#     start_button = gr.Button("🎙 Start Recording")
#     stop_button = gr.Button("🛑 Stop Recording")
#     diarize_button = gr.Button("🧠 Generate Diarization", visible=False)
#     save_button = gr.Button("💾 Save to File", visible=False)

#     start_button.click(live_transcription, inputs=[language_selector], outputs=[output_text])
#     stop_button.click(stop_transcription, outputs=[output_text, diarize_button, save_button])
#     diarize_button.click(generate_diarization, inputs=[language_selector], outputs=[output_text, save_button])
#     save_button.click(save_to_file, inputs=[output_text], outputs=[output_text])

# app.launch()

#  Added diarization

import queue
import sys
import gradio as gr
import threading
import io
from google.cloud import speech, translate_v2 as translate
import pyaudio
import wave
from pyannote.audio import Pipeline
from pydub import AudioSegment

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

translate_client = translate.Client()

stop_recording = False  # Global flag to control stop

class MicrophoneStream:
    def __init__(self, rate=RATE, chunk=CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True
        self.frames = []

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

        wf = wave.open("recorded_audio.wav", 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self._audio_interface.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self._rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        self.frames.append(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            if stop_recording:
                return
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)

def translate_text(text, target_language="mr"):
    if not text.strip():
        return ""
    translation = translate_client.translate(text, target_language=target_language)
    return translation["translatedText"]

def live_transcription(input_language):
    global stop_recording
    stop_recording = False

    language_map = {"Marathi": "mr-IN", "Hindi": "hi-IN", "English": "en-US"}
    selected_language = language_map.get(input_language, "mr-IN")

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=selected_language,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    full_translated_text = ""  # Running transcript

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )
        responses = client.streaming_recognize(streaming_config, requests)

        for response in responses:
            if stop_recording:
                break

            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript.strip()

            if result.is_final and transcript:
                translated = translate_text(transcript)
                full_translated_text += f"{translated}\n"
                yield full_translated_text

def diarize_audio(file_path="recorded_audio.wav", target_language="mr"):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=""
    )
    diarization = pipeline(file_path)

    audio = AudioSegment.from_wav(file_path)
    client = speech.SpeechClient()
    diarized_output = ""

    language_code_map = {"mr": "mr-IN", "hi": "hi-IN", "en": "en-US"}
    selected_lang_code = language_code_map.get(target_language, "mr-IN")

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_ms = int(turn.start * 1000)
        end_ms = int(turn.end * 1000)

        segment = audio[start_ms:end_ms]
        audio_chunk = io.BytesIO()
        segment.export(audio_chunk, format="wav")
        audio_chunk.seek(0)

        audio_content = audio_chunk.read()
        audio_sample = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=selected_lang_code,
        )

        response = client.recognize(config=config, audio=audio_sample)
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript

        if transcript.strip():
            translated = translate_text(transcript, target_language)
        else:
            translated = "[No speech detected]"

        diarized_output += f"[{round(turn.start, 1)}s - {round(turn.end, 1)}s] {speaker}: {translated}\n"

    return diarized_output

def stop_transcription():
    global stop_recording
    stop_recording = True
    return (
        gr.update(value="✅ Recording stopped. Now you can generate diarization.", interactive=True),
        gr.update(visible=True),  # Show diarization button
        gr.update(visible=False)  # Hide save until diarization
    )

def generate_diarization(language):
    language_map = {"Marathi": "mr", "Hindi": "hi", "English": "en"}
    target_lang = language_map.get(language, "mr")
    diarized_result = diarize_audio("recorded_audio.wav", target_language=target_lang)
    return diarized_result, gr.update(visible=True)

def save_to_file(text):
    with open("translated_text.txt", "w", encoding="utf-8") as file:
        file.write(text)
    return "✅ Text saved to 'translated_text.txt'!"

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("## 🎤 Live Speech Transcription & Translation to Marathi with Speaker Diarization")

    language_selector = gr.Dropdown(
        ["Marathi", "Hindi", "English"], label="Select Input Language", value="Marathi"
    )

    output_text = gr.Textbox(
        label="Translated Text with Speaker Diarization",
        interactive=False,
        lines=10,
        max_lines=20,
        show_copy_button=True,
    )

    start_button = gr.Button("🎙 Start Recording")
    stop_button = gr.Button("🛑 Stop Recording")
    diarize_button = gr.Button("🧠 Generate Diarization", visible=False)
    save_button = gr.Button("💾 Save to File", visible=False)

    start_button.click(live_transcription, inputs=[language_selector], outputs=[output_text])
    stop_button.click(stop_transcription, outputs=[output_text, diarize_button, save_button])
    diarize_button.click(generate_diarization, inputs=[language_selector], outputs=[output_text, save_button])
    save_button.click(save_to_file, inputs=[output_text], outputs=[output_text])

app.launch()
