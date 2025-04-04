import queue
import sys
import gradio as gr
import threading
from google.cloud import speech, translate_v2 as translate
import pyaudio

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

# Initialize Google Cloud Translate client
translate_client = translate.Client()

# Global flag to control stop function
stop_recording = False


class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate=RATE, chunk=CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

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

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        """Generates audio chunks from the stream of audio data in chunks."""
        while not self.closed:
            if stop_recording:  # Stop the generator when stop button is pressed
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
    """Translate text to Marathi using Google Translate API."""
    if not text.strip():
        return ""
    translation = translate_client.translate(text, target_language=target_language)
    return translation["translatedText"]


def live_transcription(input_language):
    """Real-time speech transcription and translation to Marathi."""
    global stop_recording
    stop_recording = False  # Reset stop flag when starting recording

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

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )
        responses = client.streaming_recognize(streaming_config, requests)

        for response in responses:
            if stop_recording:
                break  # Stop listening if user clicks stop

            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript
            translated_text = translate_text(transcript)

            yield translated_text  # Only return the latest translation


def stop_transcription():
    """Stops the live transcription and makes the output editable."""
    global stop_recording
    stop_recording = True
    return gr.update(interactive=True), gr.update(visible=True)  # Make textbox editable and show save button


def save_to_file(text):
    """Saves the translated text to a file."""
    with open("translated_text.txt", "w", encoding="utf-8") as file:
        file.write(text)
    return "✅ Text saved to 'translated_text.txt'!"


# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("## 🎤 Live Speech Transcription & Translation to Marathi")
    
    language_selector = gr.Dropdown(
        ["Marathi", "Hindi", "English"], label="Select Input Language", value="Marathi"
    )

    output_text = gr.Textbox(
        label="Translated Text (Marathi)",
        interactive=False,  # Initially not editable
        lines=10,  # Makes it taller (fits more text)
        max_lines=20,  # Allows scrolling if more lines
        show_copy_button=True,  # Adds a copy button
    )

    start_button = gr.Button("🎙 Start Recording")
    stop_button = gr.Button("🛑 Stop Recording")
    save_button = gr.Button("💾 Save to File", visible=False)  # Initially hidden

    start_button.click(live_transcription, inputs=[language_selector], outputs=[output_text])
    stop_button.click(stop_transcription, outputs=[output_text, save_button])
    save_button.click(save_to_file, inputs=[output_text], outputs=[output_text])

app.launch()
