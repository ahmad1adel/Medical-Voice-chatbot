import time
from flask import Flask, render_template, send_from_directory, request, jsonify
from dotenv import load_dotenv

load_dotenv()
from flask_socketio import SocketIO, emit
import logging
import base64
import io
import tempfile
import os
import asyncio
from azure.cognitiveservices.speech import SpeechConfig, AudioConfig, SpeechRecognizer, SpeechSynthesizer, ResultReason, SpeechSynthesisOutputFormat, AutoDetectSourceLanguageConfig, SourceLanguageConfig, PropertyId # noqa
from openai import AzureOpenAI
from pydub import AudioSegment
from deep_translator import GoogleTranslator
import threading
import re
from PIL import Image
import requests
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes, VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from io import BytesIO

app = Flask(__name__)
# Update your Socket.IO initialization
socketio = SocketIO(app, 
                   cors_allowed_origins="*",  # Temporarily allow all for testing
                   async_mode='gevent',  # Explicitly set async mode
                   engineio_logger=True,  # Enable engineio logging
                   logger=True)  # Enable Socket.IO logging

logging.basicConfig(level=logging.DEBUG)

# Azure Speech-to-Text Configuration
speech_key = os.getenv("AZURE_SPEECH_KEY")
service_region = os.getenv("AZURE_SPEECH_REGION")
speech_config = SpeechConfig(subscription=speech_key, region=service_region)

# Azure OpenAI Configuration
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2023-05-15"
)

# Azure Computer Vision Configuration
vision_key = os.getenv("AZURE_VISION_KEY")
vision_endpoint = os.getenv("AZURE_VISION_ENDPOINT")
vision_client = ComputerVisionClient(vision_endpoint, CognitiveServicesCredentials(vision_key))

# Supported languages and their voices
languages = {
    "ar-SA": "ar-SA-HamedNeural",  # Arabic (Saudi)
    "en-US": "en-US-JennyNeural",  # English (US)
    "fr-FR": "fr-FR-DeniseNeural",  # French
    "es-ES": "es-ES-ElviraNeural",  # Spanish
    "de-DE": "de-DE-KatjaNeural",   # German
    "it-IT": "it-IT-ElsaNeural",    # Italian
    "pt-BR": "pt-BR-FranciscaNeural", # Portuguese
    "ru-RU": "ru-RU-DariyaNeural",  # Russian
    "ja-JP": "ja-JP-NanamiNeural",  # Japanese
    "ko-KR": "ko-KR-SunHiNeural",   # Korean
    "zh-CN": "zh-CN-XiaoxiaoNeural" # Chinese
}

# Create a thread pool for audio processing
audio_thread_pool = []

def cleanup_audio_threads():
    """Clean up completed audio processing threads."""
    global audio_thread_pool
    audio_thread_pool = [t for t in audio_thread_pool if t.is_alive()]

def is_arabic_text(text):
    """Check if the text contains Arabic characters."""
    arabic_range = re.compile('[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return bool(arabic_range.search(text))

def detect_language(text):
    """Detect the language of the text."""
    try:
        from langdetect import detect
        lang_code = detect(text)
        # Map detected language to supported language code
        lang_map = {
            'ar': 'ar-SA',
            'en': 'en-US',
            'fr': 'fr-FR',
            'es': 'es-ES',
            'de': 'de-DE',
            'it': 'it-IT',
            'pt': 'pt-BR',
            'ru': 'ru-RU',
            'ja': 'ja-JP',
            'ko': 'ko-KR',
            'zh': 'zh-CN'
        }
        return lang_map.get(lang_code, 'en-US')  # Default to English if language not supported
    except Exception as e:
        print(f"Language detection error: {str(e)}")
        return 'en-US'  # Default to English on error

def process_audio_chunk(audio_data, sid, language=None, voice=None):
    """Process audio chunk in a separate thread."""
    try:
        print("\n=== Starting Audio Processing ===")
        # Decode the base64 audio chunk
        audio_stream = io.BytesIO(audio_data)
        audio = AudioSegment.from_file(audio_stream, format="webm")
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

        # Create a temporary file that won't be automatically deleted
        temp_wav_path = os.path.join(tempfile.gettempdir(), f"audio_{sid}_{int(time.time())}.wav")

        try:
            # Export audio to the temporary file
            audio.export(temp_wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])

            # Configure Speech-to-Text
            print("\n--- Speech Recognition Configuration ---")
            audio_config = AudioConfig(filename=temp_wav_path)
            
            # Use selected language or default to English
            selected_language = language or "en-US"
            selected_voice = voice or "en-US-JennyNeural"
            
            print(f"Using selected language: {selected_language}")
            print(f"Using selected voice: {selected_voice}")
            
            # Configure speech recognition with selected language
            speech_config.speech_recognition_language = selected_language
            recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            
            print("Attempting speech recognition...")
            result = recognizer.recognize_once()
            
            if result.reason == ResultReason.RecognizedSpeech:
                user_input = result.text.strip()
                print(f"\n=== Speech Recognition Results ===")
                print(f"Detected Language: {selected_language}")
                print(f"Recognized Text: {user_input}")

                # Send immediate processing status
                socketio.emit('processing', {'status': 'processing'}, room=sid)

                # Chatbot Response
                print("\n--- Getting Chatbot Response ---")
                chatbot_response = get_chatbot_response(user_input, selected_language)
                print(f"Chatbot Response: {chatbot_response}")

                # Text-to-Speech
                print("\n--- Text-to-Speech Processing ---")
                print(f"Using TTS Voice: {selected_voice}")
                speech_config.speech_synthesis_voice_name = selected_voice
                speech_config.set_speech_synthesis_output_format(SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)

                # Create synthesizer without audio output config
                synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=None)
                tts_result = synthesizer.speak_text_async(chatbot_response).get()

                if tts_result.reason == ResultReason.SynthesizingAudioCompleted:
                    print("Text-to-Speech completed successfully")
                    audio_data = tts_result.audio_data
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    socketio.emit('chatbot_response',
                                {'text': chatbot_response, 'audio': audio_base64, 'language': selected_language},
                                room=sid)
                else:
                    print(f"TTS Failed: {tts_result.reason}")
                    logging.error(f"TTS failed: {tts_result.reason}, Details: {tts_result.error_details}")
                    socketio.emit('error', {'message': 'Speech synthesis failed.'}, room=sid)
            else:
                print(f"Recognition failed: {result.reason}")
                socketio.emit('error', {'message': 'Speech recognition failed.'}, room=sid)

            # Clean up
            recognizer = None
            audio_stream.close()

        finally:
            # Ensure the temporary file is deleted after processing
            try:
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)
            except Exception as e:
                logging.error(f"Error deleting temporary file: {e}")

    except Exception as e:
        print(f"\nError in process_audio_chunk: {str(e)}")
        logging.error(f"Error processing audio chunk: {e}")
        socketio.emit('error', {'message': str(e)}, room=sid)

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    try:
        # Decode the base64 audio chunk
        audio_data = base64.b64decode(data['chunk'])
        
        # Get selected language and voice from the data
        selected_language = data.get('language', 'en-US')
        selected_voice = data.get('voice', 'en-US-JennyNeural')

        # Clean up old threads
        cleanup_audio_threads()

        # Start new thread for audio processing with selected language and voice
        thread = threading.Thread(
            target=process_audio_chunk,
            args=(audio_data, request.sid),
            kwargs={'language': selected_language, 'voice': selected_voice}
        )
        thread.daemon = True
        thread.start()
        audio_thread_pool.append(thread)

    except Exception as e:
        logging.error(f"Error handling audio chunk: {e}")
        emit('error', {'message': str(e)})


def get_tts_voice(language):
    """Select appropriate TTS voice based on detected language."""
    # Extract base language code (e.g., 'ar' from 'ar-SA')
    base_language = language.split('-')[0]
    return languages.get(language, languages.get(base_language, languages["en-US"]))


def translate_text(text, source_lang, target_lang):
    """Translate text using Google Translator."""
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon') # noqa


@socketio.on('connect')
def handle_connect():
    logging.debug("Client connected.")


def get_chatbot_response(user_input, detected_language):
    """Handle chatbot response with Arabic translation when needed."""
    try:
        print("\n=== Chatbot Processing ===")
        print(f"Input Language: {detected_language}")
        print(f"Original Input: {user_input}")

        # If Arabic is detected, translate to English for GPT
        if detected_language.startswith("ar"):
            print("Translating Arabic input to English...")
            try:
                user_input = translate_text(user_input, 'ar', 'en')
                print(f"Translated Input: {user_input}")
            except Exception as e:
                print(f"Translation error (Arabic to English): {str(e)}")
                return "عذرًا، حدث خطأ في الترجمة."

        # Always use English system message for consistency
        system_message = "You are a medical chatbot. Keep responses concise and focused on health conditions."
        
        print("\nSending to GPT...")
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ],
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.5
        )
        gpt_response = response.choices[0].message.content.strip()
        print(f"GPT Response: {gpt_response}")

        # If original input was Arabic, translate response back to Arabic
        if detected_language.startswith("ar"):
            print("\nTranslating response back to Arabic...")
            try:
                gpt_response = translate_text(gpt_response, 'en', 'ar')
                print(f"Translated Response: {gpt_response}")
                
                # Verify the translation contains Arabic characters
                if not any('\u0600' <= char <= '\u06FF' for char in gpt_response):
                    raise Exception("Translation does not contain Arabic characters")
                
                print(f"Final Arabic Response: {gpt_response}")
            except Exception as e:
                print(f"Translation error (English to Arabic): {str(e)}")
                return "عذرًا، حدث خطأ في الترجمة."
        
        print(f"\nFinal Response Language: {detected_language}")
        print(f"Final Response: {gpt_response}")
        return gpt_response
    except Exception as e:
        print(f"\nError in get_chatbot_response: {str(e)}")
        logging.error(f"Error generating chatbot response: {e}")
        return "عذرًا، حدث خطأ ما." if detected_language.startswith("ar") else "Sorry, an error occurred."


@socketio.on('end_recording')
def handle_end_recording(data):
    try:
        language = data.get('language', 'en-US')
        voice = data.get('voice', 'en-US-JennyNeural')
        request_summary = data.get('requestSummary', False)
        
        if request_summary:
            # Generate summary using GPT-4
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical assistant. Provide a concise summary of the conversation."},
                    {"role": "user", "content": "Please summarize our conversation."}
                ],
                max_tokens=50
            )
            
            summary_text = response.choices[0].message.content
            
            # Convert summary to speech
            speech_config = SpeechConfig(subscription=speech_key, region=service_region)
            speech_config.speech_synthesis_voice_name = voice
            synthesizer = SpeechSynthesizer(speech_config=speech_config)
            result = synthesizer.speak_text_async(summary_text).get()
            
            if result.reason == ResultReason.SynthesizingAudioCompleted:
                audio_data = base64.b64encode(result.audio_data).decode('utf-8')
                emit('summary', {
                    'text': summary_text,
                    'audio': audio_data
                })
            else:
                emit('error', {'message': 'Error generating summary speech'})
    except Exception as e:
        logging.error(f"Error in end_recording: {str(e)}")
        emit('error', {'message': str(e)})

@socketio.on('chat_message')
def handle_chat_message(data):
    try:
        message = data.get('message', '')
        if not message:
            emit('error', {'message': 'No message provided'})
            return

        # Get response from GPT-4
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical assistant. Provide clear, accurate medical information."},
                {"role": "user", "content": message}
            ],
            max_tokens=50
        )
        
        bot_response = response.choices[0].message.content
        emit('chat_response', {'message': bot_response})
        
    except Exception as e:
        logging.error(f"Error in chat_message: {str(e)}")
        emit('error', {'message': str(e)})

def analyze_medical_image(image_data, prompt=None):
    """
    Analyze a medical image using Azure Computer Vision and GPT-4.
    """
    try:
        # Decode base64 image data
        image_bytes = base64.b64decode(image_data)
        image_stream = BytesIO(image_bytes)
        
        # Analyze image using Azure Computer Vision
        analysis = vision_client.analyze_image_in_stream(
            image_stream,
            visual_features=[
                VisualFeatureTypes.categories,
                VisualFeatureTypes.description,
                VisualFeatureTypes.tags,
                VisualFeatureTypes.objects
            ]
        )
        
        # Extract text and tags for GPT-4
        detected_text = ""
        if hasattr(analysis, 'description') and analysis.description.captions:
            detected_text = analysis.description.captions[0].text
        
        tags = [tag.name for tag in analysis.tags] if hasattr(analysis, 'tags') else []
        categories = [category.name for category in analysis.categories] if hasattr(analysis, 'categories') else []
        
        # Create prompt for GPT-4
        gpt_prompt = f"""Analyze this medical image based on the following information:
        Detected Text: {detected_text}
        Tags: {', '.join(tags)}
        Categories: {', '.join(categories)}
        
        User's Specific Request: {prompt if prompt else 'Provide a general medical analysis'}
        
        Please provide a detailed medical analysis in the following format:
        ### Medical Analysis
        - Detailed interpretation of the image
        - Specific medical findings
        - Potential conditions or abnormalities
        
        ### Detected Conditions
        - List any specific medical conditions detected
        - Include confidence levels for each detection
        
        ### Recommendations
        - Professional medical recommendations
        - Suggested next steps
        - Any precautions or warnings
        
        ### Additional Notes
        - Technical details about the image
        - Quality assessment
        - Limitations of the analysis"""
        
        # Get GPT-4 analysis
        try:
            gpt_response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical imaging expert. Provide detailed, accurate analysis of medical images."},
                    {"role": "user", "content": gpt_prompt}
                ],
                temperature=0.7,
                max_tokens=50
            )
            
            gpt_analysis = gpt_response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in GPT analysis: {str(e)}")
            gpt_analysis = "Error in detailed analysis. Please consult a healthcare professional."
        
        # Combine results
        response = {
            "analysis": gpt_analysis,
            "detected_text": detected_text,
            "tags": tags,
            "categories": categories,
            "confidence_score": analysis.description.captions[0].confidence if hasattr(analysis, 'description') and analysis.description.captions else 0.0
        }
        
        return response
        
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        raise Exception(f"Error analyzing image: {str(e)}")

@socketio.on('analyze_image')
def handle_image_analysis(data):
    try:
        image_data = data.get('image')
        prompt = data.get('prompt')
        
        if not image_data:
            raise ValueError("No image data provided")
            
        analysis_result = analyze_medical_image(image_data, prompt)
        emit('image_analysis', {'analysis': analysis_result['analysis']})
        
    except Exception as e:
        print(f"Error in image analysis: {str(e)}")
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True, port=8000)