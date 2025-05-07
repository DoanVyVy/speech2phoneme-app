from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
import torch
import torchaudio
import librosa
import os
import numpy as np
from pydub import AudioSegment
import tempfile
import soundfile as sf
import io
import re

# Create a directory for the tokenizer files if it doesn't exist
os.makedirs("tokenizer_data", exist_ok=True)

# Load feature extractor and tokenizer for English speech recognition
print("🔁 Đang tải mô hình từ Hugging Face...")
# Sử dụng mô hình tiếng Anh tốt hơn từ Facebook
MODEL_ID = "facebook/wav2vec2-large-960h-lv60-self"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.eval()
print("✅ Mô hình đã sẵn sàng!")

# Bảng ánh xạ phiên âm chuẩn ARPABET sang IPA
# ARPABET là chuẩn được sử dụng rộng rãi cho phiên âm tiếng Anh Mỹ
ARPABET_TO_IPA = {
    'AA': 'ɑ', 'AA0': 'ɑ', 'AA1': 'ˈɑ', 'AA2': 'ˌɑ',
    'AE': 'æ', 'AE0': 'æ', 'AE1': 'ˈæ', 'AE2': 'ˌæ',
    'AH': 'ʌ', 'AH0': 'ə', 'AH1': 'ˈʌ', 'AH2': 'ˌʌ',
    'AO': 'ɔ', 'AO0': 'ɔ', 'AO1': 'ˈɔ', 'AO2': 'ˌɔ',
    'AW': 'aʊ', 'AW0': 'aʊ', 'AW1': 'ˈaʊ', 'AW2': 'ˌaʊ',
    'AY': 'aɪ', 'AY0': 'aɪ', 'AY1': 'ˈaɪ', 'AY2': 'ˌaɪ',
    'B': 'b',
    'CH': 'tʃ',
    'D': 'd',
    'DH': 'ð',
    'EH': 'ɛ', 'EH0': 'ɛ', 'EH1': 'ˈɛ', 'EH2': 'ˌɛ',
    'ER': 'ɝ', 'ER0': 'ɝ', 'ER1': 'ˈɝ', 'ER2': 'ˌɝ',
    'EY': 'eɪ', 'EY0': 'eɪ', 'EY1': 'ˈeɪ', 'EY2': 'ˌeɪ',
    'F': 'f',
    'G': 'ɡ',
    'HH': 'h',
    'IH': 'ɪ', 'IH0': 'ɪ', 'IH1': 'ˈɪ', 'IH2': 'ˌɪ',
    'IY': 'i', 'IY0': 'i', 'IY1': 'ˈi', 'IY2': 'ˌi',
    'JH': 'dʒ',
    'K': 'k',
    'L': 'l',
    'M': 'm',
    'N': 'n',
    'NG': 'ŋ',
    'OW': 'oʊ', 'OW0': 'oʊ', 'OW1': 'ˈoʊ', 'OW2': 'ˌoʊ',
    'OY': 'ɔɪ', 'OY0': 'ɔɪ', 'OY1': 'ˈɔɪ', 'OY2': 'ˌɔɪ',
    'P': 'p',
    'R': 'ɹ',
    'S': 's',
    'SH': 'ʃ',
    'T': 't',
    'TH': 'θ',
    'UH': 'ʊ', 'UH0': 'ʊ', 'UH1': 'ˈʊ', 'UH2': 'ˌʊ',
    'UW': 'u', 'UW0': 'u', 'UW1': 'ˈu', 'UW2': 'ˌu',
    'V': 'v',
    'W': 'w',
    'Y': 'j',
    'Z': 'z',
    'ZH': 'ʒ',
    ' ': ' ',
}

# Bổ sung bảng ánh xạ từ ký tự thường sang IPA cho các trường hợp không rõ ràng
CHAR_TO_IPA = {
    'a': 'æ',
    'b': 'b',
    'c': 'k',
    'd': 'd',
    'e': 'ɛ',
    'f': 'f',
    'g': 'g',
    'h': 'h',
    'i': 'ɪ',
    'j': 'dʒ',
    'k': 'k',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'o': 'ɑ',
    'p': 'p',
    'q': 'k',
    'r': 'ɹ',
    's': 's',
    't': 't',
    'u': 'ʌ',
    'v': 'v',
    'w': 'w',
    'x': 'ks',
    'y': 'j',
    'z': 'z',
    'th': 'θ',
    'sh': 'ʃ',
    'ch': 'tʃ',
    'zh': 'ʒ',
    'ng': 'ŋ',
    ' ': ' ',
}

def load_audio_file(file_path, target_sr=16000):
    """
    Load audio file using multiple fallback methods
    """
    print(f"📂 Loading audio file: {file_path}")
    
    try:
        # Method 1: Try using pydub (handles many formats)
        try:
            audio_segment = AudioSegment.from_file(file_path)
            # Convert to wav format with required sample rate
            audio_segment = audio_segment.set_frame_rate(target_sr)
            
            # Convert to numpy array
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # Convert to mono if stereo
            if audio_segment.channels > 1:
                samples = samples.reshape((-1, audio_segment.channels)).mean(axis=1)
            
            # Normalize
            samples = samples / (2**15) if samples.dtype == np.int16 else samples / np.max(np.abs(samples))
            
            print("✅ Audio loaded using pydub")
            return samples, target_sr
            
        except Exception as e:
            print(f"⚠️ Pydub loading failed: {str(e)}")
            
        # Method 2: Try using torchaudio
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
                
            waveform = waveform.squeeze().numpy()
            print("✅ Audio loaded using torchaudio")
            return waveform, target_sr
            
        except Exception as e:
            print(f"⚠️ Torchaudio loading failed: {str(e)}")
            
        # Method 3: Try using soundfile
        try:
            audio, sr = sf.read(file_path)
            if sr != target_sr:
                # Resample using numpy (simple method)
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            print("✅ Audio loaded using soundfile")
            return audio, target_sr
            
        except Exception as e:
            print(f"⚠️ Soundfile loading failed: {str(e)}")
            
        # Method 4: Last resort, try librosa
        audio, sr = librosa.load(file_path, sr=target_sr)
        print("✅ Audio loaded using librosa")
        return audio, target_sr
        
    except Exception as e:
        raise RuntimeError(f"Could not load audio file {file_path}: {str(e)}")

def extract_words_from_transcript(transcript):
    """
    Trích xuất các từ từ bản phiên âm thô
    """
    # Loại bỏ các ký tự đặc biệt và chuyển thành chữ thường
    cleaned = re.sub(r'[^a-zA-Z\s]', '', transcript).lower()
    # Tách thành các từ
    words = cleaned.split()
    return words

def convert_to_ipa(text):
    """
    Chuyển đổi văn bản thành ký hiệu IPA
    """
    try:
        words = extract_words_from_transcript(text)
        ipa_words = []
        
        for word in words:
            # Chuyển đổi từng từ sang IPA
            ipa_word = ""
            i = 0
            while i < len(word):
                # Kiểm tra các tổ hợp 2 ký tự trước
                if i < len(word) - 1 and word[i:i+2] in CHAR_TO_IPA:
                    ipa_word += CHAR_TO_IPA[word[i:i+2]]
                    i += 2
                # Nếu không, xử lý từng ký tự
                elif word[i] in CHAR_TO_IPA:
                    ipa_word += CHAR_TO_IPA[word[i]]
                    i += 1
                else:
                    ipa_word += word[i]
                    i += 1
            
            ipa_words.append(ipa_word)
        
        # Ghép các từ IPA
        return ' '.join(ipa_words)
    except Exception as e:
        print(f"⚠️ Error converting to IPA: {str(e)}")
        return text

def transcribe(audio_path, output_ipa=True):
    """
    Process audio file and transcribe to phonemes
    """
    # Load audio with better error handling
    audio, sr = load_audio_file(audio_path, target_sr=16000)
    
    # Convert to tensor for model input
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values

    # Run inference
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    
    # Decode to text (English words)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    # Chuyển đổi sang IPA nếu được yêu cầu
    if output_ipa:
        ipa_transcription = convert_to_ipa(transcription)
        return {
            "raw": transcription,
            "ipa": ipa_transcription
        }
    
    return transcription