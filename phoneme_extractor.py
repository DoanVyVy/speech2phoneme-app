import torch
import numpy as np
import librosa
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from model_utils import load_audio_file, transcribe, convert_to_ipa

class PhonemeExtractor:
    def __init__(self):
        """
        Khởi tạo mô hình wav2vec2 để trích xuất phiên âm
        """
        print("🔄 Khởi tạo bộ trích xuất phiên âm trực tiếp...")
        
        # Sử dụng mô hình tiếng Anh đáng tin cậy thay vì mô hình đa ngôn ngữ
        self.model_id = "facebook/wav2vec2-large-960h-lv60-self"
        
        try:
            # Tải mô hình
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
            self.model.eval()
            self.ready = True
            print("✅ Bộ trích xuất phiên âm đã sẵn sàng")
        except Exception as e:
            print(f"⚠️ Lỗi khi tải mô hình phiên âm: {str(e)}")
            self.ready = False
            
        # Ánh xạ phiên âm
        self.phoneme_map = {
            # Nguyên âm cơ bản
            "i": "i", "e": "e", "a": "a", "o": "o", "u": "u",
            # Phụ âm phổ biến
            "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", 
            "g": "g", "f": "f", "v": "v", "s": "s", "z": "z",
            "m": "m", "n": "n", "l": "l", "r": "ɹ", "w": "w", 
            # Ký hiệu đặc biệt
            "th": "θ", "dh": "ð", "sh": "ʃ", "zh": "ʒ",
            " ": " "
        }
            
    def get_phonemes_from_audio(self, audio_path):
        """
        Trích xuất phiên âm từ âm thanh
        """
        # Sử dụng gián tiếp qua transcribe nếu mô hình không khởi tạo thành công
        if not hasattr(self, 'ready') or not self.ready:
            result = transcribe(audio_path, output_ipa=True)
            return {
                "raw_phonemes": result["raw"],
                "ipa": result["ipa"],
                "method": "indirect"
            }
        
        # Phương pháp trực tiếp
        try:
            # Tải và xử lý âm thanh
            audio, sr = load_audio_file(audio_path, target_sr=16000)
            
            # Trích xuất đặc trưng
            with torch.no_grad():
                input_values = self.processor(audio, return_tensors="pt", sampling_rate=16000).input_values
                logits = self.model(input_values).logits
            
            # Lấy dự đoán
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Giải mã thành văn bản
            text = self.processor.batch_decode(predicted_ids)[0]
            
            # Chuyển thành IPA
            ipa = convert_to_ipa(text)
            
            return {
                "raw_phonemes": text,
                "ipa": ipa,
                "method": "direct"
            }
        except Exception as e:
            print(f"⚠️ Lỗi khi trích xuất phiên âm trực tiếp: {str(e)}")
            # Fallback về phương pháp gián tiếp
            result = transcribe(audio_path, output_ipa=True)
            return {
                "raw_phonemes": result["raw"],
                "ipa": result["ipa"],
                "method": "indirect (fallback)"
            }

    def compare_phonemes(self, reference_phonemes, user_phonemes):
        """
        So sánh phiên âm tham chiếu với phiên âm người dùng
        """
        try:
            from Levenshtein import distance
            
            # Chuẩn hóa chuỗi phiên âm
            ref = ''.join(c for c in reference_phonemes if c.strip())
            user = ''.join(c for c in user_phonemes if c.strip())
            
            # Tính khoảng cách Levenshtein
            max_len = max(len(ref), len(user))
            if max_len == 0:
                similarity = 100.0
            else:
                edit_distance = distance(ref, user)
                similarity = ((max_len - edit_distance) / max_len) * 100
            
            return {
                "similarity": round(similarity, 2),
                "score": round(similarity / 10, 2),
                "edit_distance": distance(ref, user)
            }
        except Exception as e:
            print(f"⚠️ Lỗi khi so sánh phiên âm: {str(e)}")
            return {
                "similarity": 0,
                "score": 0,
                "edit_distance": -1
            }

# Tạo instance toàn cục
phoneme_extractor = PhonemeExtractor()

def extract_phonemes(audio_path):
    """
    Hàm wrapper để trích xuất phiên âm từ âm thanh
    """
    return phoneme_extractor.get_phonemes_from_audio(audio_path)