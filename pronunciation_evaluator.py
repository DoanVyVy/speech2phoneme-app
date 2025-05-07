import torch
import numpy as np
import Levenshtein
import re
from model_utils import transcribe, convert_to_ipa
from phoneme_extractor import phoneme_extractor, extract_phonemes
import librosa

class PronunciationEvaluator:
    def __init__(self):
        """Khởi tạo đánh giá phát âm"""
        # Tải bộ từ điển phát âm CMU (Carnegie Mellon University) nếu có
        # Hoặc sử dụng các tài nguyên phát âm khác
        self.phoneme_weights = {
            # Trọng số cho các lỗi phát âm thường gặp (với người nói tiếng Việt)
            'θ': 2.0,  # th khó phát âm
            'ð': 2.0,  # th voiced khó phát âm
            'ɹ': 1.5,  # r khó phát âm
            'ʃ': 1.5,  # sh khó phát âm
            'ʒ': 1.5,  # zh khó phát âm
            'æ': 1.5,  # âm æ khó phát âm
            'ə': 1.2,  # schwa khó phát âm
        }
        
    def get_standard_pronunciation(self, text):
        """
        Lấy phát âm chuẩn cho một đoạn văn bản tiếng Anh
        Trong thực tế, bạn có thể sử dụng từ điển phát âm như CMUdict
        """
        # Đây là một giải pháp tạm thời, ứng dụng thực tế cần từ điển phát âm
        return convert_to_ipa(text)
        
    def dynamic_time_warping(self, seq1, seq2):
        """
        Tính toán DTW (Dynamic Time Warping) giữa hai chuỗi ký tự
        """
        n, m = len(seq1), len(seq2)
        dtw_matrix = np.zeros((n+1, m+1))
        
        for i in range(n+1):
            for j in range(m+1):
                if i == 0 and j == 0:
                    dtw_matrix[i, j] = 0
                elif i == 0:
                    dtw_matrix[i, j] = dtw_matrix[i, j-1] + 1
                elif j == 0:
                    dtw_matrix[i, j] = dtw_matrix[i-1, j] + 1
                else:
                    cost = 0 if seq1[i-1] == seq2[j-1] else 1
                    dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],      # insertion
                                                dtw_matrix[i, j-1],      # deletion
                                                dtw_matrix[i-1, j-1])    # substitution
                    
        return dtw_matrix[n, m]
        
    def weighted_levenshtein_distance(self, ref_ipa, user_ipa):
        """
        Tính khoảng cách Levenshtein có trọng số giữa phiên âm chuẩn và phiên âm người dùng
        """
        n, m = len(ref_ipa), len(user_ipa)
        dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
        
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_ipa[i-1] == user_ipa[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    weight = self.phoneme_weights.get(ref_ipa[i-1], 1.0)
                    dp[i][j] = min(
                        dp[i-1][j] + weight,  # deletion
                        dp[i][j-1] + 1.0,     # insertion
                        dp[i-1][j-1] + weight # substitution
                    )
        
        return dp[n][m]
        
    def analyze_rhythm_and_stress(self, audio_path, reference_text):
        """
        Phân tích nhịp điệu và trọng âm
        """
        try:
            # Tải âm thanh
            y, sr = librosa.load(audio_path, sr=None)
            
            # Trích xuất đặc trưng
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            rhythm_regularity = np.std(onset_env)
            
            # Phân tích trọng âm thông qua biên độ
            rms = librosa.feature.rms(y=y)[0]
            stress_contrast = np.max(rms) / np.mean(rms)
            
            return {
                "tempo": tempo,
                "rhythm_regularity": float(rhythm_regularity),
                "stress_contrast": float(stress_contrast)
            }
        except Exception as e:
            print(f"Error analyzing rhythm: {str(e)}")
            return {
                "tempo": 0,
                "rhythm_regularity": 0,
                "stress_contrast": 0
            }
    
    def evaluate(self, audio_path, reference_text):
        """
        Đánh giá phát âm
        
        Args:
            audio_path: Đường dẫn đến file âm thanh người dùng
            reference_text: Văn bản tham chiếu (tiếng Anh)
            
        Returns:
            Dict chứa kết quả đánh giá
        """
        # NEW METHOD: Lấy phiên âm trực tiếp từ âm thanh (không qua văn bản)
        user_phonemes = extract_phonemes(audio_path)
        user_text = transcribe(audio_path, output_ipa=False)  # Vẫn lấy văn bản để hiển thị
        user_ipa_direct = user_phonemes["ipa"]  # Phiên âm IPA trực tiếp từ âm thanh
        
        # Lưu cả phiên âm gián tiếp (qua văn bản) để so sánh với phương pháp cũ
        user_result = transcribe(audio_path, output_ipa=True)
        user_ipa_indirect = user_result["ipa"]  # Phiên âm IPA gián tiếp qua văn bản
        
        # Lấy phát âm chuẩn (ground truth)
        reference_ipa = self.get_standard_pronunciation(reference_text)
        
        # Tính điểm độ chính xác văn bản (ASR accuracy)
        text_distance = Levenshtein.distance(reference_text.lower(), user_text.lower())
        text_max_len = max(len(reference_text), len(user_text))
        text_accuracy = ((text_max_len - text_distance) / text_max_len) * 100 if text_max_len > 0 else 100
        
        # Tính điểm độ chính xác phiên âm trực tiếp (Direct Phoneme accuracy)
        direct_ipa_distance = self.weighted_levenshtein_distance(reference_ipa, user_ipa_direct)
        direct_ipa_max_len = max(len(reference_ipa), len(user_ipa_direct))
        direct_pronunciation_accuracy = ((direct_ipa_max_len - direct_ipa_distance) / direct_ipa_max_len) * 100 if direct_ipa_max_len > 0 else 100
        
        # Tính điểm độ chính xác phiên âm gián tiếp (qua văn bản) cho tham chiếu
        indirect_ipa_distance = self.weighted_levenshtein_distance(reference_ipa, user_ipa_indirect)
        indirect_ipa_max_len = max(len(reference_ipa), len(user_ipa_indirect))
        indirect_pronunciation_accuracy = ((indirect_ipa_max_len - indirect_ipa_distance) / indirect_ipa_max_len) * 100 if indirect_ipa_max_len > 0 else 100
        
        # So sánh độ chính xác giữa hai phương pháp phiên âm
        print(f"🔍 Độ chính xác phiên âm trực tiếp: {direct_pronunciation_accuracy:.2f}%")
        print(f"🔍 Độ chính xác phiên âm gián tiếp: {indirect_pronunciation_accuracy:.2f}%")
        
        # Sử dụng phương pháp có độ chính xác cao hơn để đánh giá
        pronunciation_accuracy = max(direct_pronunciation_accuracy, indirect_pronunciation_accuracy)
        user_ipa = user_ipa_direct if direct_pronunciation_accuracy >= indirect_pronunciation_accuracy else user_ipa_indirect
        
        # Phân tích nhịp điệu và trọng âm
        rhythm_analysis = self.analyze_rhythm_and_stress(audio_path, reference_text)
        
        # Tính điểm tổng hợp
        # 60% phát âm, 20% văn bản, 20% nhịp điệu
        rhythm_score = min(100, rhythm_analysis["stress_contrast"] * 20)
        total_score = 0.6 * pronunciation_accuracy + 0.2 * text_accuracy + 0.2 * rhythm_score
        
        # Xác định mức độ phát âm
        level = "Beginner"
        if total_score >= 90:
            level = "Native-like"
        elif total_score >= 80:
            level = "Advanced"
        elif total_score >= 70:
            level = "Intermediate"
        elif total_score >= 60:
            level = "Pre-intermediate"
            
        # Tìm lỗi cụ thể bằng cách so sánh từng từ
        errors = self.identify_specific_errors(reference_text, user_text, reference_ipa, user_ipa)
        
        # Phân tích phiên âm trực tiếp
        phoneme_comparison = phoneme_extractor.compare_phonemes(reference_ipa, user_ipa_direct)
        
        return {
            "score": round(total_score, 2),
            "level": level,
            "details": {
                "pronunciation_accuracy": round(pronunciation_accuracy, 2),
                "direct_pronunciation_accuracy": round(direct_pronunciation_accuracy, 2),
                "indirect_pronunciation_accuracy": round(indirect_pronunciation_accuracy, 2),
                "text_accuracy": round(text_accuracy, 2),
                "rhythm_score": round(rhythm_score, 2),
                "tempo": round(rhythm_analysis["tempo"], 2),
                "phoneme_similarity": phoneme_comparison["similarity"]
            },
            "reference": {
                "text": reference_text,
                "ipa": reference_ipa
            },
            "user": {
                "text": user_text,
                "ipa": user_ipa,
                "direct_ipa": user_ipa_direct,
                "indirect_ipa": user_ipa_indirect,
                "raw_phonemes": user_phonemes["raw_phonemes"]
            },
            "errors": errors,
            "method": "direct" if direct_pronunciation_accuracy >= indirect_pronunciation_accuracy else "indirect"
        }
    
    def identify_specific_errors(self, ref_text, user_text, ref_ipa, user_ipa):
        """
        Xác định các lỗi phát âm cụ thể
        """
        errors = []
        
        # Chuyển thành các từ
        ref_words = ref_text.lower().split()
        user_words = user_text.lower().split()
        
        # Tìm từ thiếu
        missing = set(ref_words) - set(user_words)
        if missing:
            errors.append({
                "type": "missing_words",
                "description": f"Thiếu các từ: {', '.join(missing)}"
            })
        
        # Tìm từ thừa
        extra = set(user_words) - set(ref_words)
        if extra:
            errors.append({
                "type": "extra_words",
                "description": f"Thêm các từ không có trong văn bản: {', '.join(extra)}"
            })
        
        # Phân tích lỗi thường gặp
        common_errors = self.analyze_common_errors(ref_ipa, user_ipa)
        if common_errors:
            errors.extend(common_errors)
            
        return errors
    
    def analyze_common_errors(self, ref_ipa, user_ipa):
        """
        Phân tích các lỗi phát âm phổ biến
        """
        errors = []
        
        # Kiểm tra âm "th"
        if 'θ' in ref_ipa and 'θ' not in user_ipa:
            errors.append({
                "type": "th_sound",
                "description": "Lỗi phát âm âm 'th' như trong 'think'"
            })
        
        # Kiểm tra âm "th" voiced
        if 'ð' in ref_ipa and 'ð' not in user_ipa:
            errors.append({
                "type": "th_voiced_sound",
                "description": "Lỗi phát âm âm 'th' như trong 'the'"
            })
            
        # Kiểm tra âm "r"
        if 'ɹ' in ref_ipa and 'ɹ' not in user_ipa:
            errors.append({
                "type": "r_sound",
                "description": "Lỗi phát âm âm 'r' tiếng Anh"
            })
            
        # Thêm nhiều phân tích khác...
        
        return errors

# Singleton instance
evaluator = PronunciationEvaluator()

def evaluate_pronunciation(audio_path, reference_text):
    """
    Hàm wrapper để đánh giá phát âm
    """
    return evaluator.evaluate(audio_path, reference_text)