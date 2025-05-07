"""
Script kiểm thử chức năng trích xuất phiên âm trực tiếp từ âm thanh
"""

import os
import time
from phoneme_extractor import extract_phonemes
from model_utils import transcribe, convert_to_ipa

def test_phoneme_extraction(audio_path):
    """
    Kiểm thử và so sánh hai phương pháp trích xuất phiên âm
    """
    print(f"🔍 Kiểm thử trích xuất phiên âm: {audio_path}")
    
    # Phương pháp 1: Trực tiếp từ âm thanh sang phiên âm
    start_time = time.time()
    direct_result = extract_phonemes(audio_path)
    direct_time = time.time() - start_time
    
    # Phương pháp 2: Âm thanh -> Văn bản -> Phiên âm (gián tiếp)
    start_time = time.time()
    indirect_result = transcribe(audio_path, output_ipa=True)
    indirect_time = time.time() - start_time
    
    # In kết quả
    print("-" * 50)
    print("PHƯƠNG PHÁP 1: TRỰC TIẾP TỪ ÂM THANH SANG PHIÊN ÂM")
    print(f"Phiên âm thô: {direct_result['raw_phonemes']}")
    print(f"Phiên âm IPA: {direct_result['ipa']}")
    print(f"Thời gian xử lý: {direct_time:.2f} giây")
    
    print("\n" + "-" * 50)
    print("PHƯƠNG PHÁP 2: ÂM THANH -> VĂN BẢN -> PHIÊN ÂM")
    print(f"Văn bản nhận dạng: {indirect_result['raw']}")
    print(f"Phiên âm IPA: {indirect_result['ipa']}")
    print(f"Thời gian xử lý: {indirect_time:.2f} giây")
    
    # So sánh hai phương pháp
    print("\n" + "-" * 50)
    print("SO SÁNH HAI PHƯƠNG PHÁP:")
    print(f"Phiên âm trực tiếp (P1): {direct_result['ipa']}")
    print(f"Phiên âm gián tiếp (P2): {indirect_result['ipa']}")
    
    # Tính độ tương đồng (để hiểu sự khác biệt giữa hai phương pháp)
    from Levenshtein import distance
    similarity = 100 - (distance(direct_result['ipa'], indirect_result['ipa']) / 
                        max(len(direct_result['ipa']), len(indirect_result['ipa'])) * 100)
    
    print(f"\nĐộ tương đồng giữa hai phương pháp: {similarity:.2f}%")
    print(f"Thời gian P1 vs P2: {direct_time:.2f}s vs {indirect_time:.2f}s")
    print(f"Chênh lệch thời gian: {(direct_time - indirect_time) / indirect_time * 100:.2f}%")
    
    
if __name__ == "__main__":
    # Kiểm tra file test.wav
    test_file = os.path.join('uploads', 'test.wav')
    if os.path.exists(test_file):
        test_phoneme_extraction(test_file)
    else:
        print(f"❌ Không tìm thấy file {test_file}")
    
    # Kiểm tra file recorded.wav (nếu có)
    recorded_file = os.path.join('uploads', 'recorded.wav')
    if os.path.exists(recorded_file):
        print("\n" + "=" * 70 + "\n")
        test_phoneme_extraction(recorded_file)
    else:
        print(f"\n❌ Không tìm thấy file {recorded_file}")