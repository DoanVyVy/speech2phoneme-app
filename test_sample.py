from model_utils import transcribe
import os
import time

def test_wav_file(file_name):
    file_path = os.path.join('uploads', file_name)
    
    if not os.path.exists(file_path):
        print(f"❌ File không tồn tại: {file_path}")
        return
        
    print(f"🔍 Bắt đầu phân tích file: {file_path}")
    start_time = time.time()
    
    try:
        # Chạy transcribe với kết quả IPA
        result = transcribe(file_path, output_ipa=True)
        end_time = time.time()
        
        if isinstance(result, dict):
            print(f"✅ Kết quả phiên âm gốc: {result['raw']}")
            print(f"🌟 Kết quả phiên âm IPA: {result['ipa']}")
        else:
            print(f"✅ Kết quả phiên âm: {result}")
            
        print(f"⏱️ Thời gian xử lý: {end_time - start_time:.2f} giây")
    except Exception as e:
        print(f"❌ Lỗi khi xử lý file: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test file test.wav
    test_wav_file("test.wav")
    
    # Nếu có file recorded.wav, cũng test luôn
    print("\n" + "-"*50 + "\n")
    test_wav_file("recorded.wav")