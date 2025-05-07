# Speech2Phoneme: ứng dụng chuyển đổi tiếng nói sang phiên âm IPA

Speech2Phoneme là một ứng dụng API Flask cho phép chuyển đổi tiếng nói thành phiên âm IPA (International Phonetic Alphabet) và đánh giá phát âm tiếng Anh.

## Tính năng

- **Chuyển đổi tiếng nói sang văn bản**: Sử dụng mô hình Wav2Vec2 của Facebook để nhận dạng tiếng nói
- **Chuyển đổi văn bản sang phiên âm IPA**: Chuyển đổi văn bản thành ký hiệu phiên âm quốc tế
- **Đánh giá phát âm**: Đánh giá chất lượng phát âm tiếng Anh và cung cấp điểm số cùng với phân tích chi tiết
- **Phát hiện lỗi phát âm**: Xác định các lỗi phát âm phổ biến và đưa ra gợi ý cải thiện

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/username/speech2phoneme-app.git
cd speech2phoneme-app/backend
```

2. Cài đặt các gói phụ thuộc:
```bash
pip install -r requirements.txt
```

3. Cài đặt FFmpeg (cần thiết cho xử lý âm thanh):
   - Windows: Tải từ [ffmpeg.org](https://ffmpeg.org/download.html) và thêm vào PATH
   - Linux: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`

## Sử dụng

1. Khởi động server:
```bash
python app.py
```

2. Các API endpoints:

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| /upload | POST | Chuyển đổi âm thanh sang phiên âm |
| /evaluate | POST | Đánh giá phát âm tiếng Anh |

### API /upload

Tải lên tệp âm thanh và chuyển đổi thành phiên âm.

**Request:**
- Form-data:
  - `audio`: File âm thanh (WAV, MP3, etc.)
  - `output_ipa`: Boolean (true/false) - có trả về phiên âm IPA hay không

**Response:**
```json
{
  "status": "success",
  "filename": "example.wav",
  "raw_phonemes": "this is an example",
  "ipa_phonemes": "ðɪs ɪz ən ɪgzæmpəl"
}
```

### API /evaluate

Đánh giá phát âm tiếng Anh từ tệp âm thanh.

**Request:**
- Form-data:
  - `audio`: File âm thanh người dùng đọc
  - `reference_text`: Văn bản tham chiếu người dùng đang đọc

**Response:**
```json
{
  "status": "success",
  "filename": "example.wav",
  "evaluation": {
    "score": 75.5,
    "level": "Intermediate",
    "details": {
      "pronunciation_accuracy": 78.2,
      "text_accuracy": 85.0,
      "rhythm_score": 60.0,
      "tempo": 120.5
    },
    "errors": [
      {
        "type": "th_sound",
        "description": "Lỗi phát âm âm 'th' như trong 'think'"
      }
    ]
  }
}
```

## Yêu cầu hệ thống

- Python 3.8+
- FFmpeg
- 4GB RAM trở lên (để tải mô hình Wav2Vec2)

## Công nghệ sử dụng

- Flask: Framework web API
- Transformers (Hugging Face): Mô hình Wav2Vec2 cho nhận dạng tiếng nói
- Librosa: Xử lý âm thanh và phân tích
- PyDub: Chuyển đổi định dạng âm thanh