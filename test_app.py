from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_analyze_pronunciation():
    audio_path = "hello.wav"
    transcript = "hello"

    with open(audio_path, "rb") as f:
        response = client.post(
            "/analyze/",
            files={"audio": ("hello.wav", f, "audio/wav")},
            data={"transcript": transcript}
        )

    assert response.status_code == 200
    result = response.json()

    print("Expected:", result["expected_phonemes"])
    print("User:", result["user_phonemes"])
    print("Correct:", result["correct_phonemes"])
    print("Mistakes:", result["mistakes"])
    print("Score:", result["score"])

if __name__ == "__main__":
    test_analyze_pronunciation()
