from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from g2p_en import G2p
import tempfile
import nltk
import os
import numpy as np
import librosa
import re
import noisereduce as nr

nltk.download('averaged_perceptron_tagger', quiet=True)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # nên cấu hình domain cụ thể trong production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Wav2Vec2 phoneme model
MODEL_NAME = "excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
model.eval()

g2p = G2p()

ARPABET_TO_TIMIT = {
    'AA':'aa','AE':'ae','AH':'ah','AH0':'ax','AO':'ao','AW':'aw','AY':'ay','B':'b','CH':'ch',
    'D':'d','DH':'dh','EH':'eh','ER':'er','EY':'ey','F':'f','G':'g','HH':'hh','IH':'ih','IY':'iy',
    'JH':'jh','K':'k','L':'l','M':'m','N':'n','NG':'ng','OW':'ow','OY':'oy','P':'p','R':'r',
    'S':'s','SH':'sh','T':'t','TH':'th','UH':'uh','UW':'uw','V':'v','W':'w','Y':'y','Z':'z',
    'ZH':'zh'
}

VARIANT_MAP = {'ix': 'ih', 'ux': 'uw'}

def normalize_phoneme(p):
    p_norm = p.lower().rstrip('0123456789')
    return VARIANT_MAP.get(p_norm, p_norm)

def filter_phonemes(seq):
    return [p for p in seq if re.fullmatch(r'[a-z]+', p)]

def map_arpabet_to_timit(phonemes):
    mapped = [ARPABET_TO_TIMIT.get(p.upper(), p) for p in phonemes]
    return [normalize_phoneme(p) for p in mapped]

def text_to_phonemes(text: str):
    raw = g2p(text)
    return [p for p in raw if p.strip() and p != ' ']

def audio_to_phonemes(file_path: str):
    waveform, sr = torchaudio.load(file_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

    audio_np = waveform.squeeze().numpy()
    reduced = nr.reduce_noise(y=audio_np, sr=16000, prop_decrease=1.0)

    input_values = processor(torch.tensor(reduced), sampling_rate=16000, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(input_values).logits

    preds = torch.argmax(logits, dim=-1)
    phonemes = processor.batch_decode(preds)[0].split()
    return [normalize_phoneme(p) for p in phonemes]

def aligned_comparison(expected, actual):
    m, n = len(expected), len(actual)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if expected[i - 1] == actual[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    i, j = m, n
    correct, mistakes = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and expected[i - 1] == actual[j - 1]:
            correct.append(expected[i - 1])
            i -= 1
            j -= 1
        else:
            ops = []
            if i > 0: ops.append((dp[i - 1][j] + 1, 'del'))
            if j > 0: ops.append((dp[i][j - 1] + 1, 'ins'))
            if i > 0 and j > 0: ops.append((dp[i - 1][j - 1] + 1, 'sub'))
            cost, op = min(ops, key=lambda x: x[0])
            if op == 'del':
                mistakes.append({"type": "missing", "phoneme": expected[i - 1], "position": i - 1})
                i -= 1
            elif op == 'ins':
                mistakes.append({"type": "extra", "phoneme": actual[j - 1], "position": j - 1})
                j -= 1
            else:
                mistakes.append({
                    "type": "substitution",
                    "expected": expected[i - 1],
                    "actual": actual[j - 1],
                    "position": i - 1
                })
                i -= 1
                j -= 1
    correct.reverse()
    mistakes.reverse()
    score = round(len(correct) / len(expected) * 100) if expected else 0
    return correct, mistakes, score

def get_prosody_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0])
    energy_mean = np.mean(librosa.feature.rms(y=y))
    return pitch_mean, energy_mean

@app.post("/analyze/")
async def analyze_pronunciation(audio: UploadFile = File(...), transcript: str = Form(...)):
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        expected_raw = text_to_phonemes(transcript)
        expected = filter_phonemes(map_arpabet_to_timit(expected_raw))

        actual_raw = audio_to_phonemes(tmp_path)
        actual = filter_phonemes(actual_raw)

        correct, mistakes, score = aligned_comparison(expected, actual)
        pitch, energy = get_prosody_features(tmp_path)
    finally:
        os.remove(tmp_path)

    return JSONResponse({
        "expected_phonemes": expected,
        "user_phonemes": actual,
        "correct_phonemes": correct,
        "mistakes": mistakes,
        "score": score,
        "pitch_mean": round(float(pitch), 2),
        "energy_mean": round(float(energy), 4)
    })
