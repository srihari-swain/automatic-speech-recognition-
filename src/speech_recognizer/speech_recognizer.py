
import onnxruntime
import soundfile as sf
import numpy as np
import librosa
import time
from src.configs.config_loader import read_base_config

class SpeechRecognizer:
    def __init__(self):
        config = read_base_config()
        self.session = onnxruntime.InferenceSession(config["model_path"])
        with open(config["vocab_path"], "r", encoding="utf-8") as f:
            self.vocab = [line.strip() for line in f.readlines()]
        self.blank_token = "<unk>" if "<unk>" in self.vocab else self.vocab[-1]

    def compute_log_mel(self, audio, sr, n_mels=80, win_length=400, hop_length=160, n_fft=512):
        print(f"Computing log-mel spectrogram with params: n_mels={n_mels}, win_length={win_length}, hop_length={hop_length}, n_fft={n_fft}")
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=1.0
        )
        log_mel_spec = np.log(np.clip(mel_spec, a_min=1e-5, a_max=None))
        return log_mel_spec

    def preprocess_audio(self, audio_bytes):
        """Preprocess audio file to match model input requirements"""
        try:
            if isinstance(audio_bytes, bytes):
                audio_bytes = io.BytesIO(audio_bytes)
      
            audio, sr = sf.read(audio_bytes, dtype='float32')
            
            if audio.ndim == 2:
                audio = np.mean(audio, axis=1)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Compute log-mel spectrogram
            log_mel = self.compute_log_mel(audio, sr=16000)
            
            # Normalize and prepare for model
            log_mel = log_mel.astype(np.float32)
            log_mel = np.expand_dims(log_mel, axis=0)  # (1, 80, time)
            
            # Normalize
            log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-5)
            
            # Prepare length tensor
            length = np.array([log_mel.shape[2]], dtype=np.int64)
            
            return log_mel, length
        except Exception as e:
            raise Exception(f"Audio preprocessing failed: {str(e)}")

    def beam_search(self, logits, beam_width=10, merge_repeats=True):
        vocab = self.vocab
        blank_token = self.blank_token
        beam = [('', 0.0)] 
        for t in range(logits.shape[1]):
            new_beam = []
            top_k = np.argsort(logits[0, t, :])[-beam_width:][::-1]
            top_k = [idx for idx in top_k if idx < len(vocab)]
            for seq, log_prob in beam:
                for k in top_k:
                    new_seq = seq
                    if k != vocab.index(blank_token):
                        if len(seq) == 0 or (k != vocab.index(seq[-1]) and not (merge_repeats and seq.endswith(vocab[k]))):
                            new_seq = seq + vocab[k]
                    if k == len(vocab) - 1:
                        new_log_prob = log_prob + logits[0, t, k] * 0.5
                    else:
                        new_log_prob = log_prob + logits[0, t, k]
                    new_beam.append((new_seq, new_log_prob))
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]
        return beam[0][0]

    def clean_transcription(self, text):
        text = text.replace('undefined', '').strip()
        text = text.replace('▁', ' ')
        text = ''.join(c for i, c in enumerate(text) if i == 0 or c != text[i-1])
        text = ' '.join(text.split())
        words = text.split()
        cleaned_words = []
        for word in words:
            if len(cleaned_words) == 0 or word != cleaned_words[-1]:
                cleaned_words.append(word)
        return ' '.join(cleaned_words).strip()

    def transcribe(self, audio_bytes):

        start = time.time()
        
        # Preprocess audio
        log_mel, length = self.preprocess_audio(audio_bytes)
        
        # Run inference
        input_names = {i.name for i in self.session.get_inputs()}
        feed = {}
        if "audio_signal" in input_names:
            feed["audio_signal"] = log_mel
        if "length" in input_names:
            feed["length"] = length
        
        logits = self.session.run(None, feed)[0]
        
        # Decode and clean
        transcription = self.beam_search(logits)
        transcription = self.clean_transcription(transcription)
        
        # Calculate confidence
        confidence = float(np.mean(np.exp(logits[0, :, :].max(axis=-1))))
        
        processing_time = time.time() - start
        return transcription, confidence, processing_time
