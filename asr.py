from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

class BaseModel():

    def __init__(self):

        self.tokenizer = None
        self.model = None
        self.lm_decode = None
        self.vad_model = None
        pass


    def run_asr(self, batch):
        """Takes a batch of loaded audio and outputs a
        batch of strings for each audio chunk."""

        # Tokenize the audio
        input_values = self.tokenizer(batch, return_tensors="pt", padding="longest").input_values

        # Feed it through the chosen Model
        with torch.no_grad():
            logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

        # Decode & add to our caption string
        transcription = self.tokenizer.batch_decode(predicted_ids)[0]

        return transcription


    def run_vad(audio, sr=16000):
        pass

    def single_speaker_transcribe(self, path):

        audio, sr = librosa.load(path, sr=16000)

        batch = self.run_vad(audio, sr=sr)

        return self.run_asr(batch, self.tokenizer, self.model, sr=sr)
    
    def two_speaker_transcribe(self, path, speaker_names=list()):

        pass
    

class WhisperTranscriber(BaseModel):

    def __init__(self, model_name, lm_head=True):
        super().__init__()
        self.tokenizer =  WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        


class Wav2Vec2Transcriber(BaseModel):
    def __init__(self, model_name, lm_head=True):
        super().__init__()
        if not lm_head:
            self.tokenizer = Wav2Vec2Processor.from_pretrained(model_name)
        else:
            self.tokenizer = Wav2Vec2ProcessorWithLM.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        

