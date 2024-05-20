from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch, torchaudio
import librosa
from torch.nn.functional import pad
from format_output import generate_eaf

def padded_stack(
    tensors, side= "right", mode="constant", value= 0
) -> torch.Tensor:
    """
    Stack tensors along first dimension and pad them along last dimension to ensure their size is equal.

    Args:
        tensors (List[torch.Tensor]): list of tensors to stack
        side (str): side on which to pad - "left" or "right". Defaults to "right".
        mode (str): 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        value (Union[int, float]): value to use for constant padding

    Returns:
        torch.Tensor: stacked tensor

    function from: https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting/utils.html#padded_stack
    """
    full_size = max([x.size(-1) for x in tensors])

    def make_padding(pad):
        if side == "left":
            return (pad, 0)
        elif side == "right":
            return (0, pad)
        else:
            raise ValueError(f"side for padding '{side}' is unknown")

    out = torch.stack(
        [
            pad(x, make_padding(full_size - x.size(-1)), mode=mode, value=value) if full_size - x.size(-1) > 0 else x
            for x in tensors
        ],
        dim=0,
    )
    return out


class BaseModel():

    def __init__(self):

        self.tokenizer = None
        self.model = None
        self.lm_decode = None
        self.vad_model = None


    def run_asr(self, batch, sr=16000):
        """Takes a batch of loaded audio and outputs a
        batch of strings for each audio chunk."""

        # Tokenize the audio
        input_values = self.tokenizer(batch, sampling_rate=sr, return_tensors="pt").input_features

        # Feed it through the chosen Model
        with torch.no_grad():
            raw_preds = self.model.generate(input_values)
            #:predicted_ids = torch.argmax(logits, dim=-1)

        # Decode & add to our caption string
        transcription = self.tokenizer.batch_decode(raw_preds, skip_special_tokens=True)[0]

        return transcription


    def run_vad(self, audio, sr=16000):
        """Runs vad: derived from 
        https://huggingface.co/spaces/aware-ai/german-asr/blame/f64e0bebe2c666a5dea3d28f854e53673740ea67/app.py"""

        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad')
        
        (get_speech_timestamps,
            _, read_audio,
            *_) = utils
        timestamps = get_speech_timestamps(audio, vad_model, sampling_rate=sr)

        if len(timestamps) > 0:

            timestamp_batch = [(chunk['start']/sr, chunk['end']/sr, audio[chunk['start']:chunk['end']]) for chunk in timestamps]
            raw_batch = torch.cat([audio[chunk['start']:chunk['end']] for chunk in timestamps])
            return timestamp_batch, raw_batch
        else:
            raise NotImplementedError("No speech detected.")
        

    def single_speaker_transcribe(self, path):
        
        timestamped_transcriptions = list()
        raw_audio, sr = librosa.load(path, sr=16000)
        audio = torch.from_numpy(raw_audio)
        timestamp_batch, _= self.run_vad(audio)

        for start_s, end_s, audio in timestamp_batch:
            
            transcript = self.run_asr(audio, sr=sr)
            timestamped_transcriptions.append((start_s, end_s, transcript))
        
        return timestamped_transcriptions
    
    def two_speaker_transcribe(self, path, speaker_names=list()):

        pass

    def get_timestamps(self, transcription, timestamps):

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
        



def main(audio_path):

    model = WhisperTranscriber('openai/whisper-tiny.en')

    transcribed = model.single_speaker_transcribe(audio_path)

    generate_eaf(transcribed, audio_path, 'demo_transcription/elan_output.eaf')
        

if __name__ == '__main__':
    main('/Users/anton/projects/atlas/test_speech/sw2018A.wav')