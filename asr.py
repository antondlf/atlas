from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch, torchaudio
import librosa
from torch.nn.functional import pad
from format_output import generate_eaf
from utils import select_language
import argparse
import pathlib
from pathlib import Path
from pyannote.audio import Pipeline

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

        # Feed it through the chosen Model
        with torch.no_grad():
            if self.model_type == 'whisper':
                input_values = self.tokenizer(batch, sampling_rate=sr, return_tensors="pt").input_features
                raw_preds = self.model.generate(input_values)
                transcription = self.tokenizer.batch_decode(raw_preds, skip_special_tokens=True)
            elif self.model_type == 'wav2vec2':
                input_values = self.tokenizer(batch, sampling_rate=sr, return_tensors="pt").input_values
                raw_preds = self.model(input_values).logits.squeeze().cpu().numpy()
                transcription = self.tokenizer.decode(raw_preds).text
            else:
                raise TypeError("Model class loaded is not supported.")

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


    def diarize(self, audio_path, sr=16000):

        diarization = self.diarizer(audio_path, num_speakers=n_speakers)

        # dump the diarization output to disk using RTTM format
        with open(audio_path.with_suffix(".rttm"), "w") as rttm:
            diarization.write_rttm(rttm)

        return diarization

    def single_speaker_transcribe(self, path, tiername='SPEAKER_00'):
        
        timestamped_transcriptions = dict()
        timestamped_transcriptions[tiername] = list()
        raw_audio, sr = librosa.load(path, sr=16000)
        audio = torch.from_numpy(raw_audio)
        timestamp_batch, _= self.run_vad(audio)

        for start_s, end_s, audio in timestamp_batch:
            
            transcript = self.run_asr(audio, sr=sr)
            timestamped_transcriptions.append((start_s, end_s, transcript))

        return timestamped_transcriptions, timestamped_transcriptions.keys()
    
    def two_speaker_transcribe(self, path, n_speakers=2):
        
        speaker_dict = dict()
        diarized_output = self.diarizer(path, num_speakers=n_speakers)
        raw_audio, sr = librosa.load(path, sr=16000)
        audio = torch.from_numpy(raw_audio)
        
        for speaker in diarized_output.labels():
            speaker_dict[speaker] = list()
            for segment in diarized_output.label_timeline(speaker):
                
                start_s = segment.start
                end_s = segment.end
                audio_segment = audio[int(start_s*16000):int(end_s*16000)]
                try:
                    transcript = self.run_asr(audio_segment, sr=sr)
                except RuntimeError:
                    print(f"{start_s} to {end_s} was short for speaker {speaker}")
                    continue
                speaker_dict[speaker].append((start_s, end_s, transcript))

        return speaker_dict, speaker_dict.keys()

    def get_timestamps(self, transcription, timestamps):

        pass

class WhisperTranscriber(BaseModel):

    def __init__(self, model_name, lm_head=True):
        super().__init__()
        with open('.secrets/hf_access_tok.txt', 'r') as f:
            access_token = f.read().replace('\n', '')

        self.diarizer =  Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=access_token)
        self.tokenizer =  WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model_type = "whisper"
        


class Wav2Vec2Transcriber(BaseModel):
    def __init__(self, model_name, lm_head=True):
        super().__init__()
        with open('.secrets/hf_access_tok.txt', 'rb') as f:
            access_token = f.readline().rstrip()

        self.diarizer =  Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=access_token)
        self.lm_head = lm_head
        if not lm_head:
            self.tokenizer = Wav2Vec2Processor.from_pretrained(model_name)
        else:
            self.tokenizer = Wav2Vec2ProcessorWithLM.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model_type = "wav2vec2"
        



def transcribe_audio(audio_path, language, output_path, quantization, nspeakers):

    model_name = select_language(language)

    if language == 'English':
        model = WhisperTranscriber(model_name)
    else:
        model = Wav2Vec2Transcriber(model_name)

    if nspeakers == 1:
        transcribed,tiernames = model.single_speaker_transcribe(audio_path)
    else:
        transcribed, tiernames = model.two_speaker_transcribe(audio_path)
    
    generate_eaf(transcribed, audio_path, output_path, n_speakers=nspeakers, tiernames=tiernames)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Automatic Transcription for Linguistic Annotations',
        description="Program takes in a file or directory and transcribes"\
        "audio to an elan format",
    )
    parser.add_argument(
        'audio_path', help="path to directory or wav file",
        type=Path
    )
    parser.add_argument(
        '-l', "--language",
          help="language to transcribe from. Current support includes English and Galician.",
          default='English',
          type=str
    )
    parser.add_argument(
        '-o', "--output_path",
          help="path to save output file.",
          type=Path,
          default=None
    )
    parser.add_argument(
        '-q', '--quantization',
          help="whether to quantize the model",
          type=str
    )
    parser.add_argument(
        '-n', '--n_speakers', default=1,
        help="Number of speakers expected to be in the recording",
        type=int
    )

    args = parser.parse_args()

    audio_path = args.audio_path
    language = args.language
    output_path = args.output_path if args.output_path is not None else audio_path.with_suffix('.eaf')
    quantization = args.quantization
    n_speakers = args.n_speakers
    if audio_path.is_dir():
        for file in audio_path.glob('*.wav'):
            
            out_path = file.with_suffix('.eaf') if output_path is None else output_path / f'{file.stem}.wav'
            transcribe_audio(file, language, out_path, quantization, n_speakers)

    else:

        out_path = audio_path.with_suffix('.eaf') if output_path is None else output_path
        transcribe_audio(audio_path, language, output_path, quantization, n_speakers)