
def select_language(language_name):

    LANGUAGES_SUPPORTED = {

        ('English', 'english', 'en', 'english-tiny'): 'openai/whisper-tiny',
        ('English-large', 'english-large', 'en', 'whisper-large'): 'openai/whisper-large',
        ('Galician', 'gl', 'galego', 'Galego', 'glg', 'galician-wav2vec2'): 'proxectonos/Nos_ASR-wav2vec2-large-xlsr-53-gl-with-lm', #'Akashpb13/Galician_xlsr',
        ('galician-whisper'): 'ITG/whisper-small-gl',
        ('Spanish', 'spanish', 'español', 'es'): 'jonatasgrosman/wav2vec2-large-xlsr-53-spanish',


    }
    lang_dict = {key: value for keys, value in LANGUAGES_SUPPORTED.items() for key in keys}
    return lang_dict[language_name]
