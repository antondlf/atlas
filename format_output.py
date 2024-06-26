from pympi.Elan import Eaf
from pympi.Praat import TextGrid


def generate_eaf(transcriptions, input_file, output_file, n_speakers=1, tiernames=None):

    if tiernames is None:
        tiernames = [f'Speaker:{n+1}' for n in range(0, n_speakers)]
    
    elan_file = Eaf()
    elan_file.add_linked_file(str(input_file))

    for tier in tiernames:

        elan_file.add_tier(tier)

        for start_s, end_s, text in transcriptions[tier]:

            elan_file.add_annotation(tier, int(start_s*1000), int(end_s*1000), text)

    
    elan_file.to_file(str(output_file))
