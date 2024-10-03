import pretty_midi
from utils import bin_time
import os
from pathlib import Path

def parse_midi_file(midi_file_path):
    
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    
    
    melody_attributes = []

    
    intervals = []
    
    
    for instrument in midi_data.instruments:
        
        
        if instrument.is_drum:
            continue
        
        previous_note_end = None
        
        for note in instrument.notes:
            
            note_duration = note.end - note.start

            
            if previous_note_end is not None:
                interval = note.start - previous_note_end
            else:
                interval = 0  
            intervals.append(interval)
            # print(f"Interval between note ending at {previous_note_end:.6f} and note starting at {note.start:.6f}: {interval:.6f} seconds" if previous_note_end is not None else f"First note, interval: {interval:.6f} seconds")

            
            previous_note_end = note.end
            
            # print(note.start, note.end, note_duration, previous_note_end,interval)
            melody_attributes.append((note.pitch, note_duration, interval))
    
    return melody_attributes




midi_file_path = "/home/ubuntu/wmz/AudioLDM-training-finetuning/audioldm_train/modules/clap/songcomposer_tokenizer/melody_encoder/test_data/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav_segment_0.mid"

melody_attributes = parse_midi_file(midi_file_path)


# for attribute in melody_attributes:
#     print(attribute)

second_column = []
for triple in melody_attributes:
    second_column.append(triple[1])

discretized_second_column = bin_time(second_column)

third_column = []
for triple in melody_attributes:
    third_column.append(triple[2])

discretized_third_column = bin_time(third_column)


for i, triple in enumerate(melody_attributes):
    melody_attributes[i] = (triple[0], discretized_second_column[i], discretized_third_column[i])

# for attribute in melody_attributes:
#     print(attribute)


pitch_to_note = {
    0: 'C-1', 1: 'C#-1', 2: 'D-1', 3: 'D#-1', 4: 'E-1', 5: 'F-1', 6: 'F#-1', 7: 'G-1', 8: 'G#-1', 9: 'A-1', 10: 'A#-1', 11: 'B-1',
    12: 'C0', 13: 'C#0', 14: 'D0', 15: 'D#0', 16: 'E0', 17: 'F0', 18: 'F#0', 19: 'G0', 20: 'G#0', 21: 'A0', 22: 'A#0', 23: 'B0',
    24: 'C1', 25: 'C#1', 26: 'D1', 27: 'D#1', 28: 'E1', 29: 'F1', 30: 'F#1', 31: 'G1', 32: 'G#1', 33: 'A1', 34: 'A#1', 35: 'B1',
    36: 'C2', 37: 'C#2', 38: 'D2', 39: 'D#2', 40: 'E2', 41: 'F2', 42: 'F#2', 43: 'G2', 44: 'G#2', 45: 'A2', 46: 'A#2', 47: 'B2',
    48: 'C3', 49: 'C#3', 50: 'D3', 51: 'D#3', 52: 'E3', 53: 'F3', 54: 'F#3', 55: 'G3', 56: 'G#3', 57: 'A3', 58: 'A#3', 59: 'B3',
    60: 'C4', 61: 'C#4', 62: 'D4', 63: 'D#4', 64: 'E4', 65: 'F4', 66: 'F#4', 67: 'G4', 68: 'G#4', 69: 'A4', 70: 'A#4', 71: 'B4',
    72: 'C5', 73: 'C#5', 74: 'D5', 75: 'D#5', 76: 'E5', 77: 'F5', 78: 'F#5', 79: 'G5', 80: 'G#5', 81: 'A5', 82: 'A#5', 83: 'B5',
    84: 'C6', 85: 'C#6', 86: 'D6', 87: 'D#6', 88: 'E6', 89: 'F6', 90: 'F#6', 91: 'G6', 92: 'G#6', 93: 'A6', 94: 'A#6', 95: 'B6',
    96: 'C7', 97: 'C#7', 98: 'D7', 99: 'D#7', 100: 'E7', 101: 'F7', 102: 'F#7', 103: 'G7', 104: 'G#7', 105: 'A7', 106: 'A#7', 107: 'B7',
    108: 'C8', 109: 'C#8', 110: 'D8', 111: 'D#8', 112: 'E8', 113: 'F8', 114: 'F#8', 115: 'G8', 116: 'G#8', 117: 'A8', 118: 'A#8', 119: 'B8',
    120: 'C9', 121: 'C#9', 122: 'D9', 123: 'D#9', 124: 'E9', 125: 'F9', 126: 'F#9', 127: 'G9'
}


# for attribute in melody_attributes:
#     pitch = attribute[0]
#     note_symbol = pitch_to_note.get(pitch, "Unknown")
#     print(f"Pitch: {pitch} ({note_symbol}), Duration: {attribute[1]}, Interval: {attribute[2]}")


for i, attribute in enumerate(melody_attributes):
    pitch = attribute[0]
    note_symbol = pitch_to_note.get(pitch, "Unknown")
    melody_attributes[i] = (note_symbol, attribute[1], attribute[2])

# print(melody_attributes)

formatted_prompt = '|'.join([f"<{p[0]}>,{p[1]},{p[2]}" for p in melody_attributes])
print(formatted_prompt)


# midi_file_name = os.path.basename(midi_file_path)
# print(midi_file_name)


# text_file_path = os.path.splitext(midi_file_name)[0] + '.txt'


# output_dir = "/home/ubuntu/wmz/AudioLDM-training-finetuning/audioldm_train/modules/clap/songcomposer_tokenizer/melody_encoder/test_data"
# with open(Path(output_dir) / text_file_path, 'w', encoding='utf-8') as f:
#     f.write(formatted_prompt)

