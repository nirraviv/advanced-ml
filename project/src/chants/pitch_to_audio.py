#!/usr/bin/env python3

import sys
import os
import argparse
import math

from pydub import AudioSegment
from pydub.generators import Sine
import pydub

from noteutils import text_to_keys, notes_to_keys, keys_to_pitch

def pitch_to_audio(pitch_list, note_length=125):
    song = AudioSegment.empty()
    for p in pitch_list:
        song += Sine(p).to_audio_segment(note_length)

    return song

def generate_ha_tikva():
    notes = 'a4 b4 c5 d5 e5 x e5 x f5 e5 f5 a5 e5 e5 x x d5 x d5 d5 c5 x c5 x b4 a4 b4 c5 a4 a4 x x'.split(' ')
    song = pitch_to_audio(notes_to_pitch(notes), 300)
    song.export('ha_tikva.mp3', format='mp3')

def generate_ha_tikva_2():
    notes = 'a4 b4 c5 d5 e5 x e5 x f5 e5 f5 a5 e5 e5 x x d5 x d5 d5 c5 x c5 c5 b4 a4 b4 c5 a4 a4 a4 x'.split(' ')
    song = pitch_to_audio(keys_to_pitch(notes_to_keys(notes)), 1000)
    song.export('ha_tikva.mp3', format='mp3')

def generate_ha_tikva_3():
    notes = 'a3 b3 c4 d4 e4 x e4 x f4 e4 f4 a4 e4 e4 x x d4 x d4 d4 c4 x c4 c4 b3 a3 b3 c4 a3 a3 a3 x'.split(' ')
    song = pitch_to_audio(keys_to_pitch(notes_to_keys(notes)), 1000)
    song.export('ha_tikva.mp3', format='mp3')

def main():
    parser = argparse.ArgumentParser(description='Generate audio clips from pitch tracks.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('pitch_file')
    parser.add_argument('-o', '--output_file', default='song.mp3', help='output audio file name')
    parser.add_argument('-t', '--note_rate', default=8, type=float, help='Note length in milliseconds')
    args = parser.parse_args()

    with open(args.pitch_file) as f:
        text = f.read()

    if any(char.isdigit() for char in text):
        keys = notes_to_keys(text.aplit(' '))
    else:
        keys = text_to_keys(text)
    song = pitch_to_audio(keys_to_pitch(keys), int(1000 / args.note_rate))

    song.export(args.output_file, format='mp3')

if __name__ == '__main__':
    main()
