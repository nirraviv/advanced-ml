#!/usr/bin/env python3

import sys
import os
import argparse
import math

from pydub import AudioSegment
from pydub.generators import Sine
import pydub

from noteutils import text_to_keys, notes_to_keys, keys_to_pitch, keys_to_text

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

def generate_got():
    notes = 'e5 e5 e5 a4 a4 c5 d5 e5 e5 e5 a4 a4 c5 d5 b4 b4 b4 b4 b4 b4'.split(' ')
    song = pitch_to_audio(keys_to_pitch(notes_to_keys(notes)), 500)
    song.export('ha_tikva.mp3', format='mp3')

def generate_bach():
    notes = 'e5 D5 e5 b4 c5 G4 a4'.split(' ')
    song = pitch_to_audio(keys_to_pitch(notes_to_keys(notes)), 1000)
    song.export('ha_tikva.mp3', format='mp3')

def generate_circle():
    notes = 'g4 a4 c5 c5 c5 c5 e5 d5 c5 c5 e5 e5 e5 e5 e5 d5 c5 c5 c5 c5 a4 c5 g4 g4 g4 g4 g4 g4'.split(' ')
    song = pitch_to_audio(keys_to_pitch(notes_to_keys(notes)), 500)
    song.export('ha_tikva.mp3', format='mp3')

def generate_hp():
    notes = 'c5 A4 c5 f5 D5 d5 D5 g5 f5 f5 f5'.split(' ')
    song = pitch_to_audio(keys_to_pitch(notes_to_keys(notes)), 1000)
    song.export('ha_tikva.mp3', format='mp3')

songs = [
    ('ha_tikva_2', 500, 'a4 b4 c5 d5 e5 e5 e5 e5 f5 e5 f5 a5 e5 e5 x x d5 d5 d5 d5 c5 c5 c5 c5 b4 a4 b4 c5 a4 a4 a4'),
    ('ha_tikva_2a', 1000, 'a4 b4 c5 d5 e5 e5 e5 e5 f5 e5 f5 a5 e5 e5 x x'),
    ('ha_tikva_2b', 1000, 'd5 d5 d5 d5 c5 c5 c5 c5 b4 a4 b4 c5 a4 a4 a4 x'),
    ('ha_tikva_3', 500, 'a3 b3 c4 d4 e4 e4 e4 e4 f4 e4 f4 a4 e4 e4 x x d4 d4 d4 d4 c4 c4 c4 c4 b3 a3 b3 c4 a3 a3 a3 x'),
    ('ha_tikva_3a', 1000, 'a3 b3 c4 d4 e4 e4 e4 e4 f4 e4 f4 a4 e4 e4 x x'),
    ('ha_tikva_3b', 1000, 'd4 d4 d4 d4 c4 c4 c4 c4 b3 a3 b3 c4 a3 a3 a3 x'),
    ('game_of_thrones', 500, 'e5 e5 e5 a4 a4 c5 d5 e5 e5 e5 a4 a4 c5 d5 b4 b4 b4 b4 b4 b4'),
    ('bach', 1500, 'e5 D5 e5 b4 c5 G4 a4'),
    ('circle', 500, 'g4 a4 c5 c5 c5 c5 e5 d5 c5 c5 e5 e5 e5 e5 e5 d5 c5 c5 c5 c5 a4 c5 g4 g4 g4 g4 g4 g4'),
    ('harry_potter', 1000, 'c5 A4 c5 f5 D5 d5 D5 g5 f5 f5 f5'),
]

def main():
    for s in songs:
        audio = pitch_to_audio(keys_to_pitch(notes_to_keys(s[2].split(' '))), s[1])
        audio.export('test_' + s[0] + '.mp3', format='mp3')

    with open('tests.txt', 'w') as f:
        for s in songs:
            text = keys_to_text(notes_to_keys(s[2].split(' ')))
            text = ''.join([n * int(s[1] / 1000 * 8 + 0.5) for n in text])
            f.write(text + '\n')

if __name__ == '__main__':
    main()
