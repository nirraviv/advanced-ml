#!/usr/bin/env python3

import sys
import os
import argparse
import math
import numpy as np

import essentia
import essentia.standard

import noteutils

VISUALIZE = False

def extract_pitch(filename, noteRate=8):
    sampleRate = 44100
    loader = essentia.standard.MonoLoader(filename=filename)
    audio = loader()

    equalLoudness = essentia.standard.EqualLoudness()
    audio = equalLoudness(audio)

    pitchExtractor = essentia.standard.PitchMelodia(minFrequency=150, maxFrequency=500)
    pitch, pitchConfidence = pitchExtractor(audio)

    hopSize = 128

    finalPitch = []

    for n in range(int(len(audio) / sampleRate * noteRate)):
        pitchBegin = int(n / noteRate * sampleRate / hopSize)
        pitchEnd = int((n + 1) / noteRate * sampleRate / hopSize)
        pitchRange = pitch[pitchBegin : min(pitchEnd, len(pitch))]
        p = np.median(pitchRange)
        p = noteutils.round_to_nearest_key(p)
        finalPitch.append(p)

    if VISUALIZE:
        import matplotlib.pyplot as plt

        strechedNotes = np.repeat(finalPitch, 1 / noteRate * sampleRate / hopSize)
        plt.plot(np.arange(len(pitch)) * hopSize / sampleRate, pitch)
        plt.plot(np.arange(len(strechedNotes)) * hopSize / sampleRate, strechedNotes)
        plt.ylabel('Pitch (Hz)')
        plt.xlabel('Time (sec)')
        plt.title('Singing Pitch Detection')
        plt.show()

    return finalPitch

def extract_notes(filename, noteRate=8):
    pitch = extract_pitch(filename, noteRate)
    notes = noteutils.keys_to_notes(noteutils.pitch_to_keys(pitch))
    return notes

def extract_text(filename, noteRate=8):
    pitch = extract_pitch(filename, noteRate)
    text = noteutils.keys_to_text(noteutils.pitch_to_keys(pitch))
    return text

def main():
    parser = argparse.ArgumentParser(description='Extract pitch tracks from audio clips.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_audio')
    parser.add_argument('-o', '--output_file', help='Output audio file name')
    parser.add_argument('-t', '--note_rate', default=8, type=float, help='Notes in one second')
    parser.add_argument('--out_format', default='text', choices=['text', 'notes'], help='Output format')
    parser.add_argument('-v', '--visualize', action='store_true')
    args = parser.parse_args()

    global VISUALIZE
    VISUALIZE = args.visualize

    if args.out_format == 'text':
        output = extract_text(args.input_audio, args.note_rate)
    elif args.out_format == 'notes':
        output = extract_notes(args.input_audio, args.note_rate)
        output = ' '.join(output)

    if (args.output_file):
        with open(args.output_file, 'w') as f:
            f.write(output)

    print(output)

if __name__ == '__main__':
    main()
