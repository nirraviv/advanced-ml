#!/usr/bin/env python3

import sys
import os
import multiprocessing
import json
import argparse

import audio_to_pitch

from pydub import AudioSegment
import pydub

def process_file(args, vocals_path, out_vocals_path_format):
    print('{} ({})'.format(vocals_path, os.getpid()))

    OUT_AUDIO_FORMAT = args.out_audio_format
    SILENCE_THRESHOLD = args.silence_threshold
    FADE_LENGTH = args.fade_length
    SEGMENT_LENGTH = args.segment_length
    NOTE_RATE = args.note_rate
    HOP_LENGTH = args.hop_length or SEGMENT_LENGTH

    if not os.path.isfile(vocals_path):
        print('Ignoring nonexistent vocals file: {}'.format(vocals_path))
        return []

    song = AudioSegment.from_file(vocals_path)
    total_rms = song.rms

    lyric_segments = []

    segment_id = 1

    t = 0
    while t + SEGMENT_LENGTH < song.duration_seconds:
        segment = song[t * 1000 : (t + SEGMENT_LENGTH) * 1000]
        t += HOP_LENGTH

        if segment.rms < total_rms * SILENCE_THRESHOLD:
            #print('Ignoring silent segment')
            continue

        while segment.duration_seconds > 1:
            if segment[0:1000].rms < (total_rms * SILENCE_THRESHOLD):
                segment = segment[1000:]
            else:
                break

        while segment.duration_seconds > 1:
            if segment[-1000:].rms < (total_rms * SILENCE_THRESHOLD):
                segment = segment[:-1000]
            else:
                break

        segment = pydub.effects.normalize(segment)
        segment = segment.fade_in(int(FADE_LENGTH * 1000)).fade_out(int(FADE_LENGTH * 1000))

        out_segment_vocals_path = out_vocals_path_format.format(segment_id) + '.' + OUT_AUDIO_FORMAT
        segment_id += 1

        segment.export(out_segment_vocals_path, format=OUT_AUDIO_FORMAT)
        segment_lyrics = audio_to_pitch.extract_text(out_segment_vocals_path, NOTE_RATE)
        lyric_segments.append(segment_lyrics)

    return lyric_segments

def process_file_multi(args, vocals_paths, out_vocals_path_formats):
    PARALLEL = True

    if PARALLEL:
        process_args = zip([args] * len(vocals_paths), vocals_paths, out_vocals_path_formats)
        pool = multiprocessing.Pool(4)
        results = pool.starmap(process_file, process_args)
        return results
    else:
        results = []
        for i in range(len(vocals_paths)):
            segment_lyrics = process_file(args, vocals_paths[i], out_vocals_path_formats[i])
            results.extend(segment_lyrics)
        return results

def process_like_vctk(args, vocals_paths, output_dir):
    out_vocals_dir = os.path.join(output_dir, 'wav48') # is it really WAV 48?
    out_lyrics_dir = os.path.join(output_dir, 'txt')
    for directory in [out_vocals_dir, out_lyrics_dir]:
        os.makedirs(directory, exist_ok=True)

    out_vocals_subdirs = [os.path.join(out_vocals_dir, 'p{:03d}'.format(i + 1)) for i in range(len(vocals_paths))]
    out_lyrics_subdirs = [os.path.join(out_lyrics_dir, 'p{:03d}'.format(i + 1)) for i in range(len(vocals_paths))]
    for directory in out_vocals_subdirs + out_lyrics_subdirs:
        os.makedirs(directory, exist_ok=True)

    out_vocals_path_formats = [os.path.join(out_vocals_subdirs[i], 'p{:03}_{{:03}}'.format(i + 1)) for i in range(len(vocals_paths))]

    all_lyrics = process_file_multi(args, vocals_paths, out_vocals_path_formats)

    for i, song_lyrics in enumerate(all_lyrics):
        print('Generating lyrics for p{:03}'.format(i))
        for k, segment_lyrics in enumerate(song_lyrics):
            out_segment_lyrics_path = os.path.join(out_lyrics_subdirs[i], 'p{:03}_{:03}.{}'.format(i + 1, k + 1, 'txt'))
            with open(out_segment_lyrics_path, 'w') as f:
                f.write(segment_lyrics)

def process_like_ljspeech(args, vocals_paths, output_dir):
    out_vocals_dir = os.path.join(output_dir, 'wavs')
    os.makedirs(out_vocals_dir, exist_ok=True)

    out_vocals_path_formats = ['LJ{:03d}-{{:04d}}'.format(i + 1) for i in range(len(vocals_paths))]
    out_vocals_path_formats = [os.path.join(out_vocals_dir, f) for f in out_vocals_path_formats]

    all_lyrics = process_file_multi(args, vocals_paths, out_vocals_path_formats)

    metadata = []
    for i, group_lyrics in enumerate(all_lyrics):
        for j, segment_lyrics in enumerate(group_lyrics):
            metadata.append('{}|{}|{}'.format('LJ{:03d}-{:04d}'.format(i + 1, j + 1), segment_lyrics, segment_lyrics))

    with open(os.path.join(output_dir, 'metadata.csv'), 'w') as f:
        f.write('\n'.join(sorted(metadata)))

def main():
    parser = argparse.ArgumentParser(description='Filter, segment and organize the Sing!300x30x2 dataset',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--segment_length', default=5, type=float, help='preferred length in seconds')
    parser.add_argument('--hop_length', type=float, help='hop length in seconds')
    parser.add_argument('--note_rate', default=8, type=float, help='notes per second for pitch extraction')
    parser.add_argument('--out_format', default='ljspeech', choices=['vctk', 'ljspeech'], help='format of the output dataset')
    parser.add_argument('--out_audio_format', default='wav', help='file format of the output audio files')
    parser.add_argument('--silence_threshold', default=0.1, type=float, help='threshold level for trimming silence')
    parser.add_argument('--fade_length', default=0.2, type=float, help='in/out face effect length in seconds')
    args = parser.parse_args()

    input_files = os.listdir(args.input_dir)
    input_files = [os.path.join(args.input_dir, f) for f in input_files]

    os.makedirs(args.output_dir, exist_ok=True)

    if args.out_format == 'vctk':
        process_like_vctk(args, input_files, args.output_dir)
    elif args.out_format == 'ljspeech':
        process_like_ljspeech(args, input_files, args.output_dir)
    else: assert False

if __name__ == '__main__':
    main()
