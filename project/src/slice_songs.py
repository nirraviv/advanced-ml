#!/usr/bin/python3

import sys
import os
import multiprocessing

from pydub import AudioSegment
import pydub

out_format = 'wav'
silence_threshold = 0.1

def usage():
    print('USAGE: {} input_dir output_dir segment_length_seconds'.format(__file__));

def process_file(filename, input_dir, output_dir, segment_length):
    in_path = os.path.join(input_dir, filename)
    print('{} - {}'.format(os.getpid(), in_path))
    song = AudioSegment.from_file(in_path)

    length = song.duration_seconds
    segment_count = int(length / segment_length)

    total_rms = song.rms

    for i in range(segment_count):
        segment = song[i * segment_length * 1000 : (i + 1) * segment_length * 1000]

        if segment.rms < total_rms * silence_threshold:
            #print('Ignoring silent segment')
            continue

        inner_silence = [segment[t * 1000 : (t + 1) * 1000].rms < (total_rms * silence_threshold) for t in range(int(segment_length))]
        trim_start = inner_silence.index(False)
        trim_end = inner_silence[::-1].index(False)
        segment = segment[trim_start * 1000 : (len(inner_silence) - trim_end) * 1000]
        #print('Trimming {} left {} right'.format(trim_start, trim_end))

        segment = pydub.effects.normalize(segment)

        out_path = os.path.join(output_dir, '{}_{}.{}'.format(os.path.splitext(filename)[0], i, out_format))
        segment.export(out_path, format=out_format)

def main(args):
    if len(args) != 1 + 3:
        usage()
        return

    input_dir = args[1]
    output_dir = args[2]
    segment_length = float(args[3])

    if not os.path.isdir(input_dir):
        print('ERROR: Input directory `{}` doesn\'t exist'.format(input_dir))
        return

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    input_files = os.listdir(input_dir)

    # Run using a process pool
    process_args = [(filename, input_dir, output_dir, segment_length) for filename in input_files]
    pool = multiprocessing.Pool(8)
    pool.starmap(process_file, process_args)
    #for filename in input_files:
    #    process_file(filename, input_dir, output_dir, segment_length)

if __name__ == '__main__':
    main(sys.argv)
