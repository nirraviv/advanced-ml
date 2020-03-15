import sys
import os
import multiprocessing
import argparse
import itertools

from pydub import AudioSegment
import pydub

def process_file(args, filename):
    #print('{} ({})'.format(filename, os.getpid()))

    assert os.path.isfile(filename)
    song = AudioSegment.from_file(filename)

    segment_length = args.segment_length
    silence_threshold = args.silence_threshold

    length = song.duration_seconds
    segment_count = int(length / segment_length)
    total_rms = song.rms

    silent_segments = 0

    for i in range(segment_count):
        segment = song[i * segment_length * 1000 : (i + 1) * segment_length * 1000]
        if segment.rms < total_rms * silence_threshold:
            silent_segments += 1

    silence = silent_segments / segment_count
    is_noisy = silence < 0.05

    if is_noisy:
        print(filename)

    return is_noisy

def process_file_multi(args, files):
    PARALLEL = True

    if PARALLEL:
        process_args = ((args, f) for f in files)
        pool = multiprocessing.Pool(8)
        return pool.starmap(process_file, process_args)
    else:
        return [process_file(args, f) for f in files]

def main():
    parser = argparse.ArgumentParser(description='Find noisy audio files',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir')
    parser.add_argument('--segment_length', default=0.2, type=float, help='analysis segment length in seconds')
    parser.add_argument('--silence_threshold', default=0.1, type=float, help='analysis segment length in seconds')
    parser.add_argument('--move_to')
    args = parser.parse_args()

    input_dir = args.input_dir
    move_to = args.move_to

    if not os.path.isdir(input_dir):
        print('ERROR: `{}` is not a directory'.format(input_dir))
        return

    input_files = os.listdir(input_dir)
    input_files = [os.path.join(input_dir, f) for f in input_files]

    is_noisy = process_file_multi(args, input_files)
    noisy_files = itertools.compress(input_files, is_noisy)

    if args.move_to:
        print('Moving noisy files to {}'.format(args.move_to))
        os.makedirs(move_to, exist_ok=True)
        for f in noisy_files:
            os.rename(f, os.path.join(args.move_to, os.path.basename(f)))

if __name__ == '__main__':
    main()
