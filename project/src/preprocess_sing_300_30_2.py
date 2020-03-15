import sys
import os
import multiprocessing
import json
import csv
import argparse

from pydub import AudioSegment
import pydub

def process_file(args, vocals_path, lyrics_path, out_vocals_path_format):
    print('{}, {} ({})'.format(vocals_path, lyrics_path, os.getpid()))

    OUT_AUDIO_FORMAT = args.out_audio_format
    SILENCE_THRESHOLD = args.silence_threshold
    FADE_LENGTH = args.fade_length
    SEGMENT_LENGTH = args.segment_length
    MAX_LENGTH = args.max_segment_length

    if not os.path.isfile(lyrics_path):
        print('Ignoring nonexistent JSON file: {}'.format(lyrics_path))
        return []

    with open(lyrics_path) as f:
        try:
            lyrics = json.load(f)
        except json.decoder.JSONDecodeError:
            # It happens for some of the tracks in DAMP
            print('Ignoring corrputed JSON file: {}'.format(lyrics_path))
            return []

    if not os.path.isfile(vocals_path):
        print('Ignoring nonexistent vocals file: {}'.format(vocals_path))
        return []

    song = AudioSegment.from_file(vocals_path)
    total_rms = song.rms

    t = [part['t'] for part in lyrics]
    t.append(song.duration_seconds)
    l = [part['l'] for part in lyrics]

    segment_id = 1
    start = 0
    end = 1

    lyric_segments = []

    while end <= len(lyrics):
        # Merge together multiple pieces to reach the min length
        if t[end] - t[start] < SEGMENT_LENGTH:
            end += 1
            continue

        if end == len(lyrics):
            segment = song[t[start] * 1000 : ]
        else:
            segment = song[t[start] * 1000 : t[end] * 1000]

        segment_lyrics = ' '.join(l[start:end])

        start = end
        end = start + 1

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

        if segment.duration_seconds > MAX_LENGTH:
            # A very long segment can crash some networks
            print('Ignoring too long segment')
            continue

        segment = pydub.effects.normalize(segment)
        segment = segment.fade_in(int(FADE_LENGTH * 1000)).fade_out(int(FADE_LENGTH * 1000))

        out_segment_vocals_path = out_vocals_path_format.format(segment_id) + '.' + OUT_AUDIO_FORMAT
        segment_id += 1

        lyric_segments.append(segment_lyrics)
        segment.export(out_segment_vocals_path, format=OUT_AUDIO_FORMAT)

    return lyric_segments

def process_file_multi(args, vocals_paths, lyrics_paths, out_vocals_path_formats):
    PARALLEL = True

    if PARALLEL:
        process_args = zip([args] * len(vocals_paths), vocals_paths, lyrics_paths, out_vocals_path_formats)
        pool = multiprocessing.Pool(8)
        results = pool.starmap(process_file, process_args)
        return results
    else:
        results = []
        for i in range(vocals_paths):
            segment_lyrics = process_file(args, vocals_paths[i], lyrics_paths[i], out_vocals_path_formats[i])
            results.extend(segment_lyrics)
        return results

def process_like_vctk(args, vocals_paths, lyrics_paths, output_dir):
    out_vocals_dir = os.path.join(output_dir, 'wav48') # is it really WAV 48?
    out_lyrics_dir = os.path.join(output_dir, 'txt')
    for directory in [out_vocals_dir, out_lyrics_dir]:
        os.makedirs(directory, exist_ok=True)

    out_vocals_subdirs = [os.path.join(out_vocals_dir, 'p{:03d}'.format(i + 1)) for i in range(len(vocals_paths))]
    out_lyrics_subdirs = [os.path.join(out_lyrics_dir, 'p{:03d}'.format(i + 1)) for i in range(len(vocals_paths))]
    for directory in out_vocals_subdirs + out_lyrics_subdirs:
        os.makedirs(directory, exist_ok=True)

    out_vocals_path_formats = [os.path.join(out_vocals_subdirs[i], 'p{:03}_{{:03}}'.format(i + 1)) for i in range(len(vocals_paths))]

    all_lyrics = process_file_multi(args, vocals_paths, lyrics_paths, out_vocals_path_formats)

    for i, song_lyrics in enumerate(all_lyrics):
        print('Generating lyrics for p{:03}'.format(i))
        for k, segment_lyrics in enumerate(song_lyrics):
            out_segment_lyrics_path = os.path.join(out_lyrics_subdirs[i], 'p{:03}_{:03}.{}'.format(i + 1, k + 1, 'txt'))
            with open(out_segment_lyrics_path, 'w') as f:
                f.write(segment_lyrics)

def process_like_ljspeech(args, vocals_paths, lyrics_paths, output_dir):
    out_vocals_dir = os.path.join(output_dir, 'wavs')
    os.makedirs(out_vocals_dir, exist_ok=True)

    out_vocals_path_formats = ['LJ{:03d}-{{:04d}}'.format(i + 1) for i in range(len(vocals_paths))]
    out_vocals_path_formats = [os.path.join(out_vocals_dir, f) for f in out_vocals_path_formats]

    all_lyrics = process_file_multi(args, vocals_paths, lyrics_paths, out_vocals_path_formats)

    metadata = []
    for i, group_lyrics in enumerate(all_lyrics):
        for j, segment_lyrics in enumerate(group_lyrics):
            metadata.append('{}|{}|{}'.format('LJ{:03d}-{:04d}'.format(i + 1, j + 1), segment_lyrics, segment_lyrics))

    with open(os.path.join(output_dir, 'metadata.csv'), 'w') as f:
        f.write('\n'.join(sorted(metadata)))

def main():
    parser = argparse.ArgumentParser(description='Filter, segment and organize the Sing!300x30x2 dataset',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--segment_length', default=5, type=float, help='preferred length in seconds')
    parser.add_argument('--country', default='US', help='comma separated list of selected countries. `ALL` to use all.')
    parser.add_argument('--gender', default='A', choices=['A', 'M', 'F'], help='selected gender')
    parser.add_argument('--followers_min', default=10, type=int, help='filter by followers')
    parser.add_argument('--out_format', default='ljspeech', choices=['vctk', 'ljspeech'], help='format of the output dataset')
    parser.add_argument('--out_audio_format', default='wav', help='file format of the output audio files')
    parser.add_argument('--silence_threshold', default=0.1, type=float, help='threshold level for trimming silence')
    parser.add_argument('--fade_length', default=0.2, type=float, help='in/out face effect length in seconds')
    parser.add_argument('--max_segment_length', default=24, type=float, help='absolute max segment length limit')
    args = parser.parse_args()

    country_set = set(args.country.split(','))

    input_vocals = []
    input_lyrics = []

    with open(os.path.join(args.dataset_dir, 'perfs.csv')) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            if row['gender'] != args.gender and args.gender != 'A':
                continue
            if row['country'] not in country_set and 'ALL' not in country_set:
                continue
            if int(row['followers']) < args.followers_min:
                continue
            name_fields = ['arrangement_id', 'performance_id', 'country', 'gender', 'account_id']
            vocals_file = '{}-{}-{}-{}-{}.m4a'.format(*[row[field] for field in name_fields])
            lyrics_file = '{}.json'.format(row['arrangement_id'])

            curr_country = row['country']
            input_vocals.append(os.path.join(args.dataset_dir, curr_country, curr_country + 'Vocals', vocals_file))
            input_lyrics.append(os.path.join(args.dataset_dir, curr_country, curr_country + 'Lyrics', lyrics_file))

    print(input_vocals)
    print(input_lyrics)
    print('Selected {} performances'.format(len(input_vocals)))

    os.makedirs(args.output_dir, exist_ok=True)

    if args.out_format == 'vctk':
        process_like_vctk(args, input_vocals, input_lyrics, args.output_dir)
    elif args.out_format == 'ljspeech':
        process_like_ljspeech(args, input_vocals, input_lyrics, args.output_dir)
    else: assert False

if __name__ == '__main__':
    main()
