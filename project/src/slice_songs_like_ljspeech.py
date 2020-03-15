import sys
import os
import multiprocessing
import json

from pydub import AudioSegment
import pydub

out_format = 'wav'
silence_threshold = 0.1

def usage():
    print('USAGE: {} vocals_dir lyrics_dir output_dir segment_length_second'.format(__file__));

def process_file(track_id, filename, vocals_dir, lyrics_dir, out_vocals_dir, segment_length):
    lyrics_path = os.path.join(lyrics_dir, filename.split('-')[0] + '.json')
    vocals_path = os.path.join(vocals_dir, filename)

    print('{} --- {} ({})'.format('p{:03d}'.format(track_id), filename.split('-')[0], os.getpid()))

    with open(lyrics_path) as f:
        lyrics = f.read()
        try:
            lyrics = json.loads(lyrics)
        except json.decoder.JSONDecodeError:
            # It happens for some of the tracks in DAMP
            print('Ignoring corrputed JSON file.')
            return []

    song = AudioSegment.from_file(vocals_path)
    total_rms = song.rms

    t = [part['t'] for part in lyrics]
    t.append(song.duration_seconds)
    l = [part['l'] for part in lyrics]

    segment_id = 1
    start = 0
    end = 1

    ljspeech_labels = []

    while end <= len(lyrics):
        # Merge together multiple pieces to reach the min length
        if t[end] - t[start] < segment_length:
            end += 1
            continue

        if end == len(lyrics):
            segment = song[t[start] * 1000 : ]
        else:
            segment = song[t[start] * 1000 : t[end] * 1000]

        segment_lyrics = ' '.join(l[start:end])

        start = end
        end = start + 1

        if segment.rms < total_rms * silence_threshold:
            #print('Ignoring silent segment')
            continue

        segment.duration_seconds

        while segment.duration_seconds > 1:
            if segment[0:1000].rms < (total_rms * silence_threshold):
                segment = segment[1000:]
            else:
                break

        while segment.duration_seconds > 1:
            if segment[-1000:].rms < (total_rms * silence_threshold):
                segment = segment[:-1000]
            else:
                break

        segment = pydub.effects.normalize(segment)

        segment_name = 'LJ{:03d}-{:04d}'.format(track_id, segment_id)
        out_segment_vocals_path = os.path.join(out_vocals_dir, '{}.{}'.format(segment_name, out_format))
        segment_id += 1

        ljspeech_labels.append('{}|{}|{}'.format(segment_name, segment_lyrics, segment_lyrics))
        segment.export(out_segment_vocals_path, format=out_format)

    return ljspeech_labels

def main(args):
    if len(args) != 1 + 4:
        usage()
        return

    vocals_dir = args[1]
    lyrics_dir = args[2]
    output_dir = args[3]
    segment_length = float(args[4])

    for directory in [vocals_dir, lyrics_dir]:
        if not os.path.isdir(directory):
            print('ERROR: Directory `{}` doesn\'t exist'.format(directory))
            return

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    out_vocals_dir = os.path.join(output_dir, 'wavs')
    if not os.path.isdir(out_vocals_dir):
        os.mkdir(out_vocals_dir)

    input_files = os.listdir(vocals_dir)

    # Run using a process pool
    process_args = [(i + 1, filename, vocals_dir, lyrics_dir, out_vocals_dir, segment_length) for i, filename in enumerate(input_files)]
    pool = multiprocessing.Pool(8)
    results = pool.starmap(process_file, process_args)
    results = [line for group in results for line in group]
    #results = []
    #for i, filename in enumerate(input_files):
    #    new_labels = process_file(i + 1, filename, vocals_dir, lyrics_dir, out_vocals_dir, segment_length)
    #    results.extend(new_labels)

    with open(os.path.join(output_dir, 'metadata.csv'), 'w') as f:
        f.write('\n'.join(sorted(results)))

if __name__ == '__main__':
    main(sys.argv)
