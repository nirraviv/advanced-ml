import math

"""
notes (n): a4 A4 b4 B4 c4 d4
keys  (k): 48 49 50 51 52 53
pitch (p): 440 446.16 493.88 ...
"""

def key_to_pitch(k):
    if k < 0:
        return 0
    assert 0 <= k < 88
    a4_freq = 440
    a4_index = 48
    semitone = 2 ** (1 / 12)
    return a4_freq * (semitone ** (k - a4_index))

def pitch_to_key(p):
    a4_freq = 440
    a4_index = 48
    semitone = 2 ** (1 / 12)

    if p <= 0:
        return -1

    return int(round(math.log(p / a4_freq, semitone) + a4_index))

def keys_to_pitch(keys):
    return [key_to_pitch(k) for k in keys]

def pitch_to_keys(pitch_list):
    return [pitch_to_key(p) for p in pitch_list]

def round_to_nearest_key(p):
    k = pitch_to_key(p)
    if k < 0 or k >= 88:
        return 0
    else:
        return key_to_pitch(k)

number_in_octave = {
        'c': 0,
        'd': 2,
        'e': 4,
        'f': 5,
        'g': 7,
        'a': 9,
        'b': 11,
        }

def note_to_key(note):
    try:
        if note == 'x':
            return -1

        letter = note[0]
        octave = int(note[1])
        key = number_in_octave[letter.lower()] - 9 + octave * 12
        if letter.isupper():
            key += 1

        assert key >= 0 and key < 88
        return key
    except:
        raise ValueError('Invalid note', note)

def key_to_note(key):
    if key < 0:
        return 'x'
    return 'aAbcCdDefFgG'[key % 12] + str((key + 9) // 12)

def notes_to_keys(notes):
    return [note_to_key(n) for n in notes]

def keys_to_notes(keys):
    return [key_to_note(n) for n in keys]

def notes_to_pitch(note_list):
    pitch_list = []
    for n in note_list:
        if n == 'x':
            pitch_list.append(0)
        else:
            pitch_list.append(key_to_pitch(note_to_key(n)))

    return pitch_list

def keys_to_text(keys):
    a2_index = 2 * 12
    a3_index = 3 * 12
    a5_index = 5 * 12
    a6_index = 6 * 12

    text = ''
    for k in keys:
        if k < 0 or k >= 88:
            text += 'z'
        elif a3_index <= k < a5_index:
            text += chr(ord('a') + k - a3_index)
        elif a5_index <= k < a6_index:
            text += chr(ord('A') + k - a5_index)
        elif a2_index <= k < a3_index:
            text += chr(ord('A') + 12 + k - a2_index)
        else:
            text += '?'

    return text

def text_to_keys(text):
    a2_index = 2 * 12
    a3_index = 3 * 12
    a5_index = 5 * 12
    a6_index = 6 * 12

    keys = []

    for c in text:
        if 'a' <= c <= 'x':
            keys.append(ord(c) - ord('a') + a3_index)
        elif 'A' <= c <= 'L':
            keys.append(ord(c) - ord('A') + a5_index)
        elif 'M' <= c <= 'X':
            keys.append(ord(c) - ord('M') + a2_index)
        else:
            keys.append(0)

    return keys
