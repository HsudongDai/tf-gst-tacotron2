import argparse
import codecs
import os
from multiprocessing import Pool, cpu_count

from hparams import hparams as hp
import numpy as np
from util import audio
from util.display import simple_table, progbar, stream


def get_files(indexs, wave_dir):
    print(f"Process waves in {wave_dir}")
    files = [os.path.join(wave_dir, id + ".wav") for id in indexs]
    return files


def metadata(path):
    text_dict = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            split = line.split('|')
            text_dict[split[0]] = split[-1]
    return text_dict


def convert_file(audio_path):
    y = audio.load_wav(audio_path)
    peak = np.abs(y).max()
    if hp.peak_norm or peak > 1.0:
        y *= (0.9 / peak)

    linear = audio.spectrogram(y)
    mel = audio.melspectrogram(y)
    return mel.astype(np.float32), linear.astype(np.float32)


def process_wav(params):
    wav_path, output_path = params[0], params[1]
    id = wav_path.split(os.path.sep)[-1][:-4]
    mel_spectrum, linear_spectrum = convert_file(wav_path)
    np.save(os.path.join(output_path, 'mel', '{}.npy'.format(id)), mel_spectrum, allow_pickle=False)
    np.save(os.path.join(output_path, 'linear', '{}.npy'.format(id)), linear_spectrum, allow_pickle=False)
    return id, mel_spectrum.shape[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing for Tacotron')
    parser.add_argument('whole_list', help='whole index list file path.')
    parser.add_argument('train_list', help='train index list file path.')
    parser.add_argument('valid_list', help='valid index list file path.')
    parser.add_argument('meta_path', help='metadata file path.')
    parser.add_argument('wave_dir', help='wave directory.')
    parser.add_argument('--out_path', '-o', default='training/',
                        help='file extension to search for in dataset folder')
    args = parser.parse_args()

    with open(args.whole_list, 'r') as rf:
        whole_ids = [line.strip() for line in rf.readlines()]

    with open(args.train_list, 'r') as rf:
        train_ids = [line.strip() for line in rf.readlines()]

    with open(args.valid_list, 'r') as rf:
        valid_ids = [line.strip() for line in rf.readlines()]

    meta_path = args.meta_path
    wave_dir = args.wave_dir
    out_path = args.out_path

    wav_files = get_files(whole_ids, wave_dir)

    out_mel = os.path.join(out_path, 'mel')
    out_linear = os.path.join(out_path, 'linear')
    os.makedirs(out_mel, exist_ok=True)
    os.makedirs(out_linear, exist_ok=True)

    print(f'\n{len(wav_files)} files found in "{wave_dir}"\n')

    if len(wav_files) == 0:
        raise ValueError(f"There is no wave file to process. Please check {args.whole_list}")
    else:
        text_dict = metadata(meta_path)
        simple_table([('Sample Rate', hp.sample_rate),
                      ('CPU Count', cpu_count() - 12)])

        pool = Pool(processes=cpu_count() - 12)
        dataset = {}

        params = []
        for f in wav_files:
            params.append((f, out_path))

        for i, (id, length) in enumerate(pool.imap_unordered(process_wav, params), 1):
            dataset[id] = (id, length, text_dict[id])
            bar = progbar(i, len(wav_files))
            message = f'{bar} {i}/{len(wav_files)} '
            stream(message)

        with codecs.open(os.path.join(out_path, 'train.txt'), 'w', 'utf-8') as f:
            for id in train_ids:
                item = dataset[id]
                line = str(item[0]) + '|' + str(item[1]) + '|' + item[2] + '\n'
                f.write(line)

        print('\n\ntrain set size is {}'.format(len(train_ids)))

        with codecs.open(os.path.join(out_path, 'valid.txt'), 'w', 'utf-8') as f:
            for id in valid_ids:
                item = dataset[id]
                line = str(item[0]) + '|' + str(item[1]) + '|' + item[2] + '\n'
                f.write(line)

        print('valid set size is {}'.format(len(valid_ids)))

        print('\n\nCompleted. Ready to run "python train.py" \n')
