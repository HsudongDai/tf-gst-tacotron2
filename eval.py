import argparse
import os
import re
import json
import numpy as np
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
from util import audio

def get_output_base_path(checkpoint_path, output_path=None):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  if output_path is not None:
    return os.path.join(output_path, name)
  else:
    return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  is_teacher_force = is_reference_mel = is_reference_weight = False
  mels_dir = args.mels_dir
  refs_dir = args.wave_dir
  refs_idx = args.wave_index
  refs_wet = args.refe_wet
  reference_mel = None
  reference_weight = None
  mel_targets = None
  if mels_dir is not None:
    is_teacher_force = True
  if refs_dir is not None:
    is_reference_mel = True
  if refs_wet is not None:
    is_reference_weight = True
  synth = Synthesizer(teacher_forcing_generating=is_teacher_force, reference_mel_generating=is_reference_mel, reference_weight_generating=is_reference_weight)
  synth.load(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint, args.out_dir)

  with open(args.text_path, 'r') as rf:
    text_list = [line.strip().split('|') for line in rf.readlines()]

  for index, text in text_list:
    if mels_dir is not None:
      mel_targets = os.path.join(mel_targets, f"{index}.npy")
    if refs_dir is not None:
      if refs_idx is not None:
        ref_wav = audio.load_wav(os.path.join(refs_dir, f"{refs_idx}.wav"))
      else:
        ref_wav = audio.load_wav(os.path.join(refs_dir, f"{index}.wav"))
      reference_mel = audio.melspectrogram(ref_wav).astype(np.float32).T
      wave_path = '%s_%s_ref-mel.wav' % (base_path, index)
      alignment_path = '%s_%s_ref-mel-align.png' % (base_path, index)
      weight_path = '%s_%s_ref-mel-weight.png' % (base_path, index)
      refer_path = '%s_%s_ref-mel-reference.npy' % (base_path, index)
      style_path = '%s_%s_ref-mel-style.npy' % (base_path, index)
    elif refs_wet is not None:
      reference_weight = json.loads(refs_wet)
      reference_weight = np.array(reference_weight).astype(np.float32)
      wave_path = '%s_%s_ref-%s.wav' % (base_path, index, 'specificWeight')
      alignment_path = '%s_%s_ref-%s-align.png' % (base_path, index, 'specificWeight')
      weight_path = '%s_%s_ref-%s-weight.png' % (base_path, index, 'specificWeight')
      refer_path = '%s_%s_ref-%s-reference.png' % (base_path, index, 'specificWeight')
      style_path = '%s_%s_ref-%s-style.png' % (base_path, index, 'specificWeight')
    else:
      if hparams.use_gst:
        wave_path = '%s_%s_ref-%s.wav' % (base_path, index, 'randomWeight')
        alignment_path = '%s_%s_ref-%s-align.png' % (base_path, index, 'randomWeight')
        weight_path = '%s_%s_ref-%s-weight.png' % (base_path, index, 'randomWeight')
        refer_path = '%s_%s_ref-%s-reference.png' % (base_path, index, 'randomWeight')
        style_path = '%s_%s_ref-%s-style.png' % (base_path, index, 'randomWeight')
      else:
        raise ValueError("You must set the reference audio if you don't want to use GSTs.")
  
    with open(wave_path, 'wb') as f:
      print('Synthesizing: %s' % text)
      print('Output wav file: %s' % wave_path)
      print('Output alignments: %s' % alignment_path)
      f.write(synth.synthesize(text, mel_targets=mel_targets, reference_mel=reference_mel, reference_weight=reference_weight, alignment_path=alignment_path, reference_path=refer_path, style_path=style_path, weight_path=weight_path))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint', help='Path to model checkpoint')
  parser.add_argument('text_path', help='text index|text phoneme sequence')
  parser.add_argument('--wave_dir', default=None, help='reference audio directory')
  parser.add_argument('--wave_index', default=None, help='reference audio index')
  parser.add_argument('--refe_wet', default=None, help='reference weight')
  parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--mels_dir', default=None, help='Mel-targets path, used when use teacher_force generation')
  parser.add_argument('--out_dir', default=None, help='output directory')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
