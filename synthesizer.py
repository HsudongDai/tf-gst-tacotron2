import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from models import create_model
from text import text_to_sequence
from util import audio, plot
import textwrap


class Synthesizer:
  def __init__(self, teacher_forcing_generating=False, reference_mel_generating=False, reference_weight_generating=True):
    self.teacher_forcing_generating = teacher_forcing_generating
    self.reference_mel_generating = reference_mel_generating
    self.reference_weight_generating = reference_weight_generating
  def load(self, checkpoint_path, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths') 
    # using mel-reference get style embedding
    if self.reference_mel_generating:
      reference_mel = tf.placeholder(tf.float32, [1, None, hparams.num_mels], 'reference_mel')
    else:
      reference_mel = None
    # using weights get style embedding
    if self.reference_weight_generating:
      reference_weight = tf.placeholder(tf.float32, [hparams.num_gst], 'reference_weight')
    else:
      reference_weight = None
    # Only used in teacher-forcing generating mode
    if self.teacher_forcing_generating:
      mel_targets = tf.placeholder(tf.float32, [1, None, hparams.num_mels], 'mel_targets')
    else:
      mel_targets = None

    with tf.variable_scope('model') as scope:
      self.model = create_model(model_name, hparams)
      self.model.initialize(inputs, input_lengths, mel_targets=mel_targets, reference_mel=reference_mel, reference_weight=reference_weight)
      self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])
      self.alignments = self.model.alignments[0]
      #self.reference_embeddings = self.model.refnet_outputs[0]
      self.style_embeddings = self.model.style_embeddings[0]
      self.style_weights = self.model.style_weights[0]

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, text, mel_targets=None, reference_mel=None, reference_weight=None, alignment_path=None, reference_path=None, style_path=None, weight_path=None):
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    seq = text_to_sequence(text, cleaner_names)
    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32),
    }
    if mel_targets is not None:
      mel_targets = np.expand_dims(mel_targets, 0)
      feed_dict.update({self.model.mel_targets: np.asarray(mel_targets, dtype=np.float32)})
    elif reference_mel is not None:
      reference_mel = np.expand_dims(reference_mel, 0)
      feed_dict.update({self.model.reference_mel: np.asarray(reference_mel, dtype=np.float32)})
    elif reference_weight is not None:
      feed_dict.update({self.model.reference_weight: np.asarray(reference_weight, dtype=np.float32)})

    wav, alignments, style_embeddings, style_weights = self.session.run([self.wav_output, self.alignments, self.style_embeddings, self.style_weights], feed_dict=feed_dict)
    wav = audio.inv_preemphasis(wav)
    end_point = audio.find_endpoint(wav)
    wav = wav[:end_point]
    out = io.BytesIO()
    audio.save_wav(wav, out)
    n_frame = int(end_point / (hparams.frame_shift_ms / 1000* hparams.sample_rate)) + 1
    text = '\n'.join(textwrap.wrap(text, 70, break_long_words=False))
    plot.plot_alignment(alignments[:,:n_frame], alignment_path, info='%s' % (text))
    plot.plot_weight(style_weights, weight_path)
    # np.save(reference_path, refer_embeddings)
    np.save(style_path, style_embeddings)
    return out.getvalue()
