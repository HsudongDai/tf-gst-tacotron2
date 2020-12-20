# GST Tacotron (expressive end-to-end speech syntheis using global style token)

A tensorflow implementation of the [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017) and [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1803.09047).


* This set was trained using the **guojing** with global style tokens (GSTs).
* I found using appropriate style weight, single token can be used to synthesize different wave, the scale of the weight is effect the wave style.
* I found the synthesized audio can learn the prosody of the reference audio.
* The audio quality isn't so good as the paper. Maybe more data, more training steps and the wavenet vocoder will improve the quality, as well as better attention mechanism.
      

## Quick Start:

### Installing dependencies

1. Install Python 3 & TensorFlow 1.7.

2. Install requirements:
   ```shell
   pip install -r requirements.txt
   ```

### Training

1. **Preprocess dataset**

   for any data set, we need preprocess firstly by [data_processing](https://git.jd.com/ai-tts/data_processing), and get text phoneme sequences and scaled wave files. We need data format like this:
   ```python
   id_list.scp
   train.scp
   valid.scp
   metadata.csv
   scaled_wav\
      - 000001.wav
      - 000002.wav
      - 000003.wav
      - ...
   ```
   * id_list.scp is the index of text-audio pair, train.scp and valid.scp is sub-set of the id_list.scp
   * metadata.csv is text phoneme sequence file, like this:
   ```shell
   00000|B L AE K ss B Y UW T IY ss DH AH ss AO T AH B AY AA G R AH F IY ss AH V ss EY ss HH AO R S ss sil
   00001|B AY ss AE N AH ss S UW AH L ss sil
   00002|P ER F AO R M D ss B AY ss K AE TH ER AH N ss B AY ER Z ss sil
   ```


   Preprocess the data, extract lienar- and mel-spectrograms:
    
   ```
   python3 preprocess.py [id_list_path] [train_list_path] [valid_list_path] [phone_sequence_path] [wav_dir] [--out_path=output_dir]
   ```
   * The output_dir save all wave features, train.txt and valid.txt


2. **Train a model**

   ```
   python3 train.py [--base_dir=log_dir] [--input=input_dir] [--name=log_name] [--hparams=override_hyperparameters] [--summary_interval=summary_steps] [--checkpoint_interval=checkpoint_steps]

   ```
   checkpoint will be saved in [log_dir]/[log_name] floder.
   
   The above command line will use default hyperparameters, which will train a model with cmudict-based phoneme sequence and 4-head multi-head sytle attention for global style tokens. If you set the `use_gst=False` in the hparams, it will train a model like Google's another paper [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1803.09047).

   Tunable hyperparameters are found in [hparams.py](hparams.py). You can adjust these at the command line using the `--hparams` flag, for example `--hparams="batch_size=16,outputs_per_step=2"` . Hyperparameters should generally be set to the same values at both training and eval time.

4. **Synthesize from a checkpoint**
   ### synthesize by reference audio
   ```python
   python eval.py [checkpoint] [phone_sequence_file] [--wave_dir=reference_wave_dir] [--wave_index=reference_wave_index] [--out_dir=putput_dir]
   ```
   * phone_sequence_file: valid text phoneme sequence file, seperated by |
   * wave_index: reference wave index, all valid text use the same reference audio

   ### synthesize by specidfic style weight
   ```python
   python eval.py [checkpoint] [phone_sequence_file] [--refe_wet=style_token_weight] [--out_dir=putput_dir]
   ```
   * refe_wet: specific reference weight, the weight must consider the style embedding modulus

   If you set the `--hparams` flag when training, set the same value here.


## Reference
  -  Keithito's implementation of tacotron: https://github.com/keithito/tacotron
  -  Yuxuan Wang, Daisy Stanton, Yu Zhang, RJ Skerry-Ryan, Eric Battenberg, Joel Shor, Ying Xiao, Fei Ren, Ye Jia, Rif A. Saurous. 2018. [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017)
  - RJ Skerry-Ryan, Eric Battenberg, Ying Xiao, Yuxuan Wang, Daisy Stanton, Joel Shor, Ron J. Weiss, Rob Clark, Rif A. Saurous. 2018. [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1803.09047).
