# RNNoise: a learned denoiser as an audio processor (plan 31, P5)

`Idrak.Audio.Plugins.RnNoise`; `RnNoisePlugin.Register()` registers it alone, `AudioPlugins.RegisterAll()` with the
package's other plug-ins, and the `idrak` tool registers it at startup. It registers the live audio processor
`rnnoise` in `AudioProcessors`; the library has no learned denoiser of its own, so without the plug-in `rnnoise` is an
unknown processor.

A C# port of xiph's RNNoise, its last classic release (tag v0.1.1, https://github.com/xiph/rnnoise; Jean-Marc Valin's
"A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement", 2018):

| Part | File | What |
|---|---|---|
| Frames | `RnNoiseProcessor.cs` (`RnNoiseStream`) | each channel gathered into frames of 480 samples at 48 kHz, scaled to 16-bit range (×32768, divided out after), each frame's output queued; a sample taken gives back the one 959 samples before it |
| Analysis and synthesis | `RnNoiseDenoiser.cs` | RNNoise's high-pass (b = (-2, 1), a = (-1.99599, 0.996)), the 960-sample power-complementary window over the previous frame and this one, the spectrum, 22 bands (triangles between RNNoise's band edges, 200 Hz apart at the bottom, 4.4 kHz at the top), the inverse transform overlap-added |
| Features | `RnNoiseDenoiser.cs` | the 22 band energies as a DCT of their logs (with a floor following the louder bands), their first and second differences over three frames, the pitch correlation per band as a DCT, the pitch period, the spectral variability over 8 frames: 42 values; a frame under RNNoise's energy floor is silent (passed through, the network not run) |
| Pitch | `RnNoiseDenoiser.cs` | CELT's pitch search: the signal halved and whitened by a 4th-order LPC, a coarse search decimated by 4, a fine one by 2, octave errors removed (`remove_doubling`), the gain |
| Network | `RnNoiseModel.cs` | a dense layer (42 to 24, tanh), the VAD GRU (24, ReLU) and its output (sigmoid), the noise GRU (48), the denoise GRU (96), the 22 band gains (sigmoid); RNNoise's tanh and sigmoid approximations (its table of 201 values) |
| Gains | `RnNoiseDenoiser.cs` | the pitch filter (the pitch-delayed spectrum added where a band is periodic, then each band brought back to its energy), the gains held from falling faster than 0.6 a frame, interpolated over the bins (0 above 20 kHz) |
| FFT | `RnNoiseFft.cs` | KISS FFT in float, its radix 3, 4 and 5 butterflies for 960 points |

Every computation is the C code's, in its order and in float (in double where the C code promotes to double), so the
samples are RNNoise's. The dense and GRU layers take one input at a time for all neurons at once (SIMD rows of the
weights, which RNNoise stores input by input), and the pitch correlation one lag vector at a time: neither changes the
order of any sum. Nothing is allocated per frame or, at 48 kHz, per block; a long block's channels run on separate
cores. Measured on a 4-thread x64 container (AVX2, busy): 80 to 110 µs a frame against 130 to 137 µs for the C
reference built at `-O2` on the same machine; 10 s of 16 kHz mono or 48 kHz stereo in about 85 ms warm, about 115
times real time (`benches/Idrak.Benches -- audio-process`).

## Checked against RNNoise

`tools/audio/make_rnnoise_reference.py` clones RNNoise v0.1.1, builds it with the system C compiler (`-O2
-ffp-contract=off`), and runs it on 2 s of speech-like sound at 48 kHz (silence, three voiced syllables and a fricative
over a noise floor); `tests/Idrak.Tests/data/audio/rnnoise/` holds the input, RNNoise's output and voice
probabilities, and the model file. The tests (`IDRAK_FILTER="audio processing: rnnoise"`) check the samples within
1e-4 (the processor's Apply takes the 959-sample delay off) and the voice probabilities within 1e-4; where the platform's `log10`, `sin` and `cos`
are glibc's (the C reference's), the samples come out bit for bit. They also check block-by-block runs (blocks of 1, 7,
160 and 4,096) against the whole recording bit for bit at 48 kHz (mono, stereo, with mix and attenuation) and at
16 kHz, no allocation per block, the text form and every refusal.

## The model file

RNNoise v0.1.1 carries its trained weights as C arrays (`src/rnn_data.c`); its `rnnoise_model_from_file` reads a text
format instead, which this plug-in reads too: the line `rnnoise-nu model file version 1`, then for the dense input
layer, the VAD GRU, the noise GRU, the denoise GRU, the denoise output and the VAD output: inputs, neurons, activation
(0 tanh, 1 sigmoid, 2 ReLU), the weights, a GRU's recurrent weights and the biases, as whole numbers of a signed byte
(the weights scaled by 1/256 when used). The layers' sizes are checked against what RNNoise feeds them; a file that is
not a model, is cut short, holds a weight outside a byte or a layer of another size is refused naming the file and
what is wrong. A model is read once per process (again only when its file changes).

`make_rnnoise_reference.py --model-only PATH` writes xiph's classic weights in that format (the reference run checks
the file's output against the built-in weights bit for bit). It is looked for at the step's `model=PATH` (or a name
in the cache's rnnoise folder), else `IDRAK_RNNOISE_MODEL`, else `rnnoise/rnnoise.rnnn` under `IDRAK_CACHE` (or
`~/.cache/idrak`). RNNoise's newer releases (v0.2 on) use another network and weight format; this plug-in reads the
classic one.

## Licence

The port keeps RNNoise's licence for its parts (the files say which): BSD-3-Clause, as in RNNoise's `COPYING`:

```
Copyright (c) 2017, Mozilla
Copyright (c) 2007-2017, Jean-Marc Valin
Copyright (c) 2005-2017, Xiph.Org Foundation
Copyright (c) 2003-2004, Mark Borgerding

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

- Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

- Neither the name of the Xiph.Org Foundation nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

`src/rnn_reader.c` (the model file format) is Copyright (c) 2018 Gregor Richards, and `src/rnn.c` Copyright (c)
2008-2011 Octasic Inc., 2012-2017 Jean-Marc Valin, under the same terms.
