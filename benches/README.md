# Benches

`Idrak.Benches` measures the library through its public API on every device `IDRAK_DEVICES` names (default: every GPU
found, else the CPU), and stresses it. It is the place to find a bottleneck and to prove a speed or memory change: run it
before and after, then compare.

```
dotnet run -c Release --project benches/Idrak.Benches                      # every section
dotnet run -c Release --project benches/Idrak.Benches -- ops train-decoder # some
IDRAK_BENCH_OUT=before.json dotnet run -c Release --project benches/Idrak.Benches
# ... change the library ...
IDRAK_BENCH_OUT=after.json dotnet run -c Release --project benches/Idrak.Benches
dotnet run -c Release --project benches/Idrak.Benches -- compare before.json after.json
```

| Section | What it measures |
|---|---|
| `ops` | element-wise operations, reductions, softmax, layer norm, weight norm: µs a call, GB/s, launches; forward and with the backward step |
| `matmul` | matrix products (TFLOPS) and convolutions (forward; forward and both gradients; launches) at training and decoding shapes |
| `train-decoder` | a decoder from a spec with random weights: ms a step, launches a step by operation, peak memory, tokens a second, with and without activation checkpointing |
| `infer-decoder` | the decoder decoding token by token through a float32, int8 and bfloat16 key/value cache: tokens a second, launches a token, the prompt pass |
| `train-speech` | the VITS plug-in's adversarial training step at one and several clips: s a step, launches a step by operation, peak memory, times real time |
| `memory` | stress: the pool over thousands of varied shapes (what stays cached, what is held over the start), long-sequence attention, the largest convolution-network batch the device takes |

Settings, all environment variables: `IDRAK_DEVICES` (`cpu`, `cuda:0`, `vulkan:0`, several with commas),
`IDRAK_BENCH_SIZE` (`small`, `medium`, `large`; default small on the CPU, medium on a GPU), `IDRAK_BENCH_REPEATS` (timed
rounds after the cold one, default 5), `IDRAK_BENCH_SECONDS` (the most a stress section runs, default 60),
`IDRAK_BENCH_OUT` (the rows as JSON), `IDRAK_BENCH_MODEL` (a VITS model folder for `train-speech`; default the tiny
fixture under `tests/`), `IDRAK_BENCH_CLIPS` (a folder of clips with transcripts; default synthetic clips).

Every section reports the cold round apart from the warm median (rule 76), counts launches with the library's telemetry
(rule 78) and reads peak memory from `ComputeResources` (rule 77). The kernel-level benches that reach into the backends'
internals stay in the test runner (`tests/Idrak.Tests -- --bench-gemm`, `--bench-gemv`, `--bench-conv`, ...).
