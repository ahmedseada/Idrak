#!/bin/sh
# M11: the figures derived from the saved files, for both models. Usage: derived.sh > derived.txt
#  1. each prompt profile: the pass, the sum of its kernels' times, their difference; and each kernel's calls a layer
#  2. the widening: the bytes a pass reads (the q4_k rows: 144 bytes per 256 values) and writes (bfloat16 rows; k = 2048,
#     3072, 6144, 8192 are whole blocks of 128, NativePadded, so no padding), over super_widen's time
#  3. the device's measured read rates: the bf16 head's one-row product in the decode profiles (the head's bf16 rows read
#     once: vocabulary x hidden x 2 bytes over bf16_gemv_r1_f32's time) and the kernel bench's largest bf16 gemv
#  4. memory: weights, in use after the runs, peak, from the benches and traces; the widened buffer at the largest product
#  5. host time a product call in the prompt windows: the sampled ms over the calls whose launches fall in the window
#     (10 timed passes x 28 layers x 7 products = 1,960; the warm-up pass's launches come before the window)
here=$(cd "$(dirname "$0")" && pwd)
kernels() {
  awk -v f="$(basename "$(dirname "$1")")/$(basename "$1")" '
    /^pass / { pass = $2 }
    /^Kernels \(/ { inside = 1; launches = $2; gsub(/\(/, "", launches); next }
    inside && /^exit/ { inside = 0 }
    /^layers\.[0-9]+ / { v = $3; if (lmin == "" || v < lmin) lmin = v; if (v > lmax) lmax = v }
    inside && NF >= 4 && $(NF) == "%" { sum += $(NF - 2); n[++k] = $1; c[k] = $(NF - 3) }
    END {
      printf "%-42s pass %6.2f ms, kernels %6.2f ms in %s launches, pass - kernels %+6.2f ms; a layer (drained) %s to %s ms\n", f, pass, sum, launches, pass - sum, lmin, lmax
      printf "   calls a layer:"; for (i = 1; i <= k; i++) if (c[i] >= 28) printf " %s %g;", n[i], c[i] / 28; printf "\n"
    }' "$1"
}
echo "# 1. prompt profiles: the pass against the sum of the kernels"
for m in qwen3-1.7b llama-3b; do for r in adad name; do kernels "$here/$m/profile-$r-prompt.txt"; done; done
echo
echo "# 2. the widening's bytes and rate (super_widen_k256x144_*, every prompt product, 28 layers)"
widen() {
  # MODEL VALUES_PER_LAYER PROFILE
  ms=$(awk '/^super_widen_/ { print $(NF - 2) }' "$3")
  awk -v m="$1" -v v="$2" -v ms="$ms" 'BEGIN {
    vals = v * 28; rd = vals * 144 / 256; wr = vals * 2
    printf "%-11s %.0f values a pass: %.1f MiB of q4_k rows read + %.1f MiB of bfloat16 rows written = %.1f MiB in %s ms: %.0f GB/s\n", m, vals, rd / 1048576, wr / 1048576, (rd + wr) / 1048576, ms, (rd + wr) / ms / 1e6 }'
}
widen qwen3-1.7b $((2048*2048*2 + 2048*1024*2 + 2048*6144*2 + 6144*2048)) "$here/qwen3-1.7b/profile-adad-prompt.txt"
widen llama-3b $((3072*3072*2 + 3072*1024*2 + 3072*8192*2 + 8192*3072)) "$here/llama-3b/profile-adad-prompt.txt"
echo
echo "# 3. the device's measured read rates"
head_rate() {
  ms=$(awk '/^bf16_gemv_r1_f32/ { print $(NF - 2) }' "$3")
  awk -v f="$3" -v b="$2" -v ms="$ms" -v m="$1" 'BEGIN { printf "%-11s bf16 head read once: %.1f MiB in %s ms: %.0f GB/s (%s)\n", m, b / 1048576, ms, b / ms / 1e6, f }'
}
head_rate qwen3-1.7b $((151936*2048*2)) D:/Projects/idrak/test-output/adad/m6-proof-1.7b/profile-cuda-bf16-adad-decode.txt
head_rate qwen3-1.7b $((151936*2048*2)) D:/Projects/idrak/test-output/adad/m6-proof-1.7b/profile-cuda-bf16-name-decode.txt
head_rate llama-3b $((128256*3072*2)) D:/Projects/idrak/test-output/adad/m9-llama3.2/3b/profile-cuda-bf16-adad-decode.txt
head_rate llama-3b $((128256*3072*2)) D:/Projects/idrak/test-output/adad/m9-llama3.2/3b/profile-cuda-bf16-name-decode.txt
echo "kernel bench (kernels-gemv.txt): $(grep 'gemv bfloat16 1x1024 -> 32000' "$here/kernels-gemv.txt" | sed 's/  */ /g')"
echo "no Idrak tool measures a device-to-device copy (bench --kernels offers matmul, gemv, attention)"
echo
echo "# 4. memory (MiB): weights, in use after the runs, peak; peak - weights"
for m in qwen3-1.7b llama-3b; do for f in bench-adad bench-name trace-adad trace-name; do
  awk -v f="$m/$f" '$1=="weights"&&$2=="memory"{w=$3} $1=="memory"&&$2=="in"{u=$4} $1=="memory"&&$2=="peak"{p=$3} END{printf "%-22s weights %6s  in use %6s  peak %6s  peak - weights %4d\n", f, w, u, p, p - w}' "$here/$m/$f.txt"
done; done
echo "widened buffer at the largest product (n x k x 2 bytes): qwen3-1.7b $((6144*2048*2)) bytes ($((6144*2048*2/1048576)) MiB), llama-3b $((8192*3072*2)) bytes ($((8192*3072*2/1048576)) MiB)"
echo
echo "# 5. host time a product call in the prompt windows (sampled ms / 1,960 calls)"
per() { awk -v ms="$2" -v w="$1" 'BEGIN { printf "  %-58s %6.1f ms  %5.1f us a call\n", w, ms, ms * 1000 / 1960 }'; }
for m in qwen3-1.7b llama-3b; do
  echo "## $m adad (trace-adad-focus-storedsuper.txt)"
  f=$here/$m/trace-adad-focus-storedsuper.txt
  per "CudaBackend.StoredSuper (all)" "$(awk '/^focus/ { print $3 }' "$f")"
  awk '/calls \(its callees/ { on = 1; next } on && /time inside it/ { on = 0 } on && $1 ~ /^[0-9.]+$/ { print $1, $3 }' "$f" | while read ms name; do per "  of it $name" "$ms"; done
  echo "## $m name (trace-name-focus-matmul.txt)"
  per "Tensor.MatMul (all matching frames)" "$(awk '/^focus/ { print $3 }' "$here/$m/trace-name-focus-matmul.txt")"
done
