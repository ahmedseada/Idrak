#!/bin/sh
# M10: the side speed measurement, from MODEL/results.txt (the "bench" warm lines) and the prompt profiles: lane 3 forced
# (LABEL TYPE-lane3) against the same session's native control (TYPE-ctl). The profiles' kernels summed by group: x
# quantized into q8_0 blocks (blocks_quantize_*), the block products (matmul_blocks_*), and the control's products (the
# native kind native_mma_*, the weight's conversion native_pack_w_* / super_widen_* / grid_widen_*, x packing
# native_pack_x_*, typed_sum_parts, the tensor rows gemm_tc_nt_*), the head (one call), the rest. The block products'
# rate: the layers' prompt products' work (2 m n k summed, m = 512) over their ms.
# Usage: speed.sh MODEL LAYERS HIDDEN Q KV INTER
m=$1; L=$2; h=$3; q=$4; kv=$5; f=$6
out=$(dirname "$0")/$m
work=$(awk -v L="$L" -v h="$h" -v q="$q" -v kv="$kv" -v f="$f" 'BEGIN { printf "%.0f", L * 2 * 512 * (h * q + h * kv * 2 + q * h + h * f * 2 + f * h) }')
echo "## $m: the layers' prompt products, $(awk -v w="$work" 'BEGIN { printf "%.2f", w / 1e9 }') GFLOP a 512-token pass"
printf "%-12s %13s %13s %8s | %9s %9s %9s %9s %9s %9s %9s %8s %9s\n" "profile" "bench prompt" "control" "ratio" "pass" "quantize" "blocks" "native" "convert" "x pack" "sums" "rows" "TFLOP/s"
for w in q4_0 int8; do
  for l in "$w-ctl" "$w-lane3"; do
    b=$(awk -v l="$l" '$1 == "bench" && $2 == l && $3 == "warm" { print $5 }' "$out/results.txt")
    c=$(awk -v l="$w-ctl" '$1 == "bench" && $2 == l && $3 == "warm" { print $5 }' "$out/results.txt")
    awk -v l="$l" -v b="$b" -v c="$c" -v work="$work" '
      /^pass / { pass = $2 }
      /^Kernels \(/ { inside = 1; next }
      inside && /^exit/ { inside = 0 }
      inside && NF >= 4 && $(NF) == "%" {
        ms = $(NF - 2); name = $1
        if (name ~ /^blocks_quantize_/) qz += ms
        else if (name ~ /^matmul_blocks_/) bl += ms
        else if (name ~ /^native_mma_/) nat += ms
        else if (name ~ /^native_pack_w_|^super_widen_|^grid_widen_/) cv += ms
        else if (name ~ /^native_pack_x_/) px += ms
        else if (name ~ /^typed_sum_parts/) sp += ms
        else if (name ~ /^gemm_tc_nt_/) rw += ms
      }
      END {
        printf "%-12s %13s %13s %8.3f | %9.2f %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %8.3f %9s\n", l, b, c, b / c, pass, qz, bl, nat, cv, px, sp, rw, (bl > 0 ? sprintf("%.1f", work / bl / 1e9) : "-")
      }' "$out/profile-$l-prompt.txt"
  done
done
