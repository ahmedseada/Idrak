#!/bin/sh
# M6: each CUDA prompt profile's kernels table summed by group (ms), to compare adad with name at Qwen3-1.7B and at the
# 0.6B check. Groups, by kernel name:
#   mma      the products' tensor-core kernels run by their own lane: native_mma_* (the native kind), gemm_tc_nn* (name's),
#            gemm_tc_nt_* without "multi" (adad's tensor rows, one product)
#   group    the group launches: gemm_tc_nt_* multi (adad's tensor rows) and gemm_tc_nn_*_multi (name's named kernels)
#   widen    the weight widened to bfloat16 on every call: native_pack_w_*, super_widen_*, grid_widen_*
#   packx    x packed for the native kind: native_pack_x_*
#   parts    typed_sum_parts (k split in parts, summed)
#   head     the head's one-row kernel (one call): *gemv* with 1 call
#   other    everything else (norms, attention, activation, rope, copies)
#   products mma + group + widen + packx + parts: every product of the layers, with its widening and packing
# pass: the profile's pass line (ms, each layer drained). Usage: kernel-groups.sh FILE... (prints one line a file)
for f in "$@"; do
  awk -v file="$(basename "$f")" '
    /^pass / { pass = $2 }
    /^Kernels \(/ { inside = 1; next }
    inside && /^exit/ { inside = 0 }
    inside && NF >= 4 && $(NF) == "%" {
      ms = $(NF - 2); calls = $(NF - 3); name = $1
      if (name ~ /^native_pack_w_|^super_widen_|^grid_widen_/) widen += ms
      else if (name ~ /^native_pack_x_/) packx += ms
      else if (name ~ /^typed_sum_parts/) parts += ms
      else if ($0 ~ / multi / || name ~ /_multi$/) group += ms
      else if (name ~ /^native_mma_|^gemm_tc_/) mma += ms
      else if (calls == 1 && name ~ /gemv/) head += ms
      else other += ms
      total += ms
    }
    END {
      printf "%-46s pass %6.2f  mma %6.3f  group %6.3f  widen %5.3f  packx %5.3f  parts %5.3f  head %5.3f  other %5.3f  products %6.3f  kernels %6.3f\n", file, pass, mma, group, widen, packx, parts, head, other, mma + group + widen + packx + parts, total
    }' "$f"
done
