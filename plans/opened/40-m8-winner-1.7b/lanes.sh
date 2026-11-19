#!/bin/sh
# M7: each prompt profile's kernels table summed by lane (ms). Groups, by kernel name:
#   native  the native kind's kernel: native_mma_*
#   widen   the weight widened or packed to bfloat16 on every call: native_pack_w_*, super_widen_*, grid_widen_*
#   packx   x packed for the native kind: native_pack_x_*
#   parts   typed_sum_parts
#   rows    the tensor rows by a product's own lane: gemm_tc_nt_* without "multi" (over the rows as stored, the blocks or
#           super-blocks decoded as each tile loads, or "widened" rows)
#   groups  the tensor rows' group launches: gemm_tc_nt_* multi
#   head    the head's one-row kernel (one call)
#   other   everything else (norms, attention, activation, rope, copies)
#   products native + widen + packx + parts + rows + groups
# pass: the profile's pass line (ms, each layer drained). Usage: lanes.sh FILE...
for f in "$@"; do
  awk -v file="$(basename "$f" .txt | sed 's/^profile-//; s/-prompt$//')" '
    /^pass / { pass = $2 }
    /^Kernels \(/ { inside = 1; next }
    inside && /^exit/ { inside = 0 }
    inside && NF >= 4 && $(NF) == "%" {
      ms = $(NF - 2); calls = $(NF - 3); name = $1
      if (name ~ /^native_mma_/) native += ms
      else if (name ~ /^native_pack_w_|^super_widen_|^grid_widen_/) widen += ms
      else if (name ~ /^native_pack_x_/) packx += ms
      else if (name ~ /^typed_sum_parts/) parts += ms
      else if (name ~ /^gemm_tc_nt_/ && $0 ~ / multi /) groups += ms
      else if (name ~ /^gemm_tc_nt_/) rows += ms
      else if (calls == 1 && name ~ /gemv/) head += ms
      else other += ms
    }
    END {
      printf "%-18s pass %6.2f  native %6.3f  widen %5.3f  packx %5.3f  parts %5.3f  rows %6.3f  groups %6.3f  products %6.3f  head %5.3f  other %5.3f\n", file, pass, native, widen, packx, parts, rows, groups, native + widen + packx + parts + rows + groups, head, other
    }' "$f"
done
