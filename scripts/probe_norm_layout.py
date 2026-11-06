"""Host-side compile probe: which norm hidden sizes survive vpto layout search."""
import os
import sys

os.environ.setdefault("TILELANG_CACHE_DIR", "/home/jzhuang/work_dir/.tilelang_cache")

import vllm_kernels.pto.norm as norm

norm.get_num_vec_cores = lambda: 40


def try_hidden(hidden: int) -> None:
    try:
        norm.norm_tl(
            hidden,
            "bfloat16",
            "bfloat16",
            "float",
            False,
            1e-6,
            1.0,
            False,
            0,
            False,
            False,
            False,
            False,
            2,
            0,
            hidden,
            hidden,
            hidden,
            hidden,
            hidden,
            hidden,
        )
        print(f"OK hidden={hidden}", flush=True)
    except Exception as exc:
        path = f"/tmp/norm_err_{hidden}.txt"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(str(exc))
        print(f"ERR hidden={hidden} {type(exc).__name__} wrote {path}", flush=True)


def main() -> None:
    hiddens = [int(item) for item in sys.argv[1:]] or [512, 2048, 5120, 7168]
    print("PTO_FLAGS", os.environ.get("PTO_FLAGS"), flush=True)
    for hidden in hiddens:
        try_hidden(hidden)


if __name__ == "__main__":
    main()
