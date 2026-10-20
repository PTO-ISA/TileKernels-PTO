#!/usr/bin/env python3
"""Minimal f32 -> e4m3 / bf16 saturation repro for ASC S.vcvt vs VMI V.vcvt.

Only one conversion is performed per kernel; no SwiGLU scale or mask is involved.
Run inside the A5 container after sourcing activate_vmi_migration.sh.
"""

import torch
import torch_npu

import tilelang
import tilelang.ascend.language as T

N = 64
VALUES = [
    float('inf'),
    float('-inf'),
    float('nan'),
    -0.0,
    448.0,
    448.00003,
    464.0,
    480.0,
    1e30,
]
VARIANTS = [
    ('asc_sat', False, True),
    ('asc_nosat', False, False),
    ('vmi_sat', True, True),
    ('vmi_nosat', True, False),
]
TARGETS = [
    ('e4m3', 'float8_e4m3fn', T.float8_e4m3fn),
    ('bf16', 'bfloat16', T.bfloat16),
]


def make_program(vmi: bool, sat: bool, target: str, target_tir):
    @T.prim_func
    def main(x: T.Buffer((N,), 'float32'), y: T.Buffer((N,), target)):
        with T.Kernel(1) as _:
            x_ub = T.alloc_shared((N,), 'float32')
            y_ub = T.alloc_shared((N,), target)
            T.copy(x, x_ub)
            with T.SimdVF():
                full = T.vmi.create_mask(N, size=N) if vmi else T.simd.pset(32)
                if vmi:
                    value = T.vmi.vload(x_ub[0], size=N)
                    converted = T.vmi.vcvt(
                        value,
                        target,
                        rounding='R',
                        saturate='SAT' if sat else 'NOSAT',
                    )
                    T.vmi.vstore(converted, y_ub[0], full)
                else:
                    value = T.simd.vld(x_ub[0])
                    converted = T.simd.vcvt(value, target_tir, sat=sat)
                    dist = 'PK4_B32' if target == 'float8_e4m3fn' else 'PK_B32'
                    T.simd.vsts(y_ub[0], converted, full, dist=dist)
            T.copy(y_ub, y)

    return main


def main():
    x_cpu = torch.tensor([VALUES[i % len(VALUES)] for i in range(N)], dtype=torch.float32)
    x = x_cpu.to('npu')
    for target, target_param, target_tir in TARGETS:
        results = {}
        for name, vmi, sat in VARIANTS:
            program = make_program(vmi, sat, target_param, target_tir)
            kernel = tilelang.compile(program, target='pto' if vmi else 'asc', out_idx=-1)
            y = kernel(x)
            torch.npu.synchronize()
            if target_param == 'bfloat16':
                bits = (y.cpu().view(torch.int16).to(torch.int32) & 0xFFFF).tolist()
            else:
                bits = y.cpu().view(torch.uint8).to(torch.int32).tolist()
            results[name] = bits[: len(VALUES)]
            source = kernel.get_kernel_source()
            path = f'/tmp/repro_{target}_saturate_{name}.txt'
            with open(path, 'w') as f:
                f.write(source)
            print(f'--- {target} {name} source: {path}')
            for line in source.splitlines():
                if any(tok in line for tok in ('vcvt', 'saturate', 'RS_ENABLE', 'RS_DISABLE')):
                    print('   ', line.strip())

        print(f'\n=== {target} bit patterns (hex) ===')
        print('value                 ' + ' '.join(f'{name:>10}' for name, _, _ in VARIANTS))
        for i, value in enumerate(VALUES):
            print(f'{value!r:<22}' + ' '.join(f'{results[name][i]:>#10x}' for name, _, _ in VARIANTS))
        print()


if __name__ == '__main__':
    main()
