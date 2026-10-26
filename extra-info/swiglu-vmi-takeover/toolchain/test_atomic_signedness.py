"""Compile regression for signed/unsigned atomic operands and returned values."""
from ptodsl import pto

@pto.simt
def probe(a: pto.ptr(pto.si32, 'gm'), b: pto.ptr(pto.si64, 'gm'),
          c: pto.ptr(pto.ui32, 'gm'), d: pto.ptr(pto.ui64, 'gm')):
    # Store the returned value through the authored pointer: verifies that the
    # internal signless normalization does not change the Python API result type.
    pto.stg(pto.atomic_add(a, 1), a, 0)
    pto.stg(pto.atomic_add(b, 1), b, 0)
    pto.stg(pto.atomic_add(c, 1), c, 0)
    pto.stg(pto.atomic_add(d, 1), d, 0)
    pto.stg(pto.atomic_min(a, -1, signedness='signed'), a, 0)
    pto.stg(pto.atomic_max(c, 1, signedness='unsigned'), c, 0)

@pto.jit(target='a5')
def entry(a: pto.ptr(pto.si32, 'gm'), b: pto.ptr(pto.si64, 'gm'),
          c: pto.ptr(pto.ui32, 'gm'), d: pto.ptr(pto.ui64, 'gm')):
    pto.simt_launch(probe, a, b, c, d, dims=(32, 1, 1))

if __name__ == '__main__':
    text = entry.compile().mlir_text()
    assert text.count('pto.atomic_add') == 4
    assert 'pto.atomic_min' in text and 'pto.atomic_max' in text
    print(text)
    print('ATOMIC_SIGNEDNESS_COMPILE_PASS')
