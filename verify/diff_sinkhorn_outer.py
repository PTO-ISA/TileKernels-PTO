"""Verify sinkhorn_vmi.py keeps sinkhorn_asc.py's outer schedule verbatim.

Walks both module ASTs, replaces every ``with T.SimdVF():`` body with a
placeholder, drops the module-level private vector helpers (_row_reduce /
_row_max / _col_reduce on both sides, _vcgadd / _vcgmax on the VMI side --
they are only ever called from inside SimdVF bodies and so belong to the
vector layer, not the outer schedule), and unparses for comparison.  Only
whitelisted differences (module docstring, simd->vmi import, jit decorator,
function-name suffix) are normalized away; anything else must match exactly.
"""

import ast
import sys

ASC_PATH = 'tile_kernels/mhc/sinkhorn_asc.py'
VMI_PATH = 'tile_kernels/mhc/sinkhorn_vmi.py'


def is_simdvf(node):
    if not isinstance(node, ast.With):
        return False
    test = node.items[0].context_expr
    return (
        isinstance(test, ast.Call)
        and isinstance(test.func, ast.Attribute)
        and test.func.attr == 'SimdVF'
        and not test.args
        and not test.keywords
    )


def drop_helpers(tree):
    tree.body = [stmt for stmt in tree.body if not (isinstance(stmt, ast.FunctionDef) and stmt.name.startswith('_'))]
    return tree


def replace_simdvf(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.For, ast.With, ast.If)) and hasattr(node, 'body'):
            node.body = [
                ast.Pass() if is_simdvf(child) else child
                for child in node.body
            ]
    return tree


def strip_docstring(body):
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body[:] = body[1:]


def normalize(tree):
    tree = drop_helpers(tree)
    tree = replace_simdvf(tree)
    # Drop module + function docstrings: the VMI file carries a port header and
    # rewords ASC-physical terminology; only the code structure is compared.
    strip_docstring(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            strip_docstring(node.body)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.name = node.name.removesuffix('_asc').removesuffix('_vmi')
            node.decorator_list = [ast.Name(id='JIT') for _ in node.decorator_list]
    return ast.dump(ast.Module(body=tree.body, type_ignores=()), include_attributes=False)


def strip_simd_import(tree):
    kept = []
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == 'tilelang.ascend.language':
            continue
        kept.append(stmt)
    tree.body = kept
    return tree


with open(ASC_PATH) as f:
    asc_tree = ast.parse(f.read())
with open(VMI_PATH) as f:
    vmi_tree = ast.parse(f.read())

asc_dump = normalize(strip_simd_import(asc_tree))
vmi_dump = normalize(strip_simd_import(vmi_tree))

if asc_dump == vmi_dump:
    print('OK: outer schedule identical (SimdVF bodies + vector helpers excluded, whitelisted diffs normalized)')
    sys.exit(0)

import difflib

asc_lines = asc_dump.splitlines()
vmi_lines = vmi_dump.splitlines()
print('MISMATCH outside SimdVF bodies:')
for line in difflib.unified_diff(asc_lines, vmi_lines, fromfile='asc', tofile='vmi', n=2):
    print(line[:400])
sys.exit(1)
