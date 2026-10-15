"""Verify pre_big_fuse_vmi.py keeps pre_big_fuse_asc.py's outer schedule verbatim.

Walks both module ASTs, replaces every ``with T.SimdVF():`` body with a
placeholder, and compares.  Only whitelisted differences (module docstring,
simd->vmi import, jit decorator, function-name suffix) are normalized away;
anything else must match exactly.
"""

import ast
import sys

ASC_PATH = 'tile_kernels/mhc/pre_big_fuse_asc.py'
VMI_PATH = 'tile_kernels/mhc/pre_big_fuse_vmi.py'


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


def replace_simdvf(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.For, ast.With, ast.If)) and hasattr(node, 'body'):
            node.body = [
                ast.Pass() if is_simdvf(child) else child
                for child in node.body
            ]
    return tree


def strip_simd_import(tree):
    kept = []
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == 'tilelang.ascend.language':
            continue
        kept.append(stmt)
    tree.body = kept
    return tree


def normalize(tree):
    tree = replace_simdvf(tree)
    # Drop the module docstring and the _u32_index/_row_* helper defs: the VMI
    # file keeps _u32_index but drops the ASC row-reduce helpers (they are
    # SimdVF-body machinery replaced by vcadd/vcmax groups).  Function bodies
    # of the kernels themselves are compared with SimdVF bodies stripped.
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        tree.body = tree.body[1:]
    tree.body = [
        stmt
        for stmt in tree.body
        if not (isinstance(stmt, ast.FunctionDef) and stmt.name in ('_row_reduce', '_row_max', '_u32_index'))
        and not (isinstance(stmt, ast.Assign) and getattr(stmt.targets[0], 'id', '') == '_VEC')
    ]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.name = node.name.removesuffix('_asc').removesuffix('_vmi')
            node.decorator_list = [ast.Name(id='JIT') for _ in node.decorator_list]
            # The kernel docstring prose differs (ASC vs VMI header); replace
            # it with a placeholder so only structure is compared.
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                node.body[0] = ast.Expr(value=ast.Constant(value='DOC'))
    return ast.dump(ast.Module(body=tree.body, type_ignores=()), include_attributes=False)


with open(ASC_PATH) as f:
    asc_tree = ast.parse(f.read())
with open(VMI_PATH) as f:
    vmi_tree = ast.parse(f.read())

asc_dump = normalize(strip_simd_import(asc_tree))
vmi_dump = normalize(strip_simd_import(vmi_tree))

if asc_dump == vmi_dump:
    print('OK: outer schedule identical (SimdVF bodies excluded, whitelisted diffs normalized)')
    sys.exit(0)

import difflib

asc_lines = asc_dump.splitlines()
vmi_lines = vmi_dump.splitlines()
print('MISMATCH outside SimdVF bodies:')
for line in difflib.unified_diff(asc_lines, vmi_lines, fromfile='asc', tofile='vmi', n=2):
    print(line[:400])
sys.exit(1)
