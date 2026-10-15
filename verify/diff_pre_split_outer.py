"""Verify pre_split_mixes_vmi.py keeps pre_split_mixes_asc.py's outer schedule verbatim.

Walks both module ASTs, replaces every ``with T.SimdVF():`` body (and the
``_u32_index`` helper body, which is the SimdVF expression layer) with a
placeholder, and unparses for comparison.  Only whitelisted differences
(module docstring, simd->vmi import, jit decorator, function-name suffix) are
normalized away; anything else must match exactly.

The ASC module also carries two SimtVF-only generic fallback kernels
(``mhc_pre_split_mixes_{fwd,bwd}_asc``) that the dispatch layer never calls;
they have no SimdVF body and are not ported, so functions without a
normalized-name counterpart are ignored on the ASC side.
"""

import ast
import sys

ASC_PATH = 'tile_kernels/mhc/pre_split_mixes_asc.py'
VMI_PATH = 'tile_kernels/mhc/pre_split_mixes_vmi.py'


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


def replace_helper(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == '_u32_index':
            node.body = [ast.Pass()]
    return tree


def normalize(tree):
    tree = replace_simdvf(tree)
    tree = replace_helper(tree)
    # Drop the module docstring: the VMI file carries a port header.
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        tree.body = tree.body[1:]
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


def keep_shared_functions(tree, keep):
    tree.body = [
        stmt
        for stmt in tree.body
        if not (isinstance(stmt, ast.FunctionDef) and stmt.name.removesuffix('_asc').removesuffix('_vmi') not in keep)
    ]


ported = {node.name.removesuffix('_vmi') for node in vmi_tree.body if isinstance(node, ast.FunctionDef)}
keep_shared_functions(asc_tree, ported)

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
