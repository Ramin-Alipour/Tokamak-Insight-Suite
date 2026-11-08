"""Static dependency inventory for specialist-product extraction planning.

The graph follows local Python imports from registered solver modules. It is an
engineering aid, not a declaration that every transitive dependency is owned by
one product; shared intersections are reported explicitly.
"""
from __future__ import annotations

import ast
from pathlib import Path

from .catalog import list_products
from tis.physics.legacy_map import LEGACY_MODULE_TARGETS

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_ANALYSIS_ROOT = _REPOSITORY_ROOT / "tokamak_analysis"


def _module_path(module_name: str) -> Path | None:
    if not module_name.startswith("tokamak_analysis."):
        return None
    relative = module_name.removeprefix("tokamak_analysis.")
    target = LEGACY_MODULE_TARGETS.get(relative)
    if target is None:
        return None
    target_path = target.replace(".", "/")
    module_file = _REPOSITORY_ROOT / f"{target_path}.py"
    package_file = _REPOSITORY_ROOT / target_path / "__init__.py"
    if module_file.is_file():
        return module_file
    if package_file.is_file():
        return package_file
    return None


def _local_imports(module_name: str) -> tuple[set[str], set[str]]:
    """Return local tokamak_analysis and shared tis imports from one module."""
    path = _module_path(module_name)
    if path is None:
        return set(), set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return set(), set()
    domain: set[str] = set()
    platform: set[str] = set()
    current_package = module_name.rpartition(".")[0]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = current_package.split(".")
                ascend = max(node.level - 1, 0)
                if ascend:
                    base_parts = base_parts[:-ascend]
                base = ".".join(base_parts)
                target = ".".join(part for part in (base, node.module) if part)
                if node.module is None:
                    imported = [
                        ".".join(part for part in (target, alias.name) if part)
                        for alias in node.names
                    ]
                else:
                    imported = [target]
            else:
                imported = [node.module] if node.module else []
        else:
            continue
        for name in imported:
            if name == "tokamak_analysis" or name.startswith("tokamak_analysis."):
                if name != module_name and _module_path(name) is not None:
                    domain.add(name)
            elif name == "tis" or name.startswith("tis."):
                platform.add(name)
    return domain, platform


def _closure(roots: set[str]) -> tuple[set[str], set[str]]:
    visited: set[str] = set()
    platform: set[str] = set()
    pending = list(roots)
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        domain_deps, platform_deps = _local_imports(module)
        platform.update(platform_deps)
        pending.extend(domain_deps - visited)
    return visited, platform


def build_dependency_inventory() -> dict:
    """Build direct/transitive solver dependency sets for all 20 products."""
    products = list_products()
    closures: dict[str, set[str]] = {}
    platform_dependencies: dict[str, set[str]] = {}
    direct_modules: dict[str, set[str]] = {}
    for product in products:
        slug = product["product_id"]
        roots = {f"tokamak_analysis.{entry['module']}" for entry in product["entry_points"]}
        direct_modules[slug] = roots
        closure, platform = _closure(roots)
        closures[slug] = closure
        platform_dependencies[slug] = platform

    usage: dict[str, int] = {}
    for closure in closures.values():
        for module in closure:
            usage[module] = usage.get(module, 0) + 1
    shared_domain = sorted(module for module, count in usage.items() if count > 1)

    product_rows = []
    for product in products:
        slug = product["product_id"]
        closure = closures[slug]
        product_rows.append({
            "product_id": slug,
            "family_id": product["id"],
            "entry_point_modules": sorted(direct_modules[slug]),
            "transitive_domain_modules": sorted(closure),
            "product_local_domain_modules": sorted(
                module for module in closure if usage[module] == 1
            ),
            "shared_domain_modules": sorted(
                module for module in closure if usage[module] > 1
            ),
            "shared_platform_dependencies": sorted(platform_dependencies[slug]),
        })
    return {
        "analysis": "AST-based local import closure from registered solver modules",
        "product_count": len(products),
        "shared_domain_modules": shared_domain,
        "products": product_rows,
        "limitations": [
            "Dynamic imports and runtime plugin loading are not visible to this static scan.",
            "A shared module is identified by reuse across product dependency closures, not by an ownership declaration.",
            "Third-party packages and data-file dependencies are intentionally excluded.",
            "The graph must be regenerated after module moves or registry changes.",
        ],
    }
