"""Static dependency inventory for relocated specialist-product solvers."""
from __future__ import annotations

import ast
from pathlib import Path

from .catalog import list_products
from tis.physics.legacy_map import LEGACY_MODULE_TARGETS

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _module_path(module_name: str) -> Path | None:
    """Resolve a legacy alias, product solver, or shared solver to source."""
    if module_name.startswith("tokamak_analysis."):
        legacy_name = module_name.removeprefix("tokamak_analysis.")
        target = LEGACY_MODULE_TARGETS.get(legacy_name)
        if target is None:
            return None
    elif module_name.startswith("tis.products.") and ".physics." in module_name:
        target = module_name
    elif module_name.startswith("tis.physics.shared."):
        target = module_name
    else:
        return None

    relative = target.replace(".", "/")
    candidate = _REPOSITORY_ROOT / f"{relative}.py"
    if candidate.is_file():
        return candidate
    package = _REPOSITORY_ROOT / relative / "__init__.py"
    if package.is_file():
        return package
    if ".physics." in target and not target.startswith("tis.physics.shared."):
        shared = _REPOSITORY_ROOT / "tis/physics/shared" / f"{target.rsplit('.', 1)[-1]}.py"
        if shared.is_file():
            return shared
    return None


def _local_imports(module_name: str) -> tuple[set[str], set[str]]:
    """Return local physical dependencies and other TIS platform imports."""
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
                parts = current_package.split(".")
                ascend = max(node.level - 1, 0)
                if ascend:
                    parts = parts[:-ascend]
                base = ".".join(parts)
                target = ".".join(part for part in (base, node.module) if part)
                imported = (
                    [".".join(part for part in (target, alias.name) if part) for alias in node.names]
                    if node.module is None else [target]
                )
            else:
                imported = [node.module] if node.module else []
        else:
            continue
        for name in imported:
            if name.startswith("tokamak_analysis.") or name.startswith("tis.products.") or name.startswith("tis.physics.shared."):
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
    """Build direct/transitive dependency sets for all 20 products."""
    products = list_products()
    closures: dict[str, set[str]] = {}
    platform_dependencies: dict[str, set[str]] = {}
    direct_modules: dict[str, set[str]] = {}
    for product in products:
        slug = product["product_id"]
        roots = {entry["module"] for entry in product["entry_points"]}
        direct_modules[slug] = roots
        closure, platform = _closure(roots)
        closures[slug] = closure
        platform_dependencies[slug] = platform

    usage: dict[str, int] = {}
    for closure in closures.values():
        for module in closure:
            usage[module] = usage.get(module, 0) + 1
    shared_domain = sorted(module for module, count in usage.items() if count > 1)

    rows = []
    for product in products:
        slug = product["product_id"]
        closure = closures[slug]
        rows.append({
            "product_id": slug,
            "family_id": product["id"],
            "entry_point_modules": sorted(direct_modules[slug]),
            "transitive_domain_modules": sorted(closure),
            "product_local_domain_modules": sorted(module for module in closure if usage[module] == 1),
            "shared_domain_modules": sorted(module for module in closure if usage[module] > 1),
            "shared_platform_dependencies": sorted(platform_dependencies[slug]),
        })
    return {
        "analysis": "AST-based local import closure from product-owned and shared solver modules",
        "product_count": len(products),
        "shared_domain_modules": shared_domain,
        "products": rows,
        "limitations": [
            "Dynamic imports and runtime plugin loading are not visible to this static scan.",
            "Shared modules are identified by reuse across product dependency closures.",
            "Third-party packages and data-file dependencies are intentionally excluded.",
        ],
    }
