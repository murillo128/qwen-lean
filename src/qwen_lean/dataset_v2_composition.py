from __future__ import annotations

import hashlib
import json
import pickle
import random
import re
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset_v2_contract import (
    derivation_family_fingerprint,
    proof_fingerprint,
    proof_variant_id,
    statement_fingerprint_v2,
    statement_id,
)
from .dataset_v2_schema import (
    DATASET_V2_SCHEMA_VERSION,
    DatasetV2Record,
    EnvironmentContext,
    LengthMetadata,
    ProofVariant,
    ProofVerification,
)


AUDIT_PREFIX = "DATASET_V2_AUDIT\t"
MISSING_PREFIX = "DATASET_V2_MISSING\t"
UNIVERSE_GROUNDING_PREFIX = "DATASET_V2_UNIVERSE_GROUNDING\t"
STRUCTURAL_ARITY = {"direct": 2, "branching": 3, "deep": 4}
CONSTANT_PRESENCE_BATCH_SIZE = 256
PERSISTED_CONTEXT_CACHE_VERSION = "dataset-v2-persisted-context-v2"
_QUOTED_NAME_COMPONENT_RE = re.compile(r"«([^»]+)»")


def lean_name_key(value: str) -> str:
    """Match Lean syntax names with `Name.toString` audit output."""

    return _QUOTED_NAME_COMPONENT_RE.sub(r"\1", value)


@dataclass(frozen=True)
class CompositionSource:
    statement_id: str
    declaration_name: str
    source_module: str
    topic_tags: tuple[str, ...]
    domain_family: str
    canonical_declaration: str = ""
    resolved_dependencies: tuple[str, ...] = ()
    type_head: str = "other"
    universe_arguments: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompositionPlan:
    synthetic_name: str
    source_lemmas: tuple[CompositionSource, ...]
    domain_family: str
    structural_class: str
    generator_family: str
    normalized_proof_dag: str
    derivation_family_fingerprint: str
    relation_edges: tuple[tuple[str, str, str], ...] = ()
    retrieval_lemmas: tuple[CompositionSource, ...] = ()
    retrieval_index: tuple[tuple[str, str], ...] = ()

    def validate(self) -> None:
        if self.structural_class not in STRUCTURAL_ARITY:
            raise ValueError(f"unknown structural class: {self.structural_class}")
        expected = STRUCTURAL_ARITY[self.structural_class]
        names = {item.declaration_name for item in self.source_lemmas}
        if len(self.source_lemmas) != expected or len(names) != expected:
            raise ValueError(
                f"{self.structural_class} composition requires {expected} distinct source lemmas"
            )
        if self.structural_class == "deep" and self.normalized_proof_dag.count("and(") < 3:
            if "path>=3" not in self.normalized_proof_dag:
                raise ValueError("deep composition must have dependency path length at least three")
        connected = {self.source_lemmas[0].declaration_name}
        pending = list(self.relation_edges)
        while pending:
            progress = False
            for edge in list(pending):
                left, right, _ = edge
                if left in connected or right in connected:
                    connected.update((left, right))
                    pending.remove(edge)
                    progress = True
            if not progress:
                break
        if connected != names:
            raise ValueError("composition sources are not connected in the dependency/relevance graph")
        retrieval_names = {item.declaration_name for item in self.retrieval_lemmas}
        if names & retrieval_names:
            raise ValueError("shortcut retrieval repeats a planned source lemma")
        if {name for name, _ in self.retrieval_index} != retrieval_names:
            raise ValueError("shortcut retrieval index does not match retrieved lemmas")


@dataclass(frozen=True)
class CompositionAudit:
    name: str
    statement_type: str
    actual_dependencies: tuple[str, ...]
    level_parameters: tuple[str, ...]


@dataclass(frozen=True)
class LeanCompositionRun:
    status: str
    exit_code: int | None
    latency_seconds: float
    diagnostic: str
    audits: tuple[CompositionAudit, ...]


@dataclass(frozen=True)
class ShortcutGateRun:
    status: str
    exit_code: int | None
    latency_seconds: float
    rejected_names: tuple[str, ...]
    diagnostic: str


def _dag(structural_class: str, labels: Sequence[str], *, final_only: bool) -> str:
    if structural_class == "direct":
        if not final_only:
            return f"and(leaf:{labels[0]},leaf:{labels[1]});path=2"
        return f"iff(forward:leaf:{labels[1]},backward:leaf:{labels[0]});path=2"
    if structural_class == "branching":
        if not final_only:
            return (
                f"and(leaf:{labels[0]},iff(forward:leaf:{labels[2]},"
                f"backward:leaf:{labels[1]}));path=3"
            )
        return (
            f"iff(forward:and(leaf:{labels[1]},leaf:{labels[2]}),"
            f"backward:leaf:{labels[0]});path=3"
        )
    if final_only:
        return (
            f"iff(forward:and(and(leaf:{labels[1]},leaf:{labels[2]}),"
            f"leaf:{labels[3]}),backward:leaf:{labels[0]});path>=3"
        )
    return (
        f"and(iff(forward:leaf:{labels[1]},backward:leaf:{labels[0]}),"
        f"iff(forward:leaf:{labels[3]},backward:leaf:{labels[2]}));path>=3"
    )


def _relation(
    left: CompositionSource, right: CompositionSource
) -> str | None:
    if (
        right.declaration_name in left.resolved_dependencies
        or left.declaration_name in right.resolved_dependencies
    ):
        return "direct-dependency"
    shared_dependencies = set(left.resolved_dependencies).intersection(
        right.resolved_dependencies
    )
    if shared_dependencies:
        return "shared-dependency-neighborhood"
    if left.source_module == right.source_module:
        return "same-source-module"
    shared_relevance = {
        tag
        for tag in set(left.topic_tags).intersection(right.topic_tags)
        if tag.startswith(("prime-family:", "riemann-relevance:", "riemann-seed:"))
    }
    if shared_relevance:
        return "shared-relevance-neighborhood"
    return None


def _graph_indexes(
    pool: Sequence[CompositionSource],
) -> tuple[
    dict[str, CompositionSource],
    dict[str, list[CompositionSource]],
    dict[str, list[CompositionSource]],
    dict[str, list[CompositionSource]],
    dict[str, list[CompositionSource]],
]:
    by_name = {item.declaration_name: item for item in pool}
    reverse_dependencies: dict[str, list[CompositionSource]] = defaultdict(list)
    type_heads: dict[str, list[CompositionSource]] = defaultdict(list)
    modules: dict[str, list[CompositionSource]] = defaultdict(list)
    relevance: dict[str, list[CompositionSource]] = defaultdict(list)
    for source in pool:
        type_heads[source.type_head].append(source)
        modules[source.source_module].append(source)
        for tag in source.topic_tags:
            if tag.startswith(
                ("prime-family:", "riemann-relevance:", "riemann-seed:")
            ):
                relevance[tag].append(source)
        for dependency in source.resolved_dependencies:
            reverse_dependencies[dependency].append(source)
    for values in (
        *reverse_dependencies.values(),
        *type_heads.values(),
        *modules.values(),
        *relevance.values(),
    ):
        values.sort(key=lambda item: item.declaration_name)
    return by_name, reverse_dependencies, type_heads, modules, relevance


def _neighbors(
    source: CompositionSource,
    by_name: Mapping[str, CompositionSource],
    reverse_dependencies: Mapping[str, Sequence[CompositionSource]],
    modules: Mapping[str, Sequence[CompositionSource]],
    relevance: Mapping[str, Sequence[CompositionSource]],
) -> list[tuple[CompositionSource, str]]:
    related: dict[str, tuple[CompositionSource, str]] = {}
    for dependency in source.resolved_dependencies:
        dependency_source = by_name.get(dependency)
        if dependency_source is not None:
            related[dependency] = (dependency_source, "direct-dependency")
        users = reverse_dependencies.get(dependency, ())
        if len(users) <= 64:
            for user in users:
                if user.declaration_name != source.declaration_name:
                    related.setdefault(
                        user.declaration_name,
                        (user, "shared-dependency-neighborhood"),
                    )
    candidates = list(modules.get(source.source_module, ()))
    for tag in source.topic_tags:
        candidates.extend(relevance.get(tag, ()))
    for candidate in candidates:
        if candidate.declaration_name == source.declaration_name:
            continue
        relation = _relation(source, candidate)
        if relation is not None:
            related.setdefault(candidate.declaration_name, (candidate, relation))
    return [related[name] for name in sorted(related)]


def build_composition_plans(
    source_pools: Mapping[str, Sequence[CompositionSource]],
    counts: Mapping[str, int],
    *,
    seed: str,
    name_prefix: str = "dataset_v2_synthetic",
) -> list[CompositionPlan]:
    plans: list[CompositionPlan] = []
    seen_sets: set[tuple[str, ...]] = set()
    global_index = 0
    for domain_family, requested in sorted(counts.items()):
        pool = tuple(source_pools.get(domain_family, ()))
        if len(pool) < max(STRUCTURAL_ARITY.values()):
            raise ValueError(
                f"domain family {domain_family} has {len(pool)} sources; at least four are required"
            )
        by_name, reverse_dependencies, type_heads, modules, relevance = _graph_indexes(
            pool
        )
        neighbor_cache: dict[str, list[tuple[CompositionSource, str]]] = {}

        def related(source: CompositionSource) -> list[tuple[CompositionSource, str]]:
            if source.declaration_name not in neighbor_cache:
                neighbor_cache[source.declaration_name] = _neighbors(
                    source,
                    by_name,
                    reverse_dependencies,
                    modules,
                    relevance,
                )
            return neighbor_cache[source.declaration_name]

        produced = 0
        attempts = 0
        while produced < requested:
            attempts += 1
            if attempts > max(2000, requested * 200):
                raise ValueError(f"could not produce {requested} unique plans for {domain_family}")
            structural = ("direct", "branching", "deep")[global_index % 3]
            arity = STRUCTURAL_ARITY[structural]
            rng = random.Random(
                int.from_bytes(
                    hashlib.sha256(
                        f"{seed}\0{domain_family}\0{attempts}\0{global_index}".encode()
                    ).digest(),
                    "big",
                )
            )
            selected_list = [pool[rng.randrange(len(pool))]]
            relation_edges: list[tuple[str, str, str]] = []
            while len(selected_list) < arity:
                selected_names = {item.declaration_name for item in selected_list}
                frontier = [
                    (source, candidate, relation)
                    for source in selected_list
                    for candidate, relation in related(source)
                    if candidate.declaration_name not in selected_names
                ]
                if not frontier:
                    break
                source, candidate, relation = frontier[rng.randrange(len(frontier))]
                selected_list.append(candidate)
                relation_edges.append(
                    (source.declaration_name, candidate.declaration_name, relation)
                )
            if len(selected_list) != arity:
                continue
            selected = tuple(selected_list)
            set_identity = tuple(sorted(item.statement_id for item in selected))
            if set_identity in seen_sets:
                continue
            seen_sets.add(set_identity)
            final_only = global_index % 10 == 0
            generator = (
                f"final-only:{structural}-graph-iff-v2"
                if final_only
                else f"{structural}-graph-logic-v2"
            )
            source_ids = tuple(item.statement_id for item in selected)
            dag = _dag(structural, source_ids, final_only=final_only)
            family = derivation_family_fingerprint(
                source_ids,
                normalized_proof_dag=dag,
                generator_family=generator,
            )
            retrieval_pool: dict[str, tuple[CompositionSource, set[str]]] = {}
            selected_names = {item.declaration_name for item in selected}
            for source in selected:
                for candidate, relation in related(source):
                    if candidate.declaration_name in selected_names:
                        continue
                    _, origins = retrieval_pool.setdefault(
                        candidate.declaration_name, (candidate, set())
                    )
                    origins.add(f"dependency-relevance-neighborhood:{relation}")
            target_head = "iff" if final_only else "and"
            for candidate in type_heads.get(target_head, ()):
                if candidate.declaration_name not in selected_names:
                    _, origins = retrieval_pool.setdefault(
                        candidate.declaration_name, (candidate, set())
                    )
                    origins.add(f"type-head:{target_head}")
            retrieval_entries = tuple(
                sorted(
                    retrieval_pool.values(),
                    key=lambda item: hashlib.sha256(
                        f"{seed}\0{global_index}\0retrieval\0{item[0].declaration_name}".encode()
                    ).hexdigest(),
                )[:4]
            )
            retrieval = tuple(item for item, _ in retrieval_entries)
            retrieval_index = tuple(
                (item.declaration_name, ",".join(sorted(origins)))
                for item, origins in retrieval_entries
            )
            plan = CompositionPlan(
                synthetic_name=f"{name_prefix}_{global_index:06d}",
                source_lemmas=selected,
                domain_family=domain_family,
                structural_class=structural,
                generator_family=generator,
                normalized_proof_dag=dag,
                derivation_family_fingerprint=family,
                relation_edges=tuple(relation_edges),
                retrieval_lemmas=retrieval,
                retrieval_index=retrieval_index,
            )
            plan.validate()
            plans.append(plan)
            produced += 1
            global_index += 1
    return plans


def _type_expression(plan: CompositionPlan) -> str:
    types = [
        f"(source_type% {item.declaration_name})"
        for item in plan.source_lemmas
    ]
    final_only = plan.generator_family.startswith("final-only:")
    if plan.structural_class == "direct" and not final_only:
        return f"{types[0]} ∧ {types[1]}"
    if plan.structural_class == "direct":
        return f"{types[0]} ↔ {types[1]}"
    if plan.structural_class == "branching" and not final_only:
        return f"{types[0]} ∧ ({types[1]} ↔ {types[2]})"
    if plan.structural_class == "branching":
        return f"{types[0]} ↔ ({types[1]} ∧ {types[2]})"
    if final_only:
        return f"{types[0]} ↔ (({types[1]} ∧ {types[2]}) ∧ {types[3]})"
    return f"({types[0]} ↔ {types[1]}) ∧ ({types[2]} ↔ {types[3]})"


def _oracle_expression(plan: CompositionPlan) -> str:
    names = []
    for item in plan.source_lemmas:
        suffix = (
            ".{" + ", ".join(item.universe_arguments) + "}"
            if item.universe_arguments
            else ""
        )
        names.append(f"@{item.declaration_name}{suffix}")
    final_only = plan.generator_family.startswith("final-only:")
    if plan.structural_class == "direct" and not final_only:
        return f"⟨{names[0]}, {names[1]}⟩"
    if plan.structural_class == "direct":
        return f"⟨fun _ => {names[1]}, fun _ => {names[0]}⟩"
    if plan.structural_class == "branching" and not final_only:
        return (
            f"⟨{names[0]}, ⟨fun _ => {names[2]}, fun _ => {names[1]}⟩⟩"
        )
    if plan.structural_class == "branching":
        return f"⟨fun _ => ⟨{names[1]}, {names[2]}⟩, fun _ => {names[0]}⟩"
    if final_only:
        return (
            f"⟨fun _ => ⟨⟨{names[1]}, {names[2]}⟩, {names[3]}⟩, "
            f"fun _ => {names[0]}⟩"
        )
    return (
        f"⟨⟨fun _ => {names[1]}, fun _ => {names[0]}⟩, "
        f"⟨fun _ => {names[3]}, fun _ => {names[2]}⟩⟩"
    )


def _composition_imports(
    plans: Sequence[CompositionPlan], *, include_retrieval: bool = False
) -> tuple[str, ...]:
    sources = [
        source
        for plan in plans
        for source in (
            plan.source_lemmas
            + (plan.retrieval_lemmas if include_retrieval else ())
        )
    ]
    pnt_flags = {
        source.source_module.startswith("PrimeNumberTheoremAnd")
        for source in sources
    }
    if len(pnt_flags) != 1:
        raise ValueError("one generated module cannot mix Mathlib and PNT+ source constants")
    if pnt_flags == {True}:
        return ("PrimeNumberTheoremAnd",)
    return tuple(
        sorted({source.source_module for source in sources})
    )


def composition_imports(
    plans: Sequence[CompositionPlan], *, include_retrieval: bool = False
) -> tuple[str, ...]:
    """Return the deterministic import context used to verify composition plans."""

    return _composition_imports(plans, include_retrieval=include_retrieval)


_LEAN_AUDIT_PRELUDE = r'''
open Lean Elab Term Command Meta

syntax "source_type% " ident : term

elab_rules : term
  | `(source_type% $id:ident) => do
      let info ← getConstInfo id.getId
      let levels := info.levelParams.map fun _ => Level.zero
      let type ← inferType (mkConst info.name levels)
      unless ← isProp type do
        throwError "Dataset-v2 composition source is not proposition-valued"
      pure type

partial def datasetV2Consts : Expr → List Name
  | .bvar _ | .fvar _ | .mvar _ | .sort _ | .lit _ => []
  | .const name _ => [name]
  | .app fn arg => datasetV2Consts fn ++ datasetV2Consts arg
  | .lam _ type body _ => datasetV2Consts type ++ datasetV2Consts body
  | .forallE _ type body _ => datasetV2Consts type ++ datasetV2Consts body
  | .letE _ type value body _ =>
      datasetV2Consts type ++ datasetV2Consts value ++ datasetV2Consts body
  | .mdata _ body => datasetV2Consts body
  | .proj _ _ body => datasetV2Consts body

def datasetV2Audit (names : Array Name) : CommandElabM Unit :=
  liftTermElabM <| do
    for name in names do
      let info ← getConstInfo name
      match info with
      | .thmInfo data =>
        let rendered := (← ppExpr data.type).pretty
        let dependencies := (datasetV2Consts data.value).eraseDups
        let json := Json.mkObj [
          ("name", Json.str name.toString),
          ("type", Json.str rendered),
          ("dependencies", toJson (dependencies.map Name.toString)),
          ("level_parameters", toJson (data.levelParams.map Name.toString))
        ]
        logInfo ("DATASET_V2_AUDIT\t" ++ json.compress)
      | _ => throwError "{name} is not a theorem"
'''.strip()


def render_composition_source(plans: Sequence[CompositionPlan]) -> str:
    if not plans:
        raise ValueError("composition source requires at least one plan")
    lines = [*(f"import {module}" for module in _composition_imports(plans)), "", _LEAN_AUDIT_PRELUDE, ""]
    for plan in plans:
        plan.validate()
        lines.extend(
            [
                f"theorem {plan.synthetic_name} : {_type_expression(plan)} := by",
                f"  exact {_oracle_expression(plan)}",
                "",
            ]
        )
    names = ", ".join(f"`{plan.synthetic_name}" for plan in plans)
    lines.append(f"run_cmd datasetV2Audit #[{names}]")
    return "\n".join(lines) + "\n"


def render_constant_audit_source(names: Sequence[str]) -> str:
    if not names:
        raise ValueError("constant audit requires at least one theorem name")
    rendered = ", ".join(f"`{name}" for name in names)
    return (
        "import PrimeNumberTheoremAnd\n\n"
        + _LEAN_AUDIT_PRELUDE
        + f"\n\nrun_cmd datasetV2Audit #[{rendered}]\n"
    )


def render_constant_presence_source(sources: Sequence[CompositionSource]) -> str:
    if not sources:
        raise ValueError("constant presence audit requires at least one source")
    pnt = {source.source_module.startswith("PrimeNumberTheoremAnd") for source in sources}
    if len(pnt) != 1:
        raise ValueError("constant presence audit cannot mix Mathlib and PNT+ sources")
    modules = (
        ("PrimeNumberTheoremAnd",)
        if pnt == {True}
        else tuple(sorted({source.source_module for source in sources}))
    )
    names = [f"`{source.declaration_name}" for source in sources]
    imports = "\n".join(f"import {module}" for module in modules)
    commands = "\n".join(
        "run_cmd datasetV2Presence #["
        + ", ".join(names[start : start + CONSTANT_PRESENCE_BATCH_SIZE])
        + "]"
        for start in range(0, len(names), CONSTANT_PRESENCE_BATCH_SIZE)
    )
    return f'''{imports}

open Lean Elab Command

def datasetV2Presence (names : Array Name) : CommandElabM Unit := do
  let env ← getEnv
  for name in names do
    if !env.contains name then
      logInfo ("{MISSING_PREFIX}" ++ name.toString)
    else
      let info ← getConstInfo name
      let arguments := info.levelParams.map fun _ => "0"
      if !arguments.isEmpty then
        let value := Json.mkObj [
          ("name", Json.str name.toString),
          ("arguments", toJson arguments)
        ]
        logInfo ("{UNIVERSE_GROUNDING_PREFIX}" ++ value.compress)

{commands}
'''


def find_missing_constants(
    sources: Sequence[CompositionSource],
    *,
    source_path: Path,
    target_root: Path,
    timeout_seconds: float = 1800.0,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(render_constant_presence_source(sources), encoding="utf-8")
    completed = subprocess.run(
        ["lake", "env", "lean", "-E", "hasSorry", str(source_path.resolve())],
        cwd=target_root,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    output = completed.stdout + "\n" + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"constant presence audit failed: {output[-8000:]}")
    missing = tuple(
        sorted(
            line.split(MISSING_PREFIX, 1)[1].strip()
            for line in output.splitlines()
            if MISSING_PREFIX in line
        )
    )
    grounding: dict[str, tuple[str, ...]] = {}
    for line in output.splitlines():
        marker = line.find(UNIVERSE_GROUNDING_PREFIX)
        if marker == -1:
            continue
        value = json.loads(line[marker + len(UNIVERSE_GROUNDING_PREFIX) :])
        grounding[str(value["name"])] = tuple(str(item) for item in value["arguments"])
    return missing, grounding


def parse_audits(output: str) -> tuple[CompositionAudit, ...]:
    audits: list[CompositionAudit] = []
    for line in output.splitlines():
        marker = line.find(AUDIT_PREFIX)
        if marker == -1:
            continue
        value = json.loads(line[marker + len(AUDIT_PREFIX) :])
        audits.append(
            CompositionAudit(
                name=str(value["name"]),
                statement_type=str(value["type"]),
                actual_dependencies=tuple(sorted(str(item) for item in value["dependencies"])),
                level_parameters=tuple(str(item) for item in value["level_parameters"]),
            )
        )
    return tuple(audits)


def run_composition_source(
    source_path: Path,
    *,
    target_root: Path,
    timeout_seconds: float = 1800.0,
) -> LeanCompositionRun:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            ["lake", "env", "lean", "-E", "hasSorry", str(source_path.resolve())],
            cwd=target_root,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return LeanCompositionRun(
            "timeout", None, time.perf_counter() - started, str(error)[-4000:], ()
        )
    output = completed.stdout + "\n" + completed.stderr
    audits = parse_audits(output)
    status = "accepted" if completed.returncode == 0 else "rejected"
    return LeanCompositionRun(
        status=status,
        exit_code=completed.returncode,
        latency_seconds=time.perf_counter() - started,
        diagnostic="" if status == "accepted" else output[-8000:],
        audits=audits,
    )


def validate_composition_audits(
    plans: Sequence[CompositionPlan],
    audits: Sequence[CompositionAudit],
) -> dict[str, Any]:
    by_name = {audit.name: audit for audit in audits}
    if len(by_name) != len(plans):
        raise ValueError(f"composition audit count mismatch: {len(by_name)} != {len(plans)}")
    actual_counts: Counter[str] = Counter()
    for plan in plans:
        audit = by_name[plan.synthetic_name]
        planned = {
            lean_name_key(item.declaration_name): item.declaration_name
            for item in plan.source_lemmas
        }
        actual_keys = {lean_name_key(item) for item in audit.actual_dependencies}
        actual = {name for key, name in planned.items() if key in actual_keys}
        if len(actual) != len(planned):
            missing = sorted(
                name for key, name in planned.items() if key not in actual_keys
            )
            raise ValueError(f"composition {plan.synthetic_name} omitted source lemmas: {missing}")
        if plan.structural_class == "direct" and len(actual) < 2:
            raise ValueError("direct composition has fewer than two actual source lemmas")
        if plan.structural_class == "branching" and len(actual) < 3:
            raise ValueError("branching composition has fewer than three actual source lemmas")
        if plan.structural_class == "deep" and len(actual) < 4:
            raise ValueError("deep composition has fewer than four actual source lemmas")
        actual_counts[plan.structural_class] += 1
    return {
        "audited": len(plans),
        "actual_dependency_failures": 0,
        "structural_classes": dict(sorted(actual_counts.items())),
    }


def _compact_type(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def render_shortcut_gate_source(
    plans: Sequence[CompositionPlan],
    audits: Mapping[str, CompositionAudit],
) -> tuple[str, dict[int, str]]:
    """Render one-line gates; any obvious closure causes its example to fail."""

    lines = [
        *(
            f"import {module}"
            for module in _composition_imports(plans, include_retrieval=True)
        ),
        "",
        _LEAN_AUDIT_PRELUDE,
        "",
    ]
    line_to_name: dict[int, str] = {}
    for plan in plans:
        if plan.synthetic_name not in audits:
            raise ValueError(f"missing composition audit for {plan.synthetic_name}")
        target = _type_expression(plan)
        attempts = [
            "fail_if_success assumption",
            "fail_if_success rfl",
            "fail_if_success (solve | simp)",
        ]
        for source in plan.source_lemmas + plan.retrieval_lemmas:
            attempts.append(f"fail_if_success (exact {source.declaration_name})")
            attempts.append(f"fail_if_success (simpa using {source.declaration_name})")
        example_lines = [
            f"example : {target} := by",
            *(f"  {attempt}" for attempt in attempts),
            f"  exact {_oracle_expression(plan)}",
        ]
        for offset in range(len(example_lines)):
            line_to_name[len(lines) + offset + 1] = plan.synthetic_name
        lines.extend(example_lines)
    return "\n".join(lines) + "\n", line_to_name


def run_shortcut_gate_source(
    source_path: Path,
    *,
    target_root: Path,
    line_to_name: Mapping[int, str],
    timeout_seconds: float = 1800.0,
) -> ShortcutGateRun:
    """Compile the frozen shortcut suite and map any successful shortcut to a plan."""

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            ["lake", "env", "lean", "-E", "hasSorry", str(source_path.resolve())],
            cwd=target_root,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return ShortcutGateRun(
            "timeout", None, time.perf_counter() - started, (), str(error)[-8000:]
        )
    output = completed.stdout + "\n" + completed.stderr
    rejected: set[str] = set()
    path_pattern = re.escape(str(source_path.resolve()))
    for match in re.finditer(rf"{path_pattern}:(\d+):\d+: error:", output):
        name = line_to_name.get(int(match.group(1)))
        if name is not None:
            rejected.add(name)
    if completed.returncode != 0 and not rejected:
        status = "infrastructure-error"
    elif rejected:
        status = "rejected-shortcuts"
    else:
        status = "accepted"
    return ShortcutGateRun(
        status=status,
        exit_code=completed.returncode,
        latency_seconds=time.perf_counter() - started,
        rejected_names=tuple(sorted(rejected)),
        diagnostic="" if status == "accepted" else output[-8000:],
    )


def records_from_compositions(
    plans: Sequence[CompositionPlan],
    audits: Mapping[str, CompositionAudit],
    *,
    environment: Mapping[str, Any],
    verification_evidence_id: str,
    shortcut_status: Mapping[str, Sequence[str]],
    verified_imports: Mapping[str, Sequence[str]] | None = None,
) -> list[DatasetV2Record]:
    records: list[DatasetV2Record] = []
    for plan in plans:
        audit = audits[plan.synthetic_name]
        imports = tuple(
            (verified_imports or {}).get(
                plan.synthetic_name, composition_imports([plan])
            )
        )
        required_imports = set(composition_imports([plan]))
        if not required_imports <= set(imports):
            raise ValueError(
                f"verified imports omit source modules for {plan.synthetic_name}"
            )
        if plan.domain_family == "pnt-plus":
            if imports != ("PrimeNumberTheoremAnd",):
                raise ValueError("PNT+ composition must use its pinned umbrella import")
        elif "PrimeNumberTheoremAnd" in imports:
            raise ValueError("Mathlib composition cannot persist the PNT+ import context")
        declaration = f"theorem {plan.synthetic_name} : {audit.statement_type}"
        proof = f"by\n  exact {_oracle_expression(plan)}"
        identity = statement_id(declaration)
        actual_dependency_keys = {
            lean_name_key(item) for item in audit.actual_dependencies
        }
        actual = tuple(
            sorted(
                item.declaration_name
                for item in plan.source_lemmas
                if lean_name_key(item.declaration_name) in actual_dependency_keys
            )
        )
        variant = ProofVariant(
            proof_variant_id=proof_variant_id(identity, proof),
            source_expression=proof,
            canonical_proof=proof,
            completion=proof[2:].lstrip(),
            transformation_kind="none",
            proof_fingerprint=proof_fingerprint(proof),
            resolved_dependencies=actual,
            verification=ProofVerification(
                status="accepted",
                environment_id=str(environment["environment_id"]),
                method="generated-module-elaboration-and-shortcut-gate",
                evidence_id=verification_evidence_id,
            ),
            source_declaration_name=plan.synthetic_name,
            source_repository="generated:dataset-v2-composition-v1",
            source_revision=verification_evidence_id,
            source_file="DatasetV2Generated.lean",
        )
        combined = f"{declaration} := {proof}"
        record = DatasetV2Record(
            schema_version=DATASET_V2_SCHEMA_VERSION,
            statement_id=identity,
            canonical_declaration=declaration,
            normalized_statement_fingerprint=statement_fingerprint_v2(declaration),
            role="training",
            sampling_group_id=identity,
            provenance="synthetic",
            environment=EnvironmentContext(
                environment_id=str(environment["environment_id"]),
                lean_toolchain=str(environment["lean_toolchain"]),
                repository="generated:dataset-v2-composition-v1",
                revision=verification_evidence_id,
                mathlib_revision=str(environment["mathlib_revision"]),
                file_path="DatasetV2Generated.lean",
                module="DatasetV2Generated",
                imports=imports,
                source_span=None,
                context_kind="generated-module",
                target_compatibility="verified-target-environment",
            ),
            proof_variants=(variant,),
            topic_tags=(
                f"domain:{'prime-number-theory' if plan.domain_family != 'generic' else 'generic'}",
                f"prime-family:{plan.domain_family}"
                if plan.domain_family != "generic"
                else "composition:generic",
            ),
            memberships=(),
            length=LengthMetadata(
                declaration_chars=len(declaration),
                proof_chars=len(proof),
                completion_chars=len(proof) - 3,
                declaration_lines=declaration.count("\n") + 1,
                proof_lines=proof.count("\n") + 1,
                utf8_bytes=len(combined.encode("utf-8")),
            ),
            derivation_family_fingerprint=plan.derivation_family_fingerprint,
            generator_family=plan.generator_family,
            structural_class=plan.structural_class,
            normalized_proof_dag=plan.normalized_proof_dag,
            source_lemma_ids=tuple(item.statement_id for item in plan.source_lemmas),
            source_relation_edges=plan.relation_edges,
            shortcut_retrieval_ids=tuple(
                item.statement_id for item in plan.retrieval_lemmas
            ),
            shortcut_retrieval_index=plan.retrieval_index,
            shortcut_checks=tuple(shortcut_status.get(plan.synthetic_name, ())),
        )
        record.validate()
        records.append(record)
    return records


def render_persisted_synthetic_context_source(
    records: Sequence[DatasetV2Record],
    plans: Mapping[str, CompositionPlan],
) -> str:
    """Reconstruct final generated theorems from their plans and persisted imports."""

    if not records:
        raise ValueError("persisted synthetic context source requires records")
    imports = records[0].environment.imports
    if not imports:
        raise ValueError("synthetic EnvironmentContext has no imports")
    if any(record.provenance != "synthetic" for record in records):
        raise ValueError("persisted context verification accepts only synthetic records")
    if any(record.environment.imports != imports for record in records):
        raise ValueError("persisted context batch mixes distinct import sets")
    lines = [
        *(f"import {module}" for module in imports),
        "",
        _LEAN_AUDIT_PRELUDE,
        "",
    ]
    audit_names: list[str] = []
    for record in sorted(records, key=lambda item: item.statement_id):
        declaration_prefix = "theorem "
        if not record.canonical_declaration.startswith(declaration_prefix):
            raise ValueError("synthetic canonical declaration is not a theorem")
        name = record.canonical_declaration[len(declaration_prefix) :].split(
            ":", 1
        )[0].strip()
        plan = plans.get(name)
        if plan is None:
            raise ValueError(f"missing persisted-context plan for {name}")
        if (
            tuple(item.statement_id for item in plan.source_lemmas)
            != record.source_lemma_ids
        ):
            raise ValueError(f"persisted-context plan sources differ for {name}")
        namespace = f"R{record.statement_id}"
        lines.extend(
            [
                "namespace DatasetV2PersistedContext",
                f"namespace {namespace}",
            ]
        )
        for variant_index, variant in enumerate(record.proof_variants):
            qualified_name = (
                f"DatasetV2PersistedContext.{namespace}.V{variant_index}.{name}"
            )
            audit_names.append(qualified_name)
            lines.extend(
                [
                    f"namespace V{variant_index}",
                    f"theorem {name} : {_type_expression(plan)} := {variant.canonical_proof}",
                    f"end V{variant_index}",
                    "",
                ]
            )
        lines.extend([f"end {namespace}", "end DatasetV2PersistedContext", ""])
    rendered_names = ", ".join(f"`{name}" for name in audit_names)
    lines.append(f"run_cmd datasetV2Audit #[{rendered_names}]")
    return "\n".join(lines) + "\n"


def verify_persisted_synthetic_contexts(
    records: Sequence[DatasetV2Record],
    *,
    plans: Mapping[str, CompositionPlan],
    output_dir: Path,
    target_root: Path,
    workers: int = 4,
    timeout_seconds: float = 1800.0,
) -> dict[str, int | str]:
    """Lean-verify every final synthetic from only its persisted imports."""

    synthetic = [record for record in records if record.provenance == "synthetic"]
    if not synthetic:
        raise ValueError("persisted context verification found no synthetic records")
    grouped: dict[tuple[str, ...], list[DatasetV2Record]] = defaultdict(list)
    for record in synthetic:
        if not record.environment.imports:
            raise ValueError(
                f"synthetic record {record.statement_id} has no persisted imports"
            )
        grouped[record.environment.imports].append(record)

    lean_dir = output_dir / "lean"
    cache_dir = output_dir / ".synthetic-context-cache"
    lean_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    groups = sorted(grouped.items())

    def verify_group(
        indexed: tuple[int, tuple[tuple[str, ...], list[DatasetV2Record]]],
    ) -> tuple[int, int]:
        index, (_, group) = indexed
        source = render_persisted_synthetic_context_source(group, plans)
        record_contract = "\0".join(
            item.canonical_declaration
            for item in sorted(group, key=lambda item: item.statement_id)
        )
        digest = hashlib.sha256(
            f"{PERSISTED_CONTEXT_CACHE_VERSION}\0{source}\0{record_contract}".encode()
        ).hexdigest()
        source_path = lean_dir / f"PersistedContext-{index:04d}.lean"
        source_path.write_text(source, encoding="utf-8")
        cache_path = cache_dir / f"{digest}.pkl"
        if cache_path.is_file():
            with cache_path.open("rb") as handle:
                cache_version, status = pickle.load(handle)
            if (
                cache_version == PERSISTED_CONTEXT_CACHE_VERSION
                and status == "accepted"
            ):
                return len(group), sum(len(item.proof_variants) for item in group)
        run = run_composition_source(
            source_path,
            target_root=target_root,
            timeout_seconds=timeout_seconds,
        )
        if run.status != "accepted":
            raise RuntimeError(
                "persisted synthetic context verification failed for "
                f"group {index}: {run.status}: {run.diagnostic}"
            )
        audited = {audit.name: audit for audit in run.audits}
        expected_audits = sum(len(item.proof_variants) for item in group)
        if len(audited) != expected_audits:
            raise RuntimeError(
                f"persisted context group {index} audited {len(audited)} "
                f"of {expected_audits} proof variants"
            )
        for record in group:
            name = record.canonical_declaration.removeprefix("theorem ").split(
                ":", 1
            )[0].strip()
            stored_type = record.canonical_declaration.split(":", 1)[1]
            for variant_index in range(len(record.proof_variants)):
                qualified_name = (
                    "DatasetV2PersistedContext."
                    f"R{record.statement_id}.V{variant_index}.{name}"
                )
                audit = audited.get(qualified_name)
                if (
                    audit is None
                    or _compact_type(audit.statement_type)
                    != _compact_type(stored_type)
                ):
                    raise RuntimeError(
                        "persisted context reconstruction changed the stored statement "
                        f"for {record.statement_id} variant {variant_index}"
                    )
        temporary_cache = cache_path.with_suffix(".tmp")
        with temporary_cache.open("wb") as handle:
            pickle.dump(
                (PERSISTED_CONTEXT_CACHE_VERSION, run.status),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        temporary_cache.replace(cache_path)
        return len(group), sum(len(item.proof_variants) for item in group)

    with ThreadPoolExecutor(max_workers=min(workers, len(groups))) as executor:
        accepted = list(executor.map(verify_group, enumerate(groups)))
    accepted_records = sum(item[0] for item in accepted)
    accepted_variants = sum(item[1] for item in accepted)
    if accepted_records != len(synthetic):
        raise RuntimeError("persisted context verification did not cover every synthetic")
    return {
        "method": "lean-exact-persisted-import-context",
        "synthetic_records": len(synthetic),
        "proof_variants": sum(len(item.proof_variants) for item in synthetic),
        "context_groups": len(groups),
        "accepted_records": accepted_records,
        "accepted_proof_variants": accepted_variants,
        "context_failures": 0,
    }


def summarize_compositions(records: Sequence[DatasetV2Record]) -> dict[str, Any]:
    return {
        "unique_statements": len({item.statement_id for item in records}),
        "unique_source_lemma_sets": len(
            {tuple(sorted(item.source_lemma_ids)) for item in records}
        ),
        "unique_derivation_families": len(
            {item.derivation_family_fingerprint for item in records}
        ),
        "structural_classes": dict(
            sorted(Counter(str(item.structural_class) for item in records).items())
        ),
        "generator_families": dict(
            sorted(Counter(str(item.generator_family) for item in records).items())
        ),
        "domain_families": dict(
            sorted(
                Counter(
                    tag.removeprefix("prime-family:")
                    for item in records
                    for tag in item.topic_tags
                    if tag.startswith("prime-family:")
                ).items()
            )
        ),
        "roles": dict(sorted(Counter(item.role for item in records).items())),
    }
