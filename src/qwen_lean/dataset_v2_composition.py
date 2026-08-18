from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
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
STRUCTURAL_ARITY = {"direct": 2, "branching": 3, "deep": 4}


@dataclass(frozen=True)
class CompositionSource:
    statement_id: str
    declaration_name: str
    source_module: str
    topic_tags: tuple[str, ...]
    domain_family: str


@dataclass(frozen=True)
class CompositionPlan:
    synthetic_name: str
    source_lemmas: tuple[CompositionSource, ...]
    domain_family: str
    structural_class: str
    generator_family: str
    normalized_proof_dag: str
    derivation_family_fingerprint: str

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
            raise ValueError("deep composition must have dependency path length at least three")


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
        return f"and(leaf:{labels[0]},leaf:{labels[1]})"
    if structural_class == "branching":
        return f"and(and(leaf:{labels[0]},leaf:{labels[1]}),leaf:{labels[2]})"
    if final_only:
        return (
            f"and(and(leaf:{labels[0]},leaf:{labels[1]}),"
            f"and(leaf:{labels[2]},leaf:{labels[3]}))"
        )
    return (
        f"and(leaf:{labels[0]},and(leaf:{labels[1]},"
        f"and(leaf:{labels[2]},leaf:{labels[3]})))"
    )


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
        produced = 0
        attempts = 0
        while produced < requested:
            attempts += 1
            if attempts > max(1000, requested * 100):
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
            selected = tuple(rng.sample(pool, arity))
            set_identity = tuple(sorted(item.statement_id for item in selected))
            if set_identity in seen_sets:
                continue
            seen_sets.add(set_identity)
            final_only = global_index % 10 == 0
            generator = (
                f"final-only:{structural}-balanced-v1"
                if final_only
                else f"{structural}-composition-v1"
            )
            source_ids = tuple(item.statement_id for item in selected)
            dag = _dag(structural, source_ids, final_only=final_only)
            family = derivation_family_fingerprint(
                source_ids,
                normalized_proof_dag=dag,
                generator_family=generator,
            )
            plan = CompositionPlan(
                synthetic_name=f"{name_prefix}_{global_index:06d}",
                source_lemmas=selected,
                domain_family=domain_family,
                structural_class=structural,
                generator_family=generator,
                normalized_proof_dag=dag,
                derivation_family_fingerprint=family,
            )
            plan.validate()
            plans.append(plan)
            produced += 1
            global_index += 1
    return plans


def _type_expression(plan: CompositionPlan) -> str:
    types = [
        f"Nonempty (source_type% {item.declaration_name})"
        for item in plan.source_lemmas
    ]
    if plan.structural_class == "direct":
        return f"{types[0]} ∧ {types[1]}"
    if plan.structural_class == "branching":
        return f"({types[0]} ∧ {types[1]}) ∧ {types[2]}"
    if plan.generator_family.startswith("final-only:"):
        return f"({types[0]} ∧ {types[1]}) ∧ ({types[2]} ∧ {types[3]})"
    return f"{types[0]} ∧ ({types[1]} ∧ ({types[2]} ∧ {types[3]}))"


def _oracle_expression(plan: CompositionPlan) -> str:
    names = [f"⟨@{item.declaration_name}⟩" for item in plan.source_lemmas]
    if plan.structural_class == "direct":
        return f"⟨{names[0]}, {names[1]}⟩"
    if plan.structural_class == "branching":
        return f"⟨⟨{names[0]}, {names[1]}⟩, {names[2]}⟩"
    if plan.generator_family.startswith("final-only:"):
        return f"⟨⟨{names[0]}, {names[1]}⟩, ⟨{names[2]}, {names[3]}⟩⟩"
    return f"⟨{names[0]}, ⟨{names[1]}, ⟨{names[2]}, {names[3]}⟩⟩⟩"


def _composition_imports(plans: Sequence[CompositionPlan]) -> tuple[str, ...]:
    pnt_flags = {
        source.source_module.startswith("PrimeNumberTheoremAnd")
        for plan in plans
        for source in plan.source_lemmas
    }
    if len(pnt_flags) != 1:
        raise ValueError("one generated module cannot mix Mathlib and PNT+ source constants")
    if pnt_flags == {True}:
        return ("PrimeNumberTheoremAnd",)
    return tuple(
        sorted({source.source_module for plan in plans for source in plan.source_lemmas})
    )


_LEAN_AUDIT_PRELUDE = r'''
open Lean Elab Term Command Meta

syntax "source_type% " ident : term

elab_rules : term
  | `(source_type% $id:ident) => do
      let info ← getConstInfo id.getId
      let levels ← info.levelParams.mapM fun _ => mkFreshLevelMVar
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
    names = ", ".join(f"`{source.declaration_name}" for source in sources)
    imports = "\n".join(f"import {module}" for module in modules)
    return f'''{imports}

open Lean Elab Command

def datasetV2Presence (names : Array Name) : CommandElabM Unit := do
  let env ← getEnv
  for name in names do
    unless env.contains name do
      logInfo ("{MISSING_PREFIX}" ++ name.toString)

run_cmd datasetV2Presence #[{names}]
'''


def find_missing_constants(
    sources: Sequence[CompositionSource],
    *,
    source_path: Path,
    target_root: Path,
    timeout_seconds: float = 1800.0,
) -> tuple[str, ...]:
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
    return tuple(
        sorted(
            line.split(MISSING_PREFIX, 1)[1].strip()
            for line in output.splitlines()
            if MISSING_PREFIX in line
        )
    )


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
        planned = {item.declaration_name for item in plan.source_lemmas}
        actual = planned.intersection(audit.actual_dependencies)
        if actual != planned:
            missing = sorted(planned - actual)
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

    lines = [*(f"import {module}" for module in _composition_imports(plans)), ""]
    line_to_name: dict[int, str] = {}
    for plan in plans:
        audit = audits[plan.synthetic_name]
        target = _compact_type(audit.statement_type)
        attempts = [
            "fail_if_success assumption",
            "fail_if_success rfl",
            "fail_if_success (solve | simp)",
        ]
        for source in plan.source_lemmas:
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
) -> list[DatasetV2Record]:
    records: list[DatasetV2Record] = []
    for plan in plans:
        audit = audits[plan.synthetic_name]
        declaration = f"theorem {plan.synthetic_name} : {audit.statement_type}"
        proof = f"by\n  exact {_oracle_expression(plan)}"
        identity = statement_id(declaration)
        actual = tuple(
            sorted(
                {item.declaration_name for item in plan.source_lemmas}.intersection(
                    audit.actual_dependencies
                )
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
                imports=("PrimeNumberTheoremAnd",),
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
            shortcut_checks=tuple(shortcut_status.get(plan.synthetic_name, ())),
        )
        record.validate()
        records.append(record)
    return records


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
