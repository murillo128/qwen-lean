from __future__ import annotations

import hashlib
import pickle
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from .dataset_v2_composition import (
    CompositionAudit,
    CompositionPlan,
    CompositionSource,
    build_composition_plans,
    composition_imports,
    find_missing_constants,
    lean_name_key,
    render_composition_source,
    render_shortcut_gate_source,
    run_composition_source,
    run_shortcut_gate_source,
    validate_composition_audits,
)
from .dataset_v2_contract import statement_fingerprint_v2
from .dataset_v2_pipeline import _statement_type_head
from .dataset_v2_schema import EnvironmentContext, ProofVerification
from .dataset_v3 import (
    build_boundaries,
    contains_placeholder,
    dataset_v3_proof_variant_id,
    dataset_v3_statement_id,
    representative_variants,
    structural_proof_fingerprint,
)
from .dataset_v3_schema import (
    DATASET_V3_SCHEMA_VERSION,
    DatasetV3ProofVariant,
    DatasetV3Record,
)


DATASET_V3_DERIVATION_FINGERPRINT_VERSION = "dataset-v3-derivation-family-v1"


def derivation_family_fingerprint(
    source_ids: Sequence[str], *, normalized_proof_dag: str, generator_family: str
) -> str:
    payload = "\0".join(
        (
            DATASET_V3_DERIVATION_FINGERPRINT_VERSION,
            *sorted(source_ids),
            normalized_proof_dag,
            generator_family,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def composition_sources_from_records(
    records: Sequence[DatasetV3Record],
) -> list[CompositionSource]:
    sources: list[CompositionSource] = []
    seen_names: set[str] = set()
    for record in records:
        if record.role != "training" or record.provenance == "synthetic":
            continue
        if "prime-family:pnt-plus" in record.topic_tags:
            continue
        variant = representative_variants(record)[0]
        name = variant.source_declaration_name
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        sources.append(
            CompositionSource(
                statement_id=record.statement_id,
                declaration_name=name,
                source_module=record.environment.module,
                topic_tags=record.topic_tags,
                domain_family="generic",
                canonical_declaration=record.canonical_declaration,
                resolved_dependencies=variant.resolved_dependencies,
                type_head=_statement_type_head(record.canonical_declaration),
            )
        )
    return sorted(sources, key=lambda item: item.declaration_name)


def _v3_generator_family(plan: CompositionPlan) -> str:
    logic_shape = "iff" if plan.generator_family.startswith("final-only:") else "and"
    prefix = "final-only:" if logic_shape == "iff" else ""
    return f"{prefix}{plan.structural_class}-source-{logic_shape}-tactic-v3"


def build_v3_composition_plans(
    sources: Sequence[CompositionSource],
    *,
    count: int,
    seed: str,
) -> list[CompositionPlan]:
    if count < 1:
        raise ValueError("Dataset-v3 composition count must be positive")
    plans = build_composition_plans(
        {"generic": sources},
        {"generic": count},
        seed=seed,
        name_prefix="dataset_v3_synthetic",
    )
    result: list[CompositionPlan] = []
    for plan in plans:
        generator = _v3_generator_family(plan)
        dag = plan.normalized_proof_dag.replace(";path", ";source-proof;path")
        family = derivation_family_fingerprint(
            [item.statement_id for item in plan.source_lemmas],
            normalized_proof_dag=dag,
            generator_family=generator,
        )
        updated = replace(
            plan,
            generator_family=generator,
            normalized_proof_dag=dag,
            derivation_family_fingerprint=family,
        )
        updated.validate()
        result.append(updated)
    return result


def _source_reference(source: CompositionSource) -> str:
    suffix = (
        ".{" + ", ".join(source.universe_arguments) + "}"
        if source.universe_arguments
        else ""
    )
    return f"@{source.declaration_name}{suffix}"


ProofTree = tuple[Any, ...]


def _proof_tree(plan: CompositionPlan) -> ProofTree:
    leaves = tuple(("leaf", index) for index in range(len(plan.source_lemmas)))
    final_only = plan.generator_family.startswith("final-only:")
    if plan.structural_class == "direct" and not final_only:
        return ("and", leaves[0], leaves[1])
    if plan.structural_class == "direct":
        return ("iff", leaves[0], leaves[1])
    if plan.structural_class == "branching" and not final_only:
        return ("and", leaves[0], ("iff", leaves[1], leaves[2]))
    if plan.structural_class == "branching":
        return ("iff", leaves[0], ("and", leaves[1], leaves[2]))
    if final_only:
        return (
            "iff",
            leaves[0],
            ("and", ("and", leaves[1], leaves[2]), leaves[3]),
        )
    return (
        "and",
        ("iff", leaves[0], leaves[1]),
        ("iff", leaves[2], leaves[3]),
    )


def _opening_tactic(connective: str, style: str) -> str:
    if style == "constructor":
        return "constructor"
    if style == "apply":
        return "apply And.intro" if connective == "and" else "apply Iff.intro"
    return "refine ⟨?_, ?_⟩"


def _render_goal(
    tree: ProofTree,
    *,
    plan: CompositionPlan,
    style: str,
    indent: str,
) -> list[str]:
    kind = str(tree[0])
    if kind == "leaf":
        source = plan.source_lemmas[int(tree[1])]
        return [indent + "exact " + _source_reference(source)]
    left, right = tree[1], tree[2]
    lines = [indent + _opening_tactic(kind, style)]
    targets = (left, right) if kind == "and" else (right, left)
    for target in targets:
        lines.append(indent + "·")
        branch_indent = indent + "  "
        if kind == "iff":
            lines.append(branch_indent + "intro _")
        lines.extend(
            _render_goal(
                target,
                plan=plan,
                style=style,
                indent=branch_indent,
            )
        )
    return lines


def render_source_preserving_proof(plan: CompositionPlan) -> str:
    selector = int(
        hashlib.sha256(plan.derivation_family_fingerprint.encode("utf-8")).hexdigest(),
        16,
    )
    style = ("constructor", "apply", "refine")[selector % 3]
    lines = ["by"]
    lines.extend(
        _render_goal(_proof_tree(plan), plan=plan, style=style, indent="  ")
    )
    proof = "\n".join(lines)
    if contains_placeholder(proof):
        raise ValueError("generated Dataset-v3 proof contains a placeholder")
    return proof


def _ground_universes(
    plans: Sequence[CompositionPlan], *, output_dir: Path, target_root: Path
) -> tuple[list[CompositionPlan], set[str]]:
    planned_sources = {
        (source.declaration_name, source.source_module): source
        for plan in plans
        for source in plan.source_lemmas + plan.retrieval_lemmas
    }
    missing_constants: set[str] = set()
    universe_arguments: dict[str, tuple[str, ...]] = {}
    sources = list(planned_sources.values())
    missing, grounding = find_missing_constants(
        sources,
        source_path=output_dir / "lean/ConstantPresence.lean",
        target_root=target_root,
    )
    missing_constants.update(lean_name_key(item) for item in missing)
    universe_arguments.update(
        (lean_name_key(name), arguments) for name, arguments in grounding.items()
    )

    def ground(source: CompositionSource) -> CompositionSource:
        return replace(
            source,
            universe_arguments=universe_arguments.get(
                lean_name_key(source.declaration_name), ()
            ),
        )

    grounded = [
        replace(
            plan,
            source_lemmas=tuple(ground(source) for source in plan.source_lemmas),
            retrieval_lemmas=tuple(
                ground(source)
                for source in plan.retrieval_lemmas
                if lean_name_key(source.declaration_name) not in missing_constants
            ),
            retrieval_index=tuple(
                item
                for item in plan.retrieval_index
                if lean_name_key(item[0]) not in missing_constants
            ),
        )
        for plan in plans
        if not any(
            lean_name_key(source.declaration_name) in missing_constants
            for source in plan.source_lemmas
        )
    ]
    return grounded, missing_constants


def _verify_plan_batches(
    plans: Sequence[CompositionPlan],
    *,
    output_dir: Path,
    target_root: Path,
    batch_size: int,
    workers: int,
) -> tuple[
    dict[str, CompositionAudit],
    set[str],
    dict[str, tuple[str, ...]],
    list[dict[str, Any]],
]:
    audits: dict[str, CompositionAudit] = {}
    shortcut_rejected: set[str] = set()
    verified_imports: dict[str, tuple[str, ...]] = {}
    batch_evidence: list[dict[str, Any]] = []
    batches = [
        list(plans[start : start + batch_size])
        for start in range(0, len(plans), batch_size)
    ]
    if workers < 1:
        raise ValueError("Dataset-v3 composition workers must be positive")

    def verify_batch(
        indexed_batch: tuple[int, list[CompositionPlan]],
    ) -> tuple[int, list[CompositionPlan], Any, Any, dict[str, Any]]:
        batch_index, batch = indexed_batch
        composition_path = output_dir / f"lean/Composition-{batch_index:04d}.lean"
        composition_path.parent.mkdir(parents=True, exist_ok=True)
        composition_source = render_composition_source(batch)
        composition_path.write_text(composition_source, encoding="utf-8")
        cache_key = hashlib.sha256(
            (
                "dataset-v3-composition-batch-cache-v1\0"
                + composition_source
            ).encode("utf-8")
        ).hexdigest()
        cache_path = output_dir / f".batch-cache/{batch_index:04d}-{cache_key}.pkl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.is_file():
            with cache_path.open("rb") as handle:
                run, shortcut_run, summary = pickle.load(handle)
            shortcut_source, _ = render_shortcut_gate_source(
                batch, {audit.name: audit for audit in run.audits}
            )
            shortcut_path = output_dir / f"lean/ShortcutGate-{batch_index:04d}.lean"
            shortcut_path.write_text(shortcut_source, encoding="utf-8")
        else:
            run = run_composition_source(composition_path, target_root=target_root)
            if run.status != "accepted":
                raise RuntimeError(
                    f"Dataset-v3 composition batch {batch_index} failed: {run.diagnostic}"
                )
            summary = validate_composition_audits(batch, run.audits)
            shortcut_source, line_map = render_shortcut_gate_source(
                batch, {audit.name: audit for audit in run.audits}
            )
            shortcut_path = output_dir / f"lean/ShortcutGate-{batch_index:04d}.lean"
            shortcut_path.write_text(shortcut_source, encoding="utf-8")
            shortcut_run = run_shortcut_gate_source(
                shortcut_path, target_root=target_root, line_to_name=line_map
            )
            if shortcut_run.status not in {"accepted", "rejected-shortcuts"}:
                raise RuntimeError(
                    f"Dataset-v3 shortcut batch {batch_index} failed: "
                    f"{shortcut_run.diagnostic}"
                )
            temporary = cache_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                pickle.dump(
                    (run, shortcut_run, summary),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            temporary.replace(cache_path)
        return batch_index, batch, run, shortcut_run, summary

    with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as executor:
        results = list(executor.map(verify_batch, enumerate(batches)))
    for batch_index, batch, run, shortcut_run, summary in sorted(results):
        audits.update((audit.name, audit) for audit in run.audits)
        shortcut_rejected.update(shortcut_run.rejected_names)
        imports = composition_imports(batch)
        verified_imports.update((plan.synthetic_name, imports) for plan in batch)
        batch_evidence.append(
            {
                "batch": batch_index,
                "composition_status": run.status,
                "shortcut_status": shortcut_run.status,
                "shortcut_rejections": len(shortcut_run.rejected_names),
                **summary,
            }
        )
    return audits, shortcut_rejected, verified_imports, batch_evidence


def _balanced_role_selection(
    plans: Sequence[CompositionPlan],
    *,
    role_counts: Mapping[str, int],
    seed: str,
) -> dict[str, str]:
    buckets: dict[tuple[str, str], list[CompositionPlan]] = defaultdict(list)
    for plan in plans:
        logic = "iff" if plan.generator_family.startswith("final-only:") else "and"
        buckets[(plan.structural_class, logic)].append(plan)
    strata = sorted(buckets)
    selected: dict[str, str] = {}
    used: set[str] = set()
    roles = ("training", "validation", "test")
    produced_by_role = {role: 0 for role in roles}

    for role in roles:
        if int(role_counts.get(role, 0)) < len(strata):
            continue
        for stratum in strata:
            candidates = sorted(
                buckets[stratum],
                key=lambda plan: hashlib.sha256(
                    f"{seed}\0{role}\0reserve\0{plan.derivation_family_fingerprint}".encode()
                ).hexdigest(),
            )
            plan = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.derivation_family_fingerprint not in used
                ),
                None,
            )
            if plan is None:
                raise RuntimeError(
                    f"Dataset-v3 composition reserve lacks {role}:{stratum}"
                )
            used.add(plan.derivation_family_fingerprint)
            selected[plan.synthetic_name] = role
            produced_by_role[role] += 1

    for role in roles:
        requested = int(role_counts.get(role, 0))
        ordered = {
            stratum: sorted(
                (
                    plan
                    for plan in buckets[stratum]
                    if plan.derivation_family_fingerprint not in used
                ),
                key=lambda plan: hashlib.sha256(
                    f"{seed}\0{role}\0{plan.derivation_family_fingerprint}".encode()
                ).hexdigest(),
            )
            for stratum in strata
        }
        cursors = {stratum: 0 for stratum in strata}
        produced = produced_by_role[role]
        while produced < requested:
            progress = False
            for stratum in strata:
                candidates = ordered[stratum]
                cursor = cursors[stratum]
                while (
                    cursor < len(candidates)
                    and candidates[cursor].derivation_family_fingerprint in used
                ):
                    cursor += 1
                cursors[stratum] = cursor
                if cursor >= len(candidates):
                    continue
                plan = candidates[cursor]
                cursors[stratum] += 1
                used.add(plan.derivation_family_fingerprint)
                selected[plan.synthetic_name] = role
                produced += 1
                progress = True
                if produced == requested:
                    break
            if not progress:
                raise RuntimeError(
                    f"Dataset-v3 composition reserve cannot fill {role}: "
                    f"{produced}/{requested}"
                )
    selected_by_name = {plan.synthetic_name: plan for plan in plans}
    if all(int(role_counts.get(role, 0)) >= len(strata) for role in role_counts):
        missing_strata = {
            role: sorted(
                set(strata)
                - {
                    (
                        selected_by_name[name].structural_class,
                        "iff"
                        if selected_by_name[name].generator_family.startswith(
                            "final-only:"
                        )
                        else "and",
                    )
                    for name, assigned_role in selected.items()
                    if assigned_role == role
                }
            )
            for role in role_counts
        }
        missing_strata = {
            role: missing for role, missing in missing_strata.items() if missing
        }
        if missing_strata:
            raise RuntimeError(
                f"Dataset-v3 roles do not cover every composition stratum: {missing_strata}"
            )
    return selected


def _synthetic_record(
    plan: CompositionPlan,
    audit: CompositionAudit,
    *,
    role: str,
    environment: Mapping[str, Any],
    imports: Sequence[str],
    evidence_id: str,
) -> DatasetV3Record:
    declaration = f"theorem {plan.synthetic_name} : {audit.statement_type}"
    statement_id = dataset_v3_statement_id(declaration)
    proof = render_source_preserving_proof(plan)
    proof_variant_id = dataset_v3_proof_variant_id(
        statement_id,
        proof,
        source_repository="generated:dataset-v3-composition-v1",
        source_revision=evidence_id,
        source_file="DatasetV3Generated.lean",
    )
    dependency_keys = {lean_name_key(item) for item in audit.actual_dependencies}
    dependencies = tuple(
        sorted(
            item.declaration_name
            for item in plan.source_lemmas
            if lean_name_key(item.declaration_name) in dependency_keys
        )
    )
    if len(dependencies) != len(plan.source_lemmas):
        raise ValueError(f"generated proof dependency audit is incomplete: {plan.synthetic_name}")
    variant = DatasetV3ProofVariant(
        proof_variant_id=proof_variant_id,
        source_expression=proof,
        proof_text=proof,
        proof_form="generated-tactic",
        transformation_kind="none",
        transformation_reason=None,
        exact_text_fingerprint=hashlib.sha256(proof.encode("utf-8")).hexdigest(),
        structural_fingerprint=structural_proof_fingerprint(proof, dependencies),
        resolved_dependencies=dependencies,
        boundaries=build_boundaries(proof, proof_variant_id),
        verification=ProofVerification(
            status="accepted",
            environment_id=str(environment["environment_id"]),
            method="generated-module-elaboration-and-shortcut-gate",
            evidence_id=evidence_id,
        ),
        source_declaration_name=plan.synthetic_name,
        source_repository="generated:dataset-v3-composition-v1",
        source_revision=evidence_id,
        source_file="DatasetV3Generated.lean",
        source_expression_verified=True,
    )
    name_to_id = {
        source.declaration_name: source.statement_id for source in plan.source_lemmas
    }
    record = DatasetV3Record(
        schema_version=DATASET_V3_SCHEMA_VERSION,
        statement_id=statement_id,
        canonical_declaration=declaration,
        normalized_statement_fingerprint=statement_fingerprint_v2(declaration),
        role=role,  # type: ignore[arg-type]
        theorem_mass_numerator=1,
        theorem_mass_denominator=1,
        provenance="synthetic",
        environment=EnvironmentContext(
            environment_id=str(environment["environment_id"]),
            lean_toolchain=str(environment["lean_toolchain"]),
            repository="generated:dataset-v3-composition-v1",
            revision=evidence_id,
            mathlib_revision=str(environment["mathlib_revision"]),
            file_path="DatasetV3Generated.lean",
            module="DatasetV3Generated",
            imports=tuple(imports),
            source_span=None,
            context_kind="generated-module",
            target_compatibility="verified-target-environment",
        ),
        proof_variants=(variant,),
        topic_tags=("domain:generic", "composition:dataset-v3"),
        memberships=(f"dataset-v3-{role}",),
        derivation_family_fingerprint=plan.derivation_family_fingerprint,
        generator_family=plan.generator_family,
        structural_class=plan.structural_class,
        logic_shape=(
            "iff" if plan.generator_family.startswith("final-only:") else "and"
        ),
        normalized_proof_dag=plan.normalized_proof_dag,
        source_lemma_ids=tuple(item.statement_id for item in plan.source_lemmas),
        source_relation_edges=tuple(
            (name_to_id[left], name_to_id[right], relation)
            for left, right, relation in plan.relation_edges
        ),
        shortcut_checks=(
            "assumption:no-closure",
            "rfl:no-closure",
            "simp:no-closure",
            "exact-indexed-single-theorem:no-closure",
            "simpa-using-indexed-single-theorem:no-closure",
        ),
    )
    record.validate()
    return record


def _verify_persisted_records(
    records: Sequence[DatasetV3Record], *, source_path: Path, target_root: Path
) -> dict[str, Any]:
    if not records:
        raise ValueError("no Dataset-v3 synthetic records to verify")
    imports = records[0].environment.imports
    if any(record.environment.imports != imports for record in records):
        raise ValueError("persisted Dataset-v3 verification batch mixes imports")
    lines = [*(f"import {module}" for module in imports), "", "set_option maxHeartbeats 1000000", ""]
    for record in sorted(records, key=lambda item: item.statement_id):
        lines.append(
            f"{record.canonical_declaration} := {record.proof_variants[0].proof_text}"
        )
        lines.append("")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("\n".join(lines), encoding="utf-8")
    completed = subprocess.run(
        ["lake", "env", "lean", "-E", "hasSorry", str(source_path.resolve())],
        cwd=target_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "persisted Dataset-v3 synthetic verification failed: "
            + (completed.stdout + completed.stderr)[-4000:]
        )
    return {
        "status": "accepted",
        "records": len(records),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }


def generate_v3_compositions(
    sources: Sequence[CompositionSource],
    *,
    role_counts: Mapping[str, int],
    seed: str,
    reserve_count: int,
    forbidden_statement_fingerprints: set[str],
    forbidden_exact_proof_fingerprints: set[str],
    forbidden_structural_proof_fingerprints: set[str],
    forbidden_derivation_fingerprints: set[str],
    environment: Mapping[str, Any],
    output_dir: Path,
    target_root: Path,
    batch_size: int,
    workers: int,
) -> tuple[list[DatasetV3Record], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plans = build_v3_composition_plans(
        sources, count=reserve_count, seed=seed
    )
    plans, missing_constants = _ground_universes(
        plans, output_dir=output_dir, target_root=target_root
    )
    audits, shortcut_rejected, verified_imports, batches = _verify_plan_batches(
        plans,
        output_dir=output_dir,
        target_root=target_root,
        batch_size=batch_size,
        workers=workers,
    )
    eligible: list[CompositionPlan] = []
    collision_counts = {
        "statement": 0,
        "exact-proof": 0,
        "structural-proof": 0,
        "derivation": 0,
    }
    provisional_records: dict[str, DatasetV3Record] = {}
    for plan in plans:
        if plan.synthetic_name in shortcut_rejected:
            continue
        audit = audits[plan.synthetic_name]
        evidence_id = "dataset-v3-composition:" + hashlib.sha256(
            (plan.synthetic_name + plan.derivation_family_fingerprint).encode("utf-8")
        ).hexdigest()
        record = _synthetic_record(
            plan,
            audit,
            role="training",
            environment=environment,
            imports=verified_imports[plan.synthetic_name],
            evidence_id=evidence_id,
        )
        variant = record.proof_variants[0]
        collisions = (
            record.normalized_statement_fingerprint
            in forbidden_statement_fingerprints,
            variant.exact_text_fingerprint in forbidden_exact_proof_fingerprints,
            variant.structural_fingerprint
            in forbidden_structural_proof_fingerprints,
            plan.derivation_family_fingerprint in forbidden_derivation_fingerprints,
        )
        if any(collisions):
            for key, collided in zip(
                ("statement", "exact-proof", "structural-proof", "derivation"),
                collisions,
                strict=True,
            ):
                collision_counts[key] += int(collided)
            continue
        eligible.append(plan)
        provisional_records[plan.synthetic_name] = record

    assignments = _balanced_role_selection(
        eligible, role_counts=role_counts, seed=seed + ":roles"
    )
    records = [
        replace(provisional_records[name], role=role, memberships=(f"dataset-v3-{role}",))
        for name, role in assignments.items()
    ]
    records.sort(key=lambda item: (item.role, item.statement_id))
    by_imports: dict[tuple[str, ...], list[DatasetV3Record]] = defaultdict(list)
    for record in records:
        record.validate()
        by_imports[record.environment.imports].append(record)
    persisted = []
    for index, group in enumerate(by_imports.values()):
        persisted.append(
            _verify_persisted_records(
                group,
                source_path=output_dir / f"lean/Persisted-{index:04d}.lean",
                target_root=target_root,
            )
        )
    role_summary = defaultdict(int)
    structural_summary = defaultdict(int)
    logic_summary = defaultdict(int)
    first_summary = defaultdict(int)
    role_strata_summary: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for record in records:
        role_summary[record.role] += 1
        structural_summary[str(record.structural_class)] += 1
        logic_summary[str(record.logic_shape)] += 1
        proof = record.proof_variants[0].proof_text
        first = proof.splitlines()[1].strip().split()[0]
        first_summary[first] += 1
        role_strata_summary[record.role][
            f"{record.structural_class}:{record.logic_shape}"
        ] += 1
    return records, {
        "requested_roles": dict(role_counts),
        "accepted_roles": dict(sorted(role_summary.items())),
        "reserve_plans": reserve_count,
        "presence_eligible_plans": len(plans),
        "shortcut_rejections": len(shortcut_rejected),
        "missing_source_constants": len(missing_constants),
        "v2_collision_rejections": dict(sorted(collision_counts.items())),
        "structural_classes": dict(sorted(structural_summary.items())),
        "logic_shapes": dict(sorted(logic_summary.items())),
        "first_constructs": dict(sorted(first_summary.items())),
        "role_strata": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(role_strata_summary.items())
        },
        "verification_batches": batches,
        "persisted_verification": persisted,
    }
