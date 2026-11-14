"""One fresh generator or static-LoRA run using the GEN-001 protocol."""

import json
import re
from pathlib import Path
from typing import Literal

import yaml
from filelock import FileLock
from huggingface_hub import HfApi
from huggingface_hub.utils import validate_repo_id
from pydantic import BaseModel, ConfigDict, Field, model_validator

from speculators.generator.campaign_config import DOMAINS, CampaignConfig


class EvaluationRows(BaseModel):
    """Fixed evaluation rows (user 2026-11-09). Subsets are named by their
    r0001 index `<category>-0000<i>`; only their validation splits are used.

    - validation: `rows_per_subset` seeded rows of each `*-00000` validation
      split (50 per category); training validation, base and every offline
      eval use them.
    - test: the same count from `*-0000<test_subset>`, never used in
      training or validation (final numbers only).
    - donors: the whole `*-0000<donor_subset>` validation split, never
      trained on or evaluated (M7 donor pool).
    """

    model_config = ConfigDict(extra="forbid")
    rows_per_subset: int = Field(ge=1)
    seed: int
    # Required without a `pool`; with a `pool` its subset lists are used.
    test_subset: int | None = Field(default=None, ge=1)
    donor_subset: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def distinct(self):
        if self.test_subset is not None and self.test_subset == self.donor_subset:
            raise ValueError("test and donor subsets must differ")
        return self


SUBSET_NAME = r"^(math|code|stem|chat)-\d{5}$"


class Pool(BaseModel):
    """Explicit training and evaluation subsets (user 2026-11-10: the 10K-math
    run). Training uses the train splits of `train_subsets`; validation and
    test use the validation splits of their subsets (`rows_per_subset` seeded
    rows each). Validation and test subsets are disjoint. A train subset may
    also be an evaluated subset (user 2026-11-14, the 100K run): training reads
    only its train split, and the evaluated rows in `rows` were checked never
    to occur in any train split of any local release (by primary id, prompt
    hash and full-sequence hash; `row_ids` refuses otherwise). There is no
    donor pool (offline eval then runs the correct donor only)."""

    model_config = ConfigDict(extra="forbid")
    train_subsets: list[str] = Field(min_length=1)
    validation_subsets: list[str] = Field(min_length=1)
    test_subsets: list[str] = Field(min_length=1)
    # The stored evaluation rows (make_eval_splits.py; repository-relative):
    # every validation / test row by primary id, checked never to occur in a
    # train split. Its subsets and roles must equal the two lists above.
    rows: Path

    @model_validator(mode="after")
    def disjoint(self):
        lists = (self.train_subsets, self.validation_subsets, self.test_subsets)
        names = [name for names in lists for name in names]
        bad = [name for name in names if not re.fullmatch(SUBSET_NAME, name)]
        if bad:
            raise ValueError(f"Subset names must be <category>-NNNNN: {bad}")
        if any(len(set(names)) != len(names) for names in lists):
            raise ValueError("Pool subsets must be unique within each list")
        if set(self.validation_subsets) & set(self.test_subsets):
            raise ValueError("Validation and test subsets must be disjoint")
        return self

    def shares_subsets(self):
        """True when a train subset is also evaluated (row-disjoint only)."""
        return bool(
            set(self.train_subsets) & set(self.validation_subsets + self.test_subsets)
        )

    def row_ids(self):
        """{subset: [primary ids]} from `rows`, after checking it matches."""
        record = json.loads(self.rows.read_text())
        roles = {
            role: [n for n, s in record["subsets"].items() if s["role"] == role]
            for role in ("validation", "test")
        }
        if roles != {"validation": self.validation_subsets, "test": self.test_subsets}:
            raise ValueError(f"{self.rows} subsets differ from the pool lists")
        if record["check"].get("overlap") != 0:
            raise ValueError(f"{self.rows} has no clean overlap check")
        if self.shares_subsets() and record.get("release") not in record["check"].get(
            "releases", []
        ):
            raise ValueError(
                f"{self.rows}: train and evaluated subsets overlap, but the overlap "
                "check did not cover the train splits of this release"
            )
        return {
            name: [row["primary_id"] for row in subset["rows"]]
            for name, subset in record["subsets"].items()
        }


class StructureStudy(CampaignConfig):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    family: Literal["modules", "architecture", "baseline"]
    arm: Literal["generator", "static_lora"] = "generator"
    lora_dtype: Literal["float32", "bfloat16"] = "bfloat16"
    repo_id: str
    provenance_dir: Path
    private: bool = False
    # False: local checkpoints only (smoke runs); no HF repo is created.
    hub_push: bool = True
    # None: the EAGLE-3 protocol (every validation row, no test/donor splits).
    evaluation_rows: EvaluationRows | None = None
    # None: the 4K layout (`<category>-00000` of every category).
    pool: Pool | None = None

    @model_validator(mode="after")
    def fresh_original_protocol(self):
        validate_repo_id(self.repo_id)
        rows = self.evaluation_rows
        if self.pool is not None:
            if rows is None:
                raise ValueError("A pool needs evaluation_rows (count and seed)")
            if rows.test_subset is not None or rows.donor_subset is not None:
                raise ValueError("With a pool, test subsets come from pool lists")
        elif rows is not None and (
            rows.test_subset is None or rows.donor_subset is None
        ):
            raise ValueError("evaluation_rows needs test_subset and donor_subset")
        if self.activation_source != "vllm" or self.pretrained_checkpoint is not None:
            raise ValueError(
                "Structure studies train fresh adapters/generators "
                "from saved trajectories"
            )
        if (
            self.training_cohort_size is not None
            or self.training_conditioning_examples is not None
        ):
            raise ValueError("Structure studies use the original random 1–8 cohorts")
        return self

    @classmethod
    def load(cls, path: Path):
        """Resolve a common YAML relative to this file and merge its sections."""
        path = Path(path).resolve()
        values = yaml.safe_load(path.read_text())
        base_path = path
        if "extends" in values:
            base_path = (path.parent / values.pop("extends")).resolve()
            common = yaml.safe_load(base_path.read_text())
            if "extends" in common:
                raise ValueError("Only one common config layer is supported")
            for key, value in values.items():
                if isinstance(value, dict) and isinstance(common.get(key), dict):
                    common[key] = {**common[key], **value}
                else:
                    common[key] = value
            values = common
        for key in ("training_recipe", "trajectory_recipe"):
            if values.get(key) is not None:
                recipe = Path(values[key])
                if not recipe.is_absolute():
                    recipe = base_path.parent / recipe
                values[key] = recipe.resolve()
        values["bank_config"] = values["training_recipe"]
        return cls.model_validate(values)


def training_job(cfg):
    pool = cfg.pool
    subsets = (
        list(pool.train_subsets)
        if pool is not None
        else [f"{domain}-00000" for domain in DOMAINS]
    )
    validation = list(pool.validation_subsets) if pool is not None else subsets
    job = {
        "kind": "pretraining",
        "stage": "pretraining",
        "condition": "last",
        "label": cfg.name,
        "arm": "fresh_lora" if cfg.arm == "static_lora" else "direct",
        "objective": "direct",
        "subsets": subsets,
        "validation_subsets": validation,
        "updates": cfg.optimization.pretraining_updates,
        "training_examples": len(subsets) * cfg.expected_train_examples,
        "tracking_group": f"structure-v1/{cfg.family}",
        "vllm_provenance_dir": str(cfg.provenance_dir.resolve()),
        "drafter_model_path": str(cfg.online.drafter_model_path.resolve()),
    }
    rows = cfg.evaluation_rows
    if rows is not None:
        job["validation_rows_per_subset"] = rows.rows_per_subset
        job["validation_rows_seed"] = rows.seed
        if pool is not None:
            job["test_subsets"] = list(pool.test_subsets)
            job["validation_row_ids"] = pool.row_ids()
            counts = {len(ids) for ids in job["validation_row_ids"].values()}
            if counts != {rows.rows_per_subset}:
                raise ValueError("Stored evaluation rows differ from rows_per_subset")
        else:
            job["test_subsets"] = [f"{d}-{rows.test_subset:05d}" for d in DOMAINS]
            job["donor_subsets"] = [f"{d}-{rows.donor_subset:05d}" for d in DOMAINS]
    if cfg.hub_push:
        job["hub_upload_interval"] = cfg.hub_upload_interval
        job["hub_repo"] = cfg.repo_id
    if cfg.arm == "static_lora":
        job["adapter_dtype"] = cfg.lora_dtype
    return job


def train_structure(cfg, *, dry_run=False):
    """Resume locally if present; launch exactly one historical episode schedule."""
    job = training_job(cfg)
    if dry_run:
        return {"job": job, "configuration": cfg.model_dump(mode="json")}
    from speculators.generator.campaign import run_one  # noqa: PLC0415
    from speculators.generator.publish import extraction_provenance  # noqa: PLC0415

    if extraction_provenance(cfg.output_root, cfg.provenance_dir) is None:
        raise FileNotFoundError("Launch the extractor with --provenance-dir first")
    if cfg.hub_push:
        api = HfApi()
        api.whoami()
        api.create_repo(
            repo_id=cfg.repo_id, repo_type="model", exist_ok=True, private=cfg.private
        )
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cfg.output_root / ".structure.lock"), timeout=0):
        return run_one(cfg, job)
