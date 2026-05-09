from __future__ import annotations

import ast
import json
import os
import os.path as osp
import pickle
import re
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..smp.file import LMUDataRoot, get_intermediate_file_path, load
from ..smp import read_ok, toliststr
from .image_base import ImageBaseDataset, img_root_map
from .utils.spatial_bench.cal_scores import (
    attach_score_cache,
    build_mcq_score_fn,
    build_na_score_fn,
    mean_relative_accuracy,
    to_float,
)
from .utils.spatial_bench.matching_func import can_match_na
from .utils.spatial_bench.tools.files import build_eval_paths, get_judge_tag_from_score_fn

DEGRADATIONS = [
    "defocus",
    "distortion",
    "haze",
    "jpeg_compression",
    "low_light",
    "low_res",
    "motion_blur",
    "over_exposure",
    "water_droplets",
]

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Please carefully inspect the provided images and "
    "correctly answer their relevant spatial and appearance questions. "
    "For each question, output only one final answer without any other text. \n\n"
    "You must wrap your final extracted succinct answer in `<answer>` and `</answer>` tags format "
    "(e.g., `<answer>A</answer>`).\n"
)

DEG_PARAM_RANGES = {
    "defocus": {"aperture_min": 10.0, "aperture_max": 15.0, "depth_min": 1.0, "depth_max": 8.0},
    "distortion": {
        "k1_min": -0.24,
        "k1_max": -0.23,
        "k2_min": 0.0001,
        "k2_max": 0.0003,
        "k3_min": 0.0001,
        "k3_max": 0.0002,
        "k4_min": 0.0000,
        "k4_max": 0.0001,
        "max_theta": 1.5,
    },
    "haze": {"density_min": 3.5, "density_max": 6.0},
    "jpeg_compression": {"quality_min": 2, "quality_max": 5},
    "low_light": {"exposure_min": 0.003, "exposure_max": 0.005},
    "low_res": {"scale_min": 0.02, "scale_max": 0.05},
    "motion_blur": {
        "trans_min": 0.2,
        "trans_max": 0.35,
        "rot_min": 0.06,
        "rot_max": 0.12,
        "sub_steps_min": 80,
        "sub_steps_max": 80,
    },
    "over_exposure": {"exposure_min": 7.0, "exposure_max": 10.0},
    "water_droplets": {
        "scale_min": 2.5,
        "scale_max": 4.0,
        "radius_min": 0.25,
        "radius_max": 0.75,
        "strength_min": 0.3,
        "strength_max": 0.5,
        "blur_sigma_min": 2.0,
        "blur_sigma_max": 2.5,
        "blur_kernel": 9,
    },
}

_REQUIRED_TSV_COLS = frozenset(
    {
        "index",
        "question",
        "answer",
        "image_path",
        "question_type",
        "sub_question_type",
        "task_group",
        "degradation_type",
    }
)

_QT_WHEN_EMPTY_SQT = {
    "distance_estimation": "NA",
    # TSV variants (observed in SpaceDG/LMUData/spacedg_bench.tsv)
    "camera_object_distance_estimation": "NA",
    "existence_estimation": "MCQ",
    "object_existence_estimation": "MCQ",
    "object_counting": "COUNT",
    "size_estimation": "SIZE_LIST",
    "object_bounding_size_estimation": "SIZE_LIST",
}
_NA_SUB_QUESTION_TYPES = frozenset({"distance_exact", "bbox_center_distance", "inter_object_distance"})


def _parse_image_list_cell(cell) -> list[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if isinstance(cell, list):
        return [str(p) for p in cell]
    s = str(cell).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return [s]
        if isinstance(v, list):
            return [str(p) for p in v]
    return [s]


def _parse_gt_numeric(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    s = re.sub(r"\s*m(eters)?$", "", s, flags=re.I).strip()
    try:
        return float(s)
    except ValueError:
        return to_float(s)


def _normalize_yes_no(s: str) -> Optional[str]:
    t = str(s).strip().lower()
    if t in ("yes", "no"):
        return t
    return None


def _extract_yes_no_from_pred(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = re.search(
        r"<\s*answer\b[^>]*>\s*(yes|no)\s*<\s*/\s*answer\s*>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).lower()
    tail = text[-800:]
    matches = list(re.finditer(r"\b(yes|no)\b", tail, flags=re.IGNORECASE))
    if matches:
        return matches[-1].group(1).lower()
    return None


def _compute_yn_score(df: pd.DataFrame) -> pd.DataFrame:
    hits, pred_e = [], []
    for _, r in df.iterrows():
        gt = _normalize_yes_no(str(r["answer"]))
        pe = _extract_yes_no_from_pred(str(r["prediction"]))
        pred_e.append(pe or "")
        hits.append(1.0 if (gt is not None and pe is not None and pe == gt) else 0.0)
    out = df.copy()
    out["pred_extracted"] = pred_e
    out["hit"] = hits
    return out


def _parse_gt_int(val) -> Optional[int]:
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return None


def _extract_int_from_pred(text: str) -> Optional[int]:
    p = can_match_na(str(text))
    if p is None:
        return None
    try:
        return int(round(float(p)))
    except (ValueError, TypeError):
        return None


def _compute_count_score(df: pd.DataFrame) -> pd.DataFrame:
    hits, pe = [], []
    for _, r in df.iterrows():
        gt = _parse_gt_int(r["answer"])
        pred = _extract_int_from_pred(str(r["prediction"]))
        pe.append(pred if pred is not None else "")
        hits.append(1.0 if (gt is not None and pred is not None and pred == gt) else 0.0)
    out = df.copy()
    out["pred_extracted"] = pe
    out["hit"] = hits
    return out


def _parse_list_floats(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    try:
        x = ast.literal_eval(s)
        if isinstance(x, list) and x and all(isinstance(i, (int, float)) for i in x):
            return [float(i) for i in x]
    except (SyntaxError, ValueError, TypeError):
        pass
    return None


def _extract_list_from_pred(text: str, need_len: int):
    if not isinstance(text, str) or need_len <= 0:
        return None
    t = text.strip()
    a, b = t.find("["), t.rfind("]")
    if a != -1 and b != -1 and b > a:
        try:
            x = ast.literal_eval(t[a : b + 1])
            if isinstance(x, list) and len(x) == need_len and all(isinstance(i, (int, float)) for i in x):
                return [float(i) for i in x]
        except (SyntaxError, ValueError, TypeError):
            pass
    t_clean = re.sub(r"(?m)^\s*\d+[\.\:]\s+", "", t)
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", t_clean)
    if len(nums) >= need_len:
        try:
            return [float(x) for x in nums[:need_len]]
        except (ValueError, TypeError):
            return None
    return None


def _compute_size_list_score(df: pd.DataFrame) -> pd.DataFrame:
    mras, pe = [], []
    for _, r in df.iterrows():
        gt = _parse_list_floats(r["answer"])
        if gt is None:
            pe.append("")
            mras.append(0.0)
            continue
        pred = _extract_list_from_pred(str(r["prediction"]), need_len=len(gt))
        if pred is None:
            pe.append("")
            mras.append(0.0)
            continue
        pe.append(pred)
        # `mean_relative_accuracy` expects scalar float inputs (pred, target).
        # For size-list tasks, compute per-element MRA then average.
        per_elem = [
            mean_relative_accuracy(float(p), float(t), 0.5, 0.95, 0.05) for p, t in zip(pred, gt)
        ]
        mras.append(float(np.mean(per_elem)) if per_elem else 0.0)

    out = df.copy()
    out["pred_extracted"] = pe
    # Keep the same metric name as the reference EASI implementation
    out["MRA:.5:.95:.05"] = mras
    return out


def infer_eval_task(row: pd.Series) -> str:
    """
    Align with the reference EASI implementation:
    - COUNT / SIZE_LIST are inferred only when sub_question_type is empty (via _QT_WHEN_EMPTY_SQT)
    - YN is only `distance_threshold`
    - NA is for numeric subtypes in _NA_SUB_QUESTION_TYPES
    - otherwise MCQ
    """
    qt = str(row["question_type"]).strip()
    sqt_raw = row["sub_question_type"]
    empty_sqt = pd.isna(sqt_raw) or str(sqt_raw).strip() == ""

    if empty_sqt:
        if qt not in _QT_WHEN_EMPTY_SQT:
            raise ValueError(
                "SpaceDGBench: sub_question_type is empty but question_type is not in "
                f"{set(_QT_WHEN_EMPTY_SQT)}. got={qt!r} (index={row.get('index')})"
            )
        return _QT_WHEN_EMPTY_SQT[qt]

    sqt = str(sqt_raw).strip()
    if sqt == "distance_threshold":
        return "YN"
    if sqt in _NA_SUB_QUESTION_TYPES:
        return "NA"
    return "MCQ"


class SpaceDGBench(ImageBaseDataset):
    """
    SpaceDG Bench (EASI local benchmark layout).

    This implementation expects images to be prepared under LMUData/images/spacedg_bench/
    and a TSV annotation file under LMUData/spacedg_bench/spacedg_bench.tsv (or via env overrides).

    If you currently have an HF-style parquet (with an `images` column), run:
      VLMEvalKit/scripts/preprocess_spacedg_bench_parquet.py
    """

    TYPE = "MIXED"
    MODALITY = "IMAGE"

    LMUData_root = LMUDataRoot()

    @classmethod
    def supported_datasets(cls):
        return ["spacedg_bench"]

    def __init__(
        self,
        dataset="spacedg_bench",
        prompt_with_degradation_info: Optional[bool] = None,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    ):
        self.system_prompt = system_prompt
        self.prompt_with_degradation_info = (
            os.environ.get("PROMPT_WITH_DEGRADATION_INFO", "False").lower() in ("true", "1", "yes")
            if prompt_with_degradation_info is None
            else bool(prompt_with_degradation_info)
        )
        super().__init__(dataset=dataset, skip_noimg=True)
        # Keep a default img_root for compatibility, but we override `dump_image()` to resolve
        # multiple common relative path conventions robustly.
        self.img_root = osp.join(LMUDataRoot(), "images", img_root_map(dataset))
        # Use `dump_image()` so relative `image_path` can be resolved with `self.img_root`.
        # We do not read parquet bytes at runtime.
        self.meta_only = False
        self._tsv_dir: Optional[str] = None

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        if self.system_prompt:
            line = line.copy()
            line["question"] = str(self.system_prompt) + str(line.get("question", ""))
        return super().build_prompt(line)

    @staticmethod
    def _dataset_root() -> str:
        # Allow explicit override.
        env = os.environ.get("SPACEDG_BENCH_ROOT", "").strip()
        if env:
            return osp.expanduser(osp.expandvars(env))
        return LMUDataRoot()

    @classmethod
    def _tsv_path(cls) -> str:
        env = os.environ.get("SPACEDG_BENCH_TSV_LMU", "").strip()
        if env:
            return osp.expanduser(osp.expandvars(env))
        return osp.join(cls._dataset_root(), "spacedg_bench.tsv")

    def load_data(self, dataset):
        tsv_path = self._tsv_path()
        if not osp.isfile(tsv_path):
            raise FileNotFoundError(
                f"SpaceDGBench TSV not found: {tsv_path}. "
                "Run `VLMEvalKit/scripts/preprocess_spacedg_bench_parquet.py` "
                "to generate TSV + extracted images under LMUData."
            )

        df = pd.read_csv(tsv_path, sep="\t")
        self._tsv_dir = osp.dirname(tsv_path)

        missing = _REQUIRED_TSV_COLS - set(df.columns)
        if missing:
            raise ValueError(f"SpaceDGBench: TSV missing columns {sorted(missing)}. tsv={tsv_path}")

        # Optional degradation prompt prefix (prepended to question)
        if self.prompt_with_degradation_info:
            deg = df.get("degradation_type", "")

            def _make_deg_prompt(d):
                d = str(d)
                if d and d != "ori":
                    params = DEG_PARAM_RANGES.get(d, {})
                    if params:
                        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                        return f"Note: The provided images have a '{d}' degradation with parameters: {param_str}.\n"
                    return f"Note: The provided images have a '{d}' degradation.\n"
                return "Note: The provided images are original images with no degradation.\n"

            df["question"] = deg.map(_make_deg_prompt) + df["question"].astype(str)

        return df

    def dump_image(self, line):
        """
        Resolve `image_path` robustly across benchmarks/TSVs.

        Supported `image_path` conventions:
        - Absolute paths
        - Relative to TSV directory (common in local datasets)
        - Relative to LMUDataRoot() (e.g. "images/xxx.jpg")
        - Relative to LMUDataRoot()/images (e.g. "spacedg_bench/xxx.jpg")
        - Relative to LMUDataRoot()/images/<dataset> (e.g. "xxx.jpg")
        """
        assert "image_path" in line, "SpaceDGBench requires `image_path` in TSV."
        rels = toliststr(line["image_path"])

        lmu = LMUDataRoot()
        tsv_dir = self._tsv_dir
        candidates = []
        if tsv_dir:
            candidates.append(tsv_dir)
        candidates.extend(
            [
                lmu,
                osp.join(lmu, "images"),
                osp.join(lmu, "images", "spacedg_bench"),
                osp.join(lmu, "images", img_root_map(self.dataset_name)),
                self.img_root,
            ]
        )

        resolved = []
        for p in rels:
            p = str(p)
            # Common convention in some TSVs: prefix with "images/" while img_root is already LMUData/images.
            # We normalize it so both styles work:
            #   "images/defocus/..."  -> "defocus/..."
            #   "images\\defocus\\..." -> "defocus/..."
            p_norm = p.replace("\\", "/")
            if p_norm.startswith("images/"):
                p = p_norm[len("images/") :]
            else:
                p = p_norm
            if read_ok(p):
                resolved.append(p)
                continue

            hit = None
            for base in candidates:
                if not base:
                    continue
                ap = osp.join(base, p)
                if read_ok(ap):
                    hit = ap
                    break
            if hit is None:
                raise FileNotFoundError(
                    f"SpaceDGBench image not found. image_path={p}. Tried: "
                    + ", ".join([x for x in candidates if x])
                )
            resolved.append(hit)

        return resolved

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        if "prediction" not in data.columns:
            warnings.warn("SpaceDGBench.evaluate: no prediction column")
            return None

        data = data.sort_values(by="index")
        data["prediction"] = data["prediction"].astype(str)

        def _extract_core_answer(t):
            m = re.search(r"<\s*answer\b[^>]*>(.*?)</\s*answer\s*>", t, flags=re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else t.strip()

        data["prediction"] = data["prediction"].apply(_extract_core_answer)
        data["eval_task"] = data.apply(infer_eval_task, axis=1)

        buckets = {
            "MCQ": data[data["eval_task"] == "MCQ"].copy(),
            "NA": data[data["eval_task"] == "NA"].copy(),
            "YN": data[data["eval_task"] == "YN"].copy(),
            "COUNT": data[data["eval_task"] == "COUNT"].copy(),
            "SIZE_LIST": data[data["eval_task"] == "SIZE_LIST"].copy(),
        }

        score_fn_mcq = build_mcq_score_fn(**judge_kwargs)
        score_fn_na = build_na_score_fn(**judge_kwargs)
        judge_tag = get_judge_tag_from_score_fn(score_fn_mcq)
        result_file, xlsx_path, acc_tsv_path = build_eval_paths(eval_file, judge_tag)

        attach_score_cache(score_fn=score_fn_mcq, eval_file=eval_file, judge_tag=judge_tag, key_col="index", sub_tag="mcq")
        attach_score_cache(score_fn=score_fn_na, eval_file=eval_file, judge_tag=judge_tag, key_col="index", sub_tag="na")

        scored = {}
        scored["mcq"] = score_fn_mcq(buckets["MCQ"]) if len(buckets["MCQ"]) else buckets["MCQ"]
        if len(buckets["NA"]):
            na_work = buckets["NA"].copy()
            na_work["answer"] = na_work["answer"].apply(_parse_gt_numeric)
            scored["na"] = score_fn_na(na_work)
        else:
            scored["na"] = buckets["NA"]
        scored["yn"] = _compute_yn_score(buckets["YN"]) if len(buckets["YN"]) else buckets["YN"]
        scored["count"] = _compute_count_score(buckets["COUNT"]) if len(buckets["COUNT"]) else buckets["COUNT"]
        scored["size_list"] = _compute_size_list_score(buckets["SIZE_LIST"]) if len(buckets["SIZE_LIST"]) else buckets["SIZE_LIST"]

        summary = self._aggregate(scored=scored, total_n=len(data))

        # ---- Save per-question details (json) and degradation summaries (csv) ----
        try:
            frames = []
            for tag, df in scored.items():
                if df is None or len(df) == 0:
                    continue
                dfx = df.copy()
                dfx["answer_type"] = tag
                # Normalize a unified per-row score in [0,1] if possible
                if "hit" in dfx.columns:
                    dfx["row_score"] = dfx["hit"].astype(float)
                elif "MRA:.5:.95:.05" in dfx.columns:
                    dfx["row_score"] = dfx["MRA:.5:.95:.05"].astype(float)
                elif "score" in dfx.columns:
                    dfx["row_score"] = dfx["score"].astype(float)
                else:
                    dfx["row_score"] = np.nan
                frames.append(dfx)

            if frames:
                detailed = pd.concat(frames, ignore_index=True)
            else:
                detailed = pd.DataFrame()

            if len(detailed):
                # Ensure degradation & task columns exist
                if "degradation_type" not in detailed.columns:
                    detailed["degradation_type"] = ""
                if "eval_task" not in detailed.columns:
                    # fallback: try infer from question_type if eval_task missing
                    detailed["eval_task"] = detailed.get("question_type", "").astype(str)

                # 1) JSON: each question + prediction + score
                detail_json_path = get_intermediate_file_path(eval_file, "_details", target_format="json")
                keep_cols = [
                    "index",
                    "question",
                    "answer",
                    "prediction",
                    "row_score",
                    "answer_type",
                    "eval_task",
                    "question_type",
                    "sub_question_type",
                    "task_group",
                    "degradation_type",
                    "image_path",
                    "pred_extracted",
                    "hit",
                    "score",
                ]
                keep_cols = [c for c in keep_cols if c in detailed.columns]
                records = detailed[keep_cols].to_dict(orient="records")
                with open(detail_json_path, "w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)

                # 2) CSV: pivot table degradation_type x eval_task (mean score)
                pivot = pd.pivot_table(
                    detailed,
                    index="degradation_type",
                    columns="eval_task",
                    values="row_score",
                    aggfunc="mean",
                )
                pivot = pivot.sort_index()
                pivot_path = get_intermediate_file_path(eval_file, "_deg_x_task", target_format="csv")
                pivot.to_csv(pivot_path)

                # 2b) CSV: pivot table degradation_type x question_type (mean score)
                # This corresponds to the finer-grained "question_type" taxonomy (often ~11 types).
                if "question_type" in detailed.columns:
                    qt_pivot = pd.pivot_table(
                        detailed,
                        index="degradation_type",
                        columns="question_type",
                        values="row_score",
                        aggfunc="mean",
                    ).sort_index()
                    qt_pivot_path = get_intermediate_file_path(eval_file, "_deg_x_question_type", target_format="csv")
                    qt_pivot.to_csv(qt_pivot_path)

                    # Also dump an overall (all degradations mixed) question_type summary
                    qt_overall = (
                        detailed.groupby("question_type", dropna=False)["row_score"]
                        .agg(["mean", "count"])
                        .reset_index()
                        .rename(columns={"mean": "score", "count": "n"})
                    )
                    qt_overall["score"] = qt_overall["score"].astype(float)
                    qt_overall_path = get_intermediate_file_path(eval_file, "_question_type_overall", target_format="csv")
                    qt_overall.to_csv(qt_overall_path, index=False)

                # 3) CSV: final score per degradation + overall
                by_deg = (
                    detailed.groupby("degradation_type", dropna=False)
                    .agg(n=("row_score", "count"), overall_acc=("row_score", "mean"))
                    .reset_index()
                )
                # add overall row
                overall_row = pd.DataFrame(
                    [{
                        "degradation_type": "OVERALL",
                        "n": int(detailed["row_score"].count()),
                        "overall_acc": float(detailed["row_score"].mean()) if detailed["row_score"].count() else 0.0,
                    }]
                )
                by_deg = pd.concat([by_deg, overall_row], ignore_index=True)
                by_deg_path = get_intermediate_file_path(eval_file, "_deg_overall", target_format="csv")
                by_deg.to_csv(by_deg_path, index=False)
        except Exception as e:
            warnings.warn(f"[SpaceDGBench] failed to save detail json/csv: {e}")

        try:
            with open(result_file, "wb") as f:
                pickle.dump({"scored": scored, "summary": summary}, f)
            print(f"[SpaceDGBench] result saved to {result_file}")
        except OSError as e:
            warnings.warn(f"[SpaceDGBench] pickle save failed: {e}")

        try:
            frames = [df.assign(bucket=tag) for tag, df in scored.items() if len(df)]
            if frames:
                pd.concat(frames, ignore_index=True).to_excel(xlsx_path, index=False, engine="xlsxwriter")
        except Exception as e:
            warnings.warn(f"[SpaceDGBench] xlsx save failed: {e}")

        try:
            if osp.exists(acc_tsv_path):
                os.remove(acc_tsv_path)
        except Exception:
            pass

        return summary

    @staticmethod
    def _aggregate(scored: dict[str, pd.DataFrame], total_n: int) -> dict:
        def _acc(df: pd.DataFrame) -> float:
            if df is None or len(df) == 0:
                return 0.0
            if "hit" in df.columns:
                return float(np.mean(df["hit"].astype(float)))
            if "MRA:.5:.95:.05" in df.columns:
                return float(np.mean(df["MRA:.5:.95:.05"].astype(float)))
            if "score" in df.columns:  # legacy fallback
                return float(np.mean(df["score"].astype(float)))
            return 0.0

        out = OrderedDict()
        out["total_n"] = int(total_n)
        for k in ["mcq", "na", "yn", "count", "size_list"]:
            df = scored.get(k, None)
            out[f"{k}_n"] = int(len(df)) if df is not None else 0
            out[f"{k}_acc"] = _acc(df)

        # Weighted overall
        total_hits = 0.0
        total_cnt = 0
        for k, df in scored.items():
            if df is None or len(df) == 0:
                continue
            if "hit" in df.columns:
                total_hits += float(np.sum(df["hit"].astype(float)))
                total_cnt += int(len(df))
            elif "score" in df.columns:
                total_hits += float(np.sum(df["score"].astype(float)))
                total_cnt += int(len(df))
        out["overall_acc"] = float(total_hits / total_cnt) if total_cnt else 0.0
        return out

