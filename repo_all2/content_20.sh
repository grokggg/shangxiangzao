#!/usr/bin/env python3
"""Taiji Analysis Benchmark — Evaluation Frontend.

A lightweight Flask app for browsing, comparing, and annotating benchmark results.

Usage:
    python3 -m nexus.extensions.tasks.taiji_analysis.eval_frontend.app \
        --results-dir /path/to/nexus_results \
        --port 8090

Features:
    - Dashboard: pass rate comparison across models, difficulty distribution
    - Case Browser: filter/search cases, view prediction vs ground truth
    - Multi-Model Compare: side-by-side comparison for a single case
    - Annotation: mark judge errors (false positive / false negative)
    - Hard Case Analysis: focus on all-fail items for judge calibration
"""

import argparse
import json
import re
import sqlite3

from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request


app = Flask(__name__, template_folder='templates', static_folder='static')

RESULTS_DIR: Path = Path('.')
ANNOTATIONS_FILE: Path = Path('annotations.json')

EPISODES_DIR = Path('/apdcephfs_gy6/share_304144884/hunyuan/debug_agent/taiji_replay/episodes')


def get_benchmark_versions() -> list[dict]:
    """Scan EPISODES_DIR for dataset_v*/ folders and build version list."""
    versions = []
    for d in sorted(EPISODES_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith('dataset_v'):
            continue
        parquet = d / 'dataset.parquet'
        dataset_json = d / 'dataset.json'
        if not parquet.exists() and not dataset_json.exists():
            continue
        m = re.match(r'dataset_v(\d+)', d.name)
        version = f'v{m.group(1)}' if m else d.name.replace('dataset_', '')
        if dataset_json.exists():
            with open(dataset_json) as f:
                cases = len(json.load(f))
        else:
            cases = 0
        mtime = datetime.fromtimestamp(d.stat().st_mtime).strftime('%Y-%m-%d')
        versions.append({
            'version': version,
            'date': mtime,
            'cases': cases,
            'description': d.name,
            'dataset': f'episodes/{d.name}/dataset.parquet',
        })
    return versions


EXCLUDE_EXPERIMENTS = {'hy3_results0709'}

ONLINE_GT_PARQUET = Path(
    '/apdcephfs_gy6/share_304144884/hunyuan/debug_agent/taiji_replay/episodes/dataset_v3_0720_218_onlineGT/dataset.parquet'
)
NEWRUN_GT_PARQUET = Path(
    '/apdcephfs_gy6/share_304144884/hunyuan/debug_agent/taiji_replay/episodes/dataset_v3_0720_218_newrunGT/dataset.parquet'
)
NEWRUN_GT_DATASET_JSON = Path(
    '/apdcephfs_gy6/share_304144884/hunyuan/debug_agent/taiji_replay/episodes/dataset_v3_0720_218_newrunGT/dataset.json'
)

_online_gt_cache: dict[str, str] | None = None
_newrun_gt_cache: dict[str, str] | None = None
_case_meta_cache: dict[str, dict] | None = None


def _load_gt_parquet(path: Path) -> dict[str, str]:
    """Load GT answers from parquet keyed by instance_id (== run_id)."""
    result = {}
    if path.exists():
        import pandas as pd

        df = pd.rea