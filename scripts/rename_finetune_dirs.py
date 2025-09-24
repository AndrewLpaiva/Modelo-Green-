"""Rename finetune directories to a simplified, human-friendly pattern.

Usage:
  python .\scripts\rename_finetune_dirs.py --dry-run
  python .\scripts\rename_finetune_dirs.py --apply
  python .\scripts\rename_finetune_dirs.py --dry-run --roots logs modelos_treinados checkpoints

The script scans each child directory in the specified roots and proposes a new
name following this general convention (examples):
  - logs -> `logs-Green-2001-2002-2`
  - modelos_treinados -> `modelo-treinado-Green-2001-2002-2`

It detects 4-digit year tokens in directory names (1900-2099). If it finds two
years it will use them as `from` and `to`. If it finds one year it will produce
`<root>-Green-<year>-1`.

By default the script performs a dry run and writes a JSON preview to
`scripts/rename_finetune_dirs_preview.json`. Use `--apply` to perform the
renames; a log will be written to
`scripts/rename_finetune_dirs_changes.log`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from typing import Dict, List, Tuple


YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def find_years(name: str) -> List[str]:
	return YEAR_RE.findall(name) and [m.group(0) for m in YEAR_RE.finditer(name)] or []


def sanitize_root_label(root: str) -> str:
	# Map directory root -> label used in new names
	if root == "logs":
		return "logs"
	if root == "modelos_treinados":
		return "modelo-treinado"
	if root == "checkpoints":
		return "checkpoints"
	# Fallback: use the directory name itself
	return root


def propose_name(root: str, dirname: str, existing_targets: set, project_token: str = "Green") -> str:
	years = YEAR_RE.findall(dirname)
	# YEAR_RE.findall returns list of '19'/'20' groups due to capture groups: fix by using finditer
	years = [m.group(0) for m in YEAR_RE.finditer(dirname)]

	base_label = sanitize_root_label(root)

	if len(years) >= 2:
		from_y, to_y = years[0], years[1]
		base = f"{base_label}-{project_token}-{from_y}-{to_y}"
	elif len(years) == 1:
		from_y = years[0]
		base = f"{base_label}-{project_token}-{from_y}"
	else:
		# No year found: include a shortened dirname slug (remove problematic chars)
		slug = re.sub(r"[^0-9A-Za-z_-]", "-", dirname)[:40]
		base = f"{base_label}-{project_token}-{slug}"

	# choose smallest integer suffix to avoid collisions
	i = 1
	candidate = f"{base}-{i}"
	while candidate in existing_targets:
		i += 1
		candidate = f"{base}-{i}"
	return candidate


def scan_and_propose(root_dir: str, roots: List[str], project_token: str) -> Dict[str, str]:
	mapping: Dict[str, str] = {}
	for root in roots:
		parent = os.path.join(root_dir, root)
		if not os.path.isdir(parent):
			continue

		children = sorted([d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))])
		existing_targets = set(children)

		for child in children:
			# Skip already-matching names to avoid re-renaming
			if re.match(rf"^{re.escape(sanitize_root_label(root))}-{re.escape(project_token)}-\d{{4}}(?:-\d{{4}})?-\d+$", child):
				# Already looks like the target pattern
				continue

			new_name = propose_name(root, child, existing_targets, project_token)
			# Ensure it's not identical
			if new_name != child and new_name not in existing_targets:
				old_path = os.path.join(parent, child)
				new_path = os.path.join(parent, new_name)
				mapping[old_path] = new_path
				existing_targets.add(new_name)

	return mapping


def perform_renames(mapping: Dict[str, str], dry_run: bool) -> Tuple[int, List[Tuple[str, str]]]:
	performed: List[Tuple[str, str]] = []
	for old, new in mapping.items():
		if dry_run:
			print(f"DRY-RUN: {old} -> {new}")
			performed.append((old, new))
			continue
		# double-check destination doesn't exist
		if os.path.exists(new):
			print(f"SKIP (destination exists): {old} -> {new}")
			continue
		try:
			os.rename(old, new)
			print(f"RENAMED: {old} -> {new}")
			performed.append((old, new))
		except Exception as e:
			print(f"ERROR renaming {old} -> {new}: {e}")
	return len(performed), performed


def main(argv: List[str]) -> int:
	parser = argparse.ArgumentParser(description="Rename finetune directories into a simplified pattern")
	parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=False, help="Show proposed renames but do not apply")
	parser.add_argument("--apply", dest="apply", action="store_true", default=False, help="Apply the proposed renames")
	parser.add_argument("--roots", nargs="*", default=["logs", "modelos_treinados", "checkpoints"], help="Root directories to scan")
	parser.add_argument("--project", default="Green", help="Project token to include in generated names (default: Green)")
	args = parser.parse_args(argv)

	if args.apply and args.dry_run:
		print("Cannot use --apply and --dry-run together. Use either --dry-run or --apply.")
		return 2

	# Default to dry-run unless --apply provided
	dry_run = not args.apply

	repo_root = os.getcwd()

	mapping = scan_and_propose(repo_root, args.roots, args.project)

	preview_path = os.path.join(repo_root, "scripts", "rename_finetune_dirs_preview.json")
	os.makedirs(os.path.dirname(preview_path), exist_ok=True)
	with open(preview_path, "w", encoding="utf-8") as fh:
		json.dump({"mapping": mapping, "dry_run": dry_run}, fh, indent=2, ensure_ascii=False)

	if not mapping:
		print("No directories to rename (no proposals).")
		return 0

	print("Proposed renames:")
	for old, new in mapping.items():
		print(f"  {old} -> {new}")

	if dry_run:
		print(f"\nDry-run mode: preview written to {preview_path}")
		return 0

	print("Applying renames now...")
	count, performed = perform_renames(mapping, dry_run=False)

	log_path = os.path.join(repo_root, "scripts", "rename_finetune_dirs_changes.log")
	with open(log_path, "a", encoding="utf-8") as fh:
		fh.write(f"Renames applied:\n")
		for old, new in performed:
			fh.write(f"{old} -> {new}\n")

	print(f"Applied {count} renames. See log: {log_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))

