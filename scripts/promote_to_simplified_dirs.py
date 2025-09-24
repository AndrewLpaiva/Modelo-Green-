"""Promove pastas de logs/modelos_treinados com nomes longos para as pastas simplificadas existentes.

Heurística:
 - Para cada long_name em logs/ ou modelos_treinados/, tenta encontrar um simplified dir que contenha tokens do long_name (por exemplo anos '2003' e '2024' ou 'Green').
 - Se encontrado, move a pasta long_name para dentro de simplified_dir como uma nova versão `vNN` (mantendo conteúdo).
 - Operação em dry-run por padrão; use --apply para executar.
"""

import os
import re
import shutil
import argparse

ROOT = os.getcwd()

SIMPLIFIED_PATTERNS = [r"logs-Green-", r"modelo-treinado-Green-"]


def find_simplified_match(longname, simplified_roots):
    lname = longname.lower()
    # extract year tokens
    years = re.findall(r"\b(19|20)\d{2}\b", longname)
    years = [m.group(0) for m in re.finditer(r"\b(19|20)\d{2}\b", longname)]
    for sroot in simplified_roots:
        for child in os.listdir(sroot):
            if not os.path.isdir(os.path.join(sroot, child)):
                continue
            lower = child.lower()
            score = 0
            if 'green' in lower and 'green' in lname:
                score += 1
            for y in years:
                if y in lower:
                    score += 1
            # also check if dataset token (e.g. '2003' or '2024') appears
            if score >= 1:
                return os.path.join(sroot, child)
    return None


def next_version_dir(base_dir):
    existing = [n for n in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, n))]
    vers = [int(n[1:]) for n in existing if re.match(r'^v\d+$', n)]
    next_v = max(vers) + 1 if vers else 1
    return os.path.join(base_dir, f'v{next_v:02d}')


def main(apply=False):
    logs_root = os.path.join(ROOT, 'logs')
    models_root = os.path.join(ROOT, 'modelos_treinados')

    long_logs = [d for d in os.listdir(logs_root) if not any(d.startswith(p) for p in ['logs-Green-'])]
    long_models = [d for d in os.listdir(models_root) if not any(d.startswith(p) for p in ['modelo-treinado-Green-'])]

    mapping = []

    for d in long_logs:
        src = os.path.join(logs_root, d)
        dest_parent = find_simplified_match(d, [logs_root])
        if dest_parent:
            dest_dir = next_version_dir(dest_parent)
            mapping.append((src, dest_dir))

    for d in long_models:
        src = os.path.join(models_root, d)
        dest_parent = find_simplified_match(d, [models_root])
        if dest_parent:
            dest_dir = next_version_dir(dest_parent)
            mapping.append((src, dest_dir))

    if not mapping:
        print('No candidates to move')
        return

    print('Proposed moves:')
    for s, t in mapping:
        print(f'  {s} -> {t}')

    if not apply:
        print('\nDry-run (no changes). Use --apply to perform moves.')
        return

    for s, t in mapping:
        try:
            os.makedirs(os.path.dirname(t), exist_ok=True)
            shutil.move(s, t)
            print(f'Moved {s} -> {t}')
        except Exception as e:
            print(f'Failed to move {s} -> {t}: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    main(apply=args.apply)
