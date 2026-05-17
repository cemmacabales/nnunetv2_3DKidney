import os
import csv
import statistics
from glob import glob

BASE_DIR = "/Users/cemmacabales/Downloads/THESIS FINAL/hd95_results"

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def compute_stats(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return {}
    return {
        "count": len(clean),
        "mean": round(statistics.mean(clean), 2),
        "median": round(statistics.median(clean), 2),
        "std": round(statistics.stdev(clean), 2) if len(clean) > 1 else 0.0,
        "min": round(min(clean), 2),
        "max": round(max(clean), 2),
    }

def summarize_csv(filepath):
    dataset_name = os.path.basename(os.path.dirname(filepath))
    rows = []
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Identify metric columns
    headers = list(rows[0].keys())
    label_cols = sorted(set([h.split('_')[0] + '_' + h.split('_')[1] for h in headers if h.startswith('label_')]))
    # e.g., ['label_1', 'label_2', ...]

    summary = {}
    for label in label_cols:
        hd95_key = f"{label}_hd95"
        esd_key = f"{label}_esd_mm"
        den_key = f"{label}_density_hu"

        hd95_vals = []
        esd_vals = []
        den_vals = []

        hd95_pos = []
        esd_pos = []
        den_pos = []

        hd95_neg = []
        esd_neg = []
        den_neg = []

        for row in rows:
            case = row.get('case', '')
            is_neg = 'neg_' in case
            is_pos = not is_neg

            h = safe_float(row.get(hd95_key, ''))
            e = safe_float(row.get(esd_key, ''))
            d = safe_float(row.get(den_key, ''))

            # Only include if not None
            if h is not None:
                hd95_vals.append(h)
                if is_pos: hd95_pos.append(h)
                else: hd95_neg.append(h)
            if e is not None:
                esd_vals.append(e)
                if is_pos: esd_pos.append(e)
                else: esd_neg.append(e)
            if d is not None:
                den_vals.append(d)
                if is_pos: den_pos.append(d)
                else: den_neg.append(d)

        summary[label] = {
            'all': {
                'hd95': compute_stats(hd95_vals),
                'diameter_mm': compute_stats(esd_vals),
                'density_hu': compute_stats(den_vals),
            },
            'positive': {
                'hd95': compute_stats(hd95_pos),
                'diameter_mm': compute_stats(esd_pos),
                'density_hu': compute_stats(den_pos),
            },
            'negative': {
                'hd95': compute_stats(hd95_neg),
                'diameter_mm': compute_stats(esd_neg),
                'density_hu': compute_stats(den_neg),
            }
        }
    return dataset_name, summary, rows

def main():
    csv_files = glob(os.path.join(BASE_DIR, "*/*.csv"))
    results = []
    for csv_file in csv_files:
        dataset_name, summary, rows = summarize_csv(csv_file)
        results.append((dataset_name, summary, rows))

    # Generate Markdown
    md_lines = []
    md_lines.append("# HD95, Diameter, and Density Results Summary\n")
    md_lines.append("This document summarizes the evaluation metrics across all datasets and folds.\n")
    md_lines.append("Metrics are extracted from the `hd95_results` folder.\n")
    md_lines.append("For each dataset and label, statistics are computed for **All** cases, **Positive** cases (non-`neg_`), and **Negative** cases (`neg_`).\n")
    md_lines.append("**Note:** HD95 (Hausdorff Distance 95th percentile), Diameter (ESD in mm), and Density (HU) are reported. Empty values are excluded from statistics.\n")

    for dataset_name, summary, rows in results:
        md_lines.append(f"\n## Dataset: {dataset_name}\n")
        md_lines.append(f"Total cases: {len(rows)}\n")

        # Count folds
        folds = sorted(set([r.get('fold', '') for r in rows if r.get('fold')]))
        md_lines.append(f"Folds: {', '.join(folds)}\n")

        for label in sorted(summary.keys()):
            md_lines.append(f"\n### {label.replace('_', ' ').title()}\n")
            for group in ['all', 'positive', 'negative']:
                stats = summary[label][group]
                if any(stats[k].get('count', 0) > 0 for k in stats):
                    md_lines.append(f"\n#### {group.title()} Cases\n")
                    md_lines.append("| Metric | Count | Mean | Median | Std | Min | Max |")
                    md_lines.append("|--------|-------|------|--------|-----|-----|-----|")
                    for metric_name in ['hd95', 'diameter_mm', 'density_hu']:
                        s = stats[metric_name]
                        if s.get('count', 0) > 0:
                            md_lines.append(f"| {metric_name.replace('_', ' ').upper()} | {s['count']} | {s['mean']} | {s['median']} | {s['std']} | {s['min']} | {s['max']} |")
                        else:
                            md_lines.append(f"| {metric_name.replace('_', ' ').upper()} | - | - | - | - | - | - |")

    output_path = os.path.join(BASE_DIR, "compiled_summary.md")
    with open(output_path, 'w') as f:
        f.write('\n'.join(md_lines))

    print(f"Summary written to: {output_path}")

if __name__ == "__main__":
    main()
