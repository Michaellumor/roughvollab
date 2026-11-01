"""Build docs/p5/index.html, the interactive tour of P5: read the committed outputs, check them and fill template.html.

Writes only that page and imports nothing from the repository. No value of m is carried into the page.

    py -3 -I -X utf8 analysis/p5_tour/build.py
"""
import csv, json, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, 'output')
SCALE = 10 ** 6


def i6(x):
    return int(round(float(x) * SCALE))


# the page's tag names the commit it was built from, so a rebuild at a later commit changes only that tag
commit = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
assert not subprocess.run(['git', '-C', REPO, 'status', '--porcelain', '--', 'output'], capture_output=True, text=True, check=True).stdout.strip(), 'output/ is not clean'
rep = json.load(open(os.path.join(OUT, 'p5_stage2_report.json'), encoding='utf-8'))
g3 = json.load(open(os.path.join(OUT, 'p5_g3_report.json'), encoding='utf-8'))

s1 = sorted(csv.DictReader(open(os.path.join(OUT, 'p5_stage1.csv'), encoding='utf-8', newline='')), key=lambda r: (int(r['i']), int(r['j'])))
assert len(s1) == 1200
BIN = {'Identified': 0, 'Partial': 1, 'Flat': 2}
refs = {}
for line in open(os.path.join(OUT, 'p5v3_ref_test.jsonl'), encoding='utf-8'):
    o = json.loads(line)
    if o.get('kind') == 'task':
        refs[o['task']] = o
assert len(refs) == 1200

theta = [None] * 400
pairs = {'c': [], 'bin': [], 'wc': [], 'exA': [], 'exB': [], 'q05': [], 'q95': [], 'covA': [], 'covB': []}
for p, r in enumerate(s1):
    i, j = int(r['i']), int(r['j'])
    assert p == i * 3 + (j - 1), (p, i, j)
    rec = refs[r['pair']]
    th = [i6(r['H']), i6(r['nu']), i6(r['rho']), i6(r['xi0'])]
    assert theta[i] in (None, th)
    theta[i] = th
    assert abs(float(rec['c_A']) - float(r['c_A'])) == 0
    pairs['c'].append(i6(r['c_A']))
    pairs['bin'].append(BIN[r['bin']])
    pairs['wc'].append(int(r['well_covered']))
    pairs['exA'].append(i6(rec['under']['A']['mean_H']))
    pairs['exB'].append(i6(rec['under']['B']['mean_H']))
    pairs['q05'].append(i6(rec['under']['A']['q05_H']))
    pairs['q95'].append(i6(rec['under']['A']['q95_H']))
    pairs['covA'].append(i6(r['cover_A']))
    pairs['covB'].append(i6(r['cover_B']))

cells = list(csv.DictReader(open(os.path.join(OUT, 'p5_stage2_cells.csv'), encoding='utf-8', newline='')))
assert len(cells) == 12000
nets = {'ensemble': {'A': [None] * 6000, 'B': [None] * 6000}, 'nre': {'A': [None] * 6000, 'B': [None] * 6000}}
labels = [r['pair'] for r in s1]
index = {lab: p for p, lab in enumerate(labels)}
filled = set()
for r in cells:
    p, q = index[r['pair']], int(r['q']) - 1
    assert r['failure'] == ''
    key = (r['calibrator'], p, q)
    assert key not in filled
    filled.add(key)
    nets[r['calibrator']]['A'][p * 5 + q] = i6(r['H_hat_A'])
    # null marks "no network here": an estimate can be negative, since nothing is clipped to the box (line 75)
    nets[r['calibrator']]['B'][p * 5 + q] = None if r['H_hat_B'] == '' else i6(r['H_hat_B'])
assert len(filled) == 12000
assert all(v is not None for c in nets.values() for v in c['A'])
assert all(v is not None for v in nets['ensemble']['B'])
assert all((nets['nre']['B'][p * 5 + q] is None) == (p % 3 < 2) for p in range(1200) for q in range(5))
assert sum(v < 0 for v in nets['ensemble']['B']) == 12 and not any(v < 0 for v in nets['ensemble']['A'])


def row(x):
    out = {k: x.get(k) for k in ('verdict', 'point', 'interval', 'per_seed', 'n_pairs', 'n_cells', 'seed_floor', 'levels_pooled', 'untestable_reason')}
    out['per_level'] = {s: ({'median': v.get('median'), 'n_pairs': v.get('n_pairs')} if isinstance(v, dict) and 'median' in v else None)
                        for s, v in (x.get('per_level') or {}).items()}
    return out


def analysis(sec_stage1, sec_cal):
    return {'bins': {b: {'count': v['count'], 'per_level': v['per_level'], 'n_well_covered': v.get('n_well_covered'), 'untestable': v['untestable']} for b, v in sec_stage1['bins'].items()},
            'n': sec_stage1['n_accepted'], 'cuts': sec_stage1['cuts'],
            'cal': {c: {'hyp': {h: row(x) for h, x in sec_cal[c]['hypotheses'].items()},
                        'floor': {b: {'floor': v['floor'], 'A': v['per_prior']['A'], 'B': v['per_prior']['B']} for b, v in sec_cal[c]['seed_floor'].items()}} for c in sec_cal}}


report = {'primary': analysis(rep['stage1'], rep['stage2']),
          'plus': analysis(rep['re_scoring']['+0.05']['stage1'], rep['re_scoring']['+0.05']['calibrators']),
          'minus': analysis(rep['re_scoring']['-0.05']['stage1'], rep['re_scoring']['-0.05']['calibrators']),
          'secondary': analysis(rep['secondary']['stage1'], rep['secondary']['calibrators']),
          'dose': {'primary': {c: {'rho': rep['dose_response'][c]['spearman'], 'iv': rep['dose_response'][c]['interval']} for c in ('ensemble', 'nre')},
                   'secondary': {c: {'rho': rep['secondary']['dose_response'][c]['spearman'], 'iv': rep['secondary']['dose_response'][c]['interval']} for c in ('ensemble', 'nre')}},
          'near': rep['stage1']['near_cutoff']}

ladder = {}
for k, c in g3['cells'].items():
    ladder[k] = {'cal': c['calibrator'], 'prior': c['prior'], 'j': c['j'], 's': c['s'], 'n_star': c['n_star'], 'passed': c['passed'],
                 'rungs': [{'n': r['n'], 'slope': r['slope'], 'slope_ok': r['slope_ok'], 'med': r['median_abs_d_n'], 'med2': r['median_abs_d_2n'], 'stable': r['stable'], 'passed': r['passed']} for r in c['rungs']]}
assert len(ladder) == 12 and g3['stability_tol'] == 0.002 and g3['slope_min'] == 0.95
assert g3['scoring_projection']['projected_core_hours'] > 155 and g3['scoring_projection']['budget_core_hours'] == 1344

data = {'scale': SCALE, 'commit': commit, 'theta': theta, 'pairs': pairs, 'nets': nets, 'report': report, 'ladder': ladder}
text = json.dumps(data, separators=(',', ':'), ensure_ascii=True)
assert '"m"' not in text, 'm must not be carried'


# the page's figures, checked against the report from the scaled numbers it carries
def med_low(v):
    v = sorted(v)
    return v[(len(v) - 1) // 2]


for cal, hyp, measure in (('ensemble', 'H-a', 'abs'), ('nre', 'H-a', 'abs'), ('ensemble', 'H-b', 'E'), ('nre', 'H-b', 'E'), ('ensemble', 'H-c', 'E'), ('nre', 'H-c', 'E')):
    row_ = report['primary']['cal'][cal]['hyp'][hyp]
    b = {'H-a': 0, 'H-b': 1, 'H-c': 0}[hyp]
    vals = []
    for p in range(1200):
        j = p % 3
        if pairs['bin'][p] != b or (cal == 'nre' and measure == 'E' and j < 2):
            continue
        for q in range(5):
            a = nets[cal]['A'][p * 5 + q] / SCALE
            ex = pairs['exA'][p] / SCALE
            if measure == 'abs':
                vals.append(abs(a - ex))
            else:
                bb = nets[cal]['B'][p * 5 + q] / SCALE
                vals.append(((bb - a) - (pairs['exB'][p] - pairs['exA'][p]) / SCALE) / 0.20)
    assert len(vals) == row_['n_cells'], (cal, hyp, len(vals))
    assert abs(med_low(vals) - row_['point']) < 2e-5, (cal, hyp, med_low(vals), row_['point'])
print('six scored primary medians reproduced from the page\'s numbers, within 2e-5')

tpl = open(os.path.join(HERE, 'template.html'), encoding='utf-8').read()
assert tpl.count('/*__DATA__*/null') == 1
html = tpl.replace('/*__DATA__*/null', text)
dst = os.path.join(REPO, 'docs', 'p5', 'index.html')
os.makedirs(os.path.dirname(dst), exist_ok=True)
open(dst, 'w', encoding='utf-8', newline='\n').write(html)
print(f'written {os.path.relpath(dst, REPO)}: {len(html.encode("utf-8")):,} bytes, data {len(text):,}; commit {commit}')
