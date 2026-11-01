"""Walk docs/p5/index.html in headless Chrome and compare what it shows with the committed outputs.

Every recorded surface goes through both explorers: script errors, S, E and |d| against p5_stage2_cells.csv, the map's counts
against the report, marks off either explorer's plot, the off-scale key against the markers, and clipped labels on each view. Then
each view at 400 and 320 px, for overflow. Reads the repository and writes only in a temporary folder. Set CHROME if Chrome is elsewhere.

    py -3 -I -X utf8 analysis/p5_tour/check.py

Exit status 1 if anything disagrees.
"""
import base64, csv, json, os, pathlib, re, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, 'output')
PAGE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'docs', 'p5', 'index.html')
CHROME = os.environ.get('CHROME', r'C:\Program Files\Google\Chrome\Application\chrome.exe')

WALK = r"""<script>
(function(){
 const out={errors:[],lean:[],fid:[],map:[],bands:[],clipped:[],keys:{lean:{left:0,right:0,both:0,bad:0},fid:{left:0,right:0,both:0,bad:0}}};
 const shapes=(svg,lo,hi)=>{let bad=0;svg.querySelectorAll('circle,polygon').forEach(e=>{const xs=e.tagName==='circle'?(e.hasAttribute('cx')?[+e.getAttribute('cx')]:[]):e.getAttribute('points').trim().split(/\s+/).map(q=>+q.split(',')[0]);for(const x of xs)if(x<lo||x>hi)bad++});return bad};
 const dirs=(svg,id)=>{const tri=[...svg.querySelectorAll('polygon')].map(e=>e.getAttribute('points').trim().split(/\s+/).map(q=>+q.split(',')[0])).filter(xs=>xs.length===3),lo=tri.some(xs=>xs[0]<xs[1]),hi=tri.some(xs=>xs[0]>xs[1]),k=document.querySelector(id),l=k.querySelector('.l'),r=k.querySelector('.r');return{lo,hi,bad:(k.hidden===(lo||hi))||(lo&&l.hidden)||(!lo&&!l.hidden)||(hi&&r.hidden)||(!hi&&!r.hidden)}};
 const tally=(t,d)=>{if(d.lo&&d.hi)t.both++;else if(d.lo)t.left++;else if(d.hi)t.right++;if(d.bad)t.bad++};
 const clip=(tag)=>{document.querySelectorAll('svg').forEach(svg=>{const sr=svg.getBoundingClientRect();if(!sr.width)return;svg.querySelectorAll('text').forEach(t=>{const r=t.getBoundingClientRect();if(r.width&&(r.left<sr.left-.5||r.right>sr.right+.5))out.clipped.push([tag,t.textContent.slice(0,40)])})})};
 go('lean',true);
 for(let lev=0;lev<3;lev++)for(let idx=0;idx<400;idx++){L.level=lev;L.idx=idx;const p=lineup(lev,'c')[idx];
  try{renderLean()}catch(e){out.errors.push(['lean',p,String(e)]);continue}
  const ro=[...document.querySelectorAll('#lRo .rv')].map(e=>e.textContent),svg=document.querySelector('#lPlot svg'),w=svg.viewBox.baseVal.width,txt=[...svg.querySelectorAll('text')].filter(t=>t.textContent.includes('G3 cell failed')).length,v=document.querySelector('#lV .vt').textContent;
  out.lean.push([p,ro[0],ro[1],ro[2],txt,shapes(svg,104,w-6),v.includes('outside H')?1:0]);
  tally(out.keys.lean,dirs(svg,'#lOffKey'));
  if(idx%50===0)clip('lean')}
 go('fidelity',true);F.order='c';F.zoom=false;
 for(let lev=0;lev<3;lev++)for(let idx=0;idx<400;idx++){F.level=lev;F.idx=idx;const p=lineup(lev,'c')[idx];
  try{renderFid()}catch(e){out.errors.push(['fid',p,String(e)]);continue}
  const ro=[...document.querySelectorAll('#fRo .rv')].map(e=>e.textContent),svg=document.querySelector('#fPlot svg'),w=svg.viewBox.baseVal.width;
  out.fid.push([p,ro[2],ro[3],shapes(svg,104,w-6),document.querySelector('#fV .vt').textContent.includes('outside H')?1:0]);
  tally(out.keys.fid,dirs(svg,'#fOffKey'));
  if(idx%50===0)clip('fid')}
 go('map',true);
 for(const cut of ['primary','plus','minus'])for(let lev=0;lev<3;lev++){M.cut=cut;M.level=lev;try{renderMap();out.map.push([cut,lev,[...document.querySelectorAll('#mapRo .rv')].map(e=>+e.textContent)])}catch(e){out.errors.push(['map',cut,lev,String(e)])}}
 go('lean',true);
 for(const [cal,grps] of [['ensemble',['all',0,1,2]],['nre',[2]]])for(const g of grps){L.cal=cal;L.grp=g;try{leanScatter();out.bands.push([cal,String(g),document.querySelectorAll('#lScatter svg circle[cx]').length,document.querySelector('#lBand').hidden?0:1])}catch(e){out.errors.push(['scatter',cal,g,String(e)])}}
 for(const v of ['hub','journey','map','intervals','training','verdicts']){try{go(v,true);document.querySelectorAll('.advanced .adv-body').forEach(b=>{b.hidden=false});if(RENDER[v])RENDER[v]();clip(v)}catch(e){out.errors.push(['view',v,String(e)])}}
 const pre=document.createElement('pre');pre.id='walkout';pre.textContent=btoa(unescape(encodeURIComponent(JSON.stringify(out))));document.body.appendChild(pre);
})();
</script>"""

PHONE = r"""<script>addEventListener('error',e=>console.error('PAGEERROR '+e.message+' line '+e.lineno));
setTimeout(()=>{document.querySelectorAll('.advanced .fold-head').forEach(b=>b.click())},200);
setTimeout(()=>{const v=location.hash,sw=document.documentElement.scrollWidth,iw=innerWidth;const wide=[...document.querySelectorAll('body *')].filter(e=>{if(e.closest('.tablewrap'))return false;const r=e.getBoundingClientRect();return r.width>0&&r.right>iw+1}).slice(0,4).map(e=>e.tagName+'.'+(e.getAttribute('class')||e.id||''));
let cut=0;document.querySelectorAll('svg').forEach(svg=>{const sr=svg.getBoundingClientRect();if(!sr.width)return;svg.querySelectorAll('text').forEach(t=>{const r=t.getBoundingClientRect();if(r.width&&(r.left<sr.left-.5||r.right>sr.right+.5))cut++})});
console.error('PROBE '+v+' scrollWidth '+sw+' of '+iw+' wide '+wide.length+' clipped '+cut)},900)</script>"""

html = open(PAGE, encoding='utf-8').read()
assert html.count('</body>') == 1
problems = []
if '"m":' in html:
    problems.append('a value of m is carried')

tmp = tempfile.mkdtemp(prefix='p5check')
try:
    walk = os.path.join(tmp, 'walk.html')
    open(walk, 'w', encoding='utf-8').write(html.replace('</body>', WALK + '</body>'))
    r = subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--hide-scrollbars', '--user-data-dir=' + os.path.join(tmp, 'prof'), '--window-size=1280,2300',
                        '--virtual-time-budget=20000', '--dump-dom', pathlib.Path(walk).as_uri()], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
    m = re.search(r'<pre id="walkout">([^<]+)</pre>', r.stdout)
    if not m:
        print('the walk produced no output', len(r.stdout))
        sys.exit(1)
    res = json.loads(base64.b64decode(m.group(1)).decode('utf-8'))

    phone = []
    inner = os.path.join(tmp, 'p.html')
    open(inner, 'w', encoding='utf-8').write(html.replace('</body>', PHONE + '</body>'))
    for width in (400, 320):
        for v in ['hub', 'journey', 'map', 'fidelity', 'lean', 'intervals', 'training', 'verdicts']:
            outer = os.path.join(tmp, f'o_{v}.html')
            open(outer, 'w', encoding='utf-8').write(f'<!doctype html><body style="margin:0"><iframe src="p.html#{v}" width="{width}" height="3400" style="border:0"></iframe></body>')
            q = subprocess.run([CHROME, '--headless=new', '--disable-gpu', '--hide-scrollbars', '--allow-file-access-from-files', '--user-data-dir=' + os.path.join(tmp, 'prof2'), '--window-size=520,3400',
                                '--virtual-time-budget=4000', '--enable-logging=stderr', '--v=0', '--screenshot=' + os.path.join(tmp, v + '.png'), pathlib.Path(outer).as_uri()], capture_output=True, text=True, errors='replace', timeout=120)
            phone.append((width, v, re.findall(r'(PROBE [^"]*|PAGEERROR [^"]*|Uncaught [^"]*)', q.stderr)))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

num = lambda s: float(s.replace('−', '-'))
low = lambda a: sorted(a)[(len(a) - 1) // 2]
rep = json.load(open(os.path.join(OUT, 'p5_stage2_report.json'), encoding='utf-8'))
s1 = sorted(csv.DictReader(open(os.path.join(OUT, 'p5_stage1.csv'), encoding='utf-8', newline='')), key=lambda r: (int(r['i']), int(r['j'])))
index = {r['pair']: p for p, r in enumerate(s1)}
E, D, Sx = {'ensemble': {}, 'nre': {}}, {'ensemble': {}, 'nre': {}}, {}
for c in csv.DictReader(open(os.path.join(OUT, 'p5_stage2_cells.csv'), encoding='utf-8', newline='')):
    p = index[c['pair']]
    D[c['calibrator']].setdefault(p, []).append(float(c['abs_dA']))
    if c['E'] != '':
        E[c['calibrator']].setdefault(p, []).append(float(c['E']))
    if c['S_exact'] != '':
        Sx[p] = float(c['S_exact'])

if res['errors']:
    problems.append(f"script errors: {res['errors'][:6]}")
bad = dict(S=0, E=0, N=0, fail=0, frame=0)
for p, s, e, n, fails, off, note in res['lean']:
    bad['S'] += abs(num(s) - Sx[p]) > .00052
    bad['E'] += abs(num(e) - low(E['ensemble'][p])) > .00052
    bad['N'] += (abs(num(n) - low(E['nre'][p])) > .00052) if p % 3 == 2 else (n != 'none')
    bad['fail'] += fails != (0 if p % 3 == 2 else 1)
    bad['frame'] += off
lean_notes = sum(x[6] for x in res['lean'])
print(f"lean: {len(res['lean'])} surfaces; wrong S {bad['S']}, wrong ensemble E {bad['E']}, wrong ratio-estimator E {bad['N']}, wrong 'G3 cell failed' {bad['fail']}, marks off the frame {bad['frame']}; off-scale note on {lean_notes}")
if len(res['lean']) != 1200 or any(bad.values()):
    problems.append('the lean explorer disagrees')
bad = dict(E=0, N=0, frame=0)
for p, de, dn, off, note in res['fid']:
    bad['E'] += abs(num(de) - low(D['ensemble'][p])) > .000052
    bad['N'] += abs(num(dn) - low(D['nre'][p])) > .000052
    bad['frame'] += off
fid_notes = sum(x[4] for x in res['fid'])
print(f"fidelity: {len(res['fid'])} surfaces; wrong ensemble |d| {bad['E']}, wrong ratio-estimator |d| {bad['N']}, marks off the frame {bad['frame']}; off-scale note on {fid_notes}")
if len(res['fid']) != 1200 or any(bad.values()):
    problems.append('the fidelity explorer disagrees')
LEV = ['0.001', '0.003', '0.005']
SRC = {'primary': rep['stage1'], 'plus': rep['re_scoring']['+0.05']['stage1'], 'minus': rep['re_scoring']['-0.05']['stage1']}
wrong = 0
for cut, lev, shown in res['map']:
    want = [SRC[cut]['bins'][b]['per_level'][LEV[lev]] for b in ('Identified', 'Partial', 'Flat')]
    wrong += [int(x) for x in shown] != [int(x) for x in want]
print(f"map: {len(res['map'])} views against the report; wrong {wrong}")
if len(res['map']) != 9 or wrong:
    problems.append('the map disagrees with the report')
want = {('ensemble', 'all'): (1200, 0), ('ensemble', '0'): (400, 0), ('ensemble', '1'): (400, 0), ('ensemble', '2'): (400, 0), ('nre', '2'): (400, 1)}
got = {(c, g): (n, b) for c, g, n, b in res['bands']}
print('scatter (dots, recorded bands shown):', got)
if got != want:
    problems.append('the scatter disagrees')
print('off-scale key against the markers:', res['keys'])
if res['keys']['lean']['bad'] or res['keys']['fid']['bad']:
    problems.append('the off-scale key disagrees with the markers')
print('clipped labels:', len(res['clipped']), sorted({tuple(x) for x in res['clipped']})[:8])
if res['clipped']:
    problems.append('clipped labels')
for width, v, lines in phone:
    ok = len(lines) == 1 and re.fullmatch(rf'PROBE #\S+ scrollWidth {width} of {width} wide 0 clipped 0', lines[0]) is not None
    print('phone', width, v, lines[:2])
    if not ok:
        problems.append(f'phone width {width}: {v}')
if problems:
    print('PROBLEMS:', problems)
    sys.exit(1)
print('all checks pass')
