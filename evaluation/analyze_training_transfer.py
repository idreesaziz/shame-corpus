"""Paired training-fit and confidence changes, keeping missing outputs explicit."""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def mean(xs):
    return statistics.mean(xs) if xs else None


def canonical(value):
    if value is None: return None
    value = value.strip(' *`.$').upper()
    value = re.sub(r'^(?:THE FINAL PRICE IS|THE ANSWER IS|IT IS)\s+', '', value)
    # Accept a numerical field with an optional unit; never mine arbitrary reasoning.
    match = re.fullmatch(r'(-?\d+(?:\.\d+)?)(?:\s+(?:MINUTES?|CREDITS?|BOLTS?|UNITS?|MEMBERS?|SQUARE METRES?|KILOMETRES? PER HOUR|M2|KM/H))?', value)
    if match: return format(float(match[1]), 'g')
    if value in ('YES', 'NO', 'UNKNOWN'): return value
    return value


def read_fields(text):
    # Some models put both requested fields on one line. Preserve that valid
    # value instead of marking e.g. "73 and Confidence: 100%" as a wrong answer.
    matches = re.findall(r'(?im)\bAnswer\**\s*:\s*\**([^\n]+)', text)
    value = matches[-1] if matches else None
    if value is not None:
        value = re.split(r'(?i)\bConfidence\**\s*:', value)[0]
        value = re.sub(r'(?i)\s+and\s*$', '', value).strip(' *`.,;')
    ratings = list(re.finditer(r'(?i)\bConfidence\**\s*:\s*\**(\d+(?:\.\d+)?)', text))
    rating = None
    if ratings:
        last = ratings[-1]; tail = text[last.end():]
        if not re.match(r'\s*(?:-|to\b|or\b|/)', tail, re.I):
            rating_value = float(last[1])
            if 0 <= rating_value <= 100: rating = rating_value
    return canonical(value), rating


def analyze(path):
    source = json.loads(path.read_text(encoding='utf8'))
    base = next(v for v in source['results'] if v['name'] == 'untouched')
    baseline = {(r['id'], r['appraisal']): r['mean_log_probability'] for r in base['likelihoods']}
    output = []
    for variant in source['results']:
        fit = defaultdict(list)
        by_document = defaultdict(dict)
        for row in variant['likelihoods']:
            delta = row['mean_log_probability']-baseline[(row['id'], row['appraisal'])]
            fit[(row['split'], row['appraisal'])].append(delta)
            by_document[(row['split'], row['id'])][row['appraisal']] = delta
        selectivity = defaultdict(list)
        if variant['framing'] != 'untouched':
            for (split, _), cells in by_document.items():
                own = cells[variant['framing']]
                selectivity[split].append(own-statistics.mean(v for k,v in cells.items() if k != variant['framing']))
        cells = defaultdict(list); paired = defaultdict(dict)
        for row in variant['rows']:
            if row['kind'] == 'profile': continue
            answer, rating = read_fields(row['text'])
            if row['id'].startswith('date/') and answer is not None:
                date = re.fullmatch(r'(\d+) MARCH', answer)
                if date: answer = date[1]
            accepted = {row['key']}
            ambiguous = row['id'].split('/')[0] == 'unknown-cause'
            if ambiguous: accepted.add('NO')
            record = {'id': row['id'], 'answer': answer, 'correct': answer in accepted,
                      'key':row['key'], 'kind':row['kind'],
                      'readable': answer is not None, 'confidence': rating,
                      'hit_cap': row['hit_cap'], 'ambiguous_item': ambiguous}
            cells[row['history']].append(record)
            paired[row['id'].split('/')[0]][row['history']] = record
        behaviors = {}
        for history, records in cells.items():
            selected = [r for r in records if not r['ambiguous_item']]
            valid = [r for r in selected if r['readable'] and r['confidence'] is not None]
            answerable = [r for r in selected if r['key']!='UNKNOWN']
            behaviors[history] = {'n': len(selected), 'correct': sum(r['correct'] for r in selected),
                'missing_answer': sum(not r['readable'] for r in selected),
                'missing_confidence': sum(r['confidence'] is None for r in selected),
                'hit_cap': sum(r['hit_cap'] for r in selected),
                'unknown_on_answerable': sum(r['answer']=='UNKNOWN' and r['key']!='UNKNOWN' for r in selected),
                'answerable_n':len(answerable),
                'answerable_correct':sum(r['correct'] for r in answerable),
                'answerable_confidence_mean':mean([r['confidence'] for r in answerable if r['confidence'] is not None]),
                'confidence_mean': mean([r['confidence'] for r in selected if r['confidence'] is not None]),
                'confidence_correct_mean': mean([r['confidence'] for r in valid if r['correct']]),
                'confidence_incorrect_mean': mean([r['confidence'] for r in valid if not r['correct']]),
                'brier_n': len(valid), 'brier': mean([(r['confidence']/100-int(r['correct']))**2 for r in valid]),
                'records': records}
        contrasts = {}
        for a,b in [('self_wrong','none'),('self_wrong','other_wrong'),('self_wrong','self_correct')]:
            selected = [c for item,c in paired.items() if item!='unknown-cause' and
                        c[a]['confidence'] is not None and c[b]['confidence'] is not None]
            contrasts[a+' minus '+b] = {'n':len(selected),
                'mean_confidence_gap_pp':mean([c[a]['confidence']-c[b]['confidence'] for c in selected])}
        output.append({'name':variant['name'],'framing':variant['framing'],'seed':variant['seed'],
            'appraisal_gain_nats_per_token':{split:{app:mean(vs) for (s,app),vs in fit.items() if s==split}
                                            for split in ['trained_document','fresh_document']},
            'own_appraisal_selectivity':{split:mean(vs) for split,vs in selectivity.items()},
            'behavior':behaviors,'paired_confidence':contrasts})
    return {'model':source['model'],'source':str(path.resolve()),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'notes':'Primary behavioral counts exclude ambiguous unknown-cause item. Missing/unreadable responses remain separate. '
                    'Likelihood gains compare identical suffixes across adapters, not different-length candidates as probabilities. '
                    'Seed and prompt repeats are not independent tasks.', 'variants':output}


def readout_analysis(path, main):
    source=json.loads(path.read_text(encoding='utf8'));output=[]
    for v in source['results']:
        old=next(x for x in main['variants'] if x['name']==v['name'])
        declines={r['id'].split('/')[0] for r in old['behavior']['none']['records']
                  if r['key']!='UNKNOWN' and r['answer']=='UNKNOWN'}
        cells=defaultdict(list);recovered=[]
        for r in v['rows']:
            ans,_=read_fields(r['text'])
            if ans is None:ans=canonical(r['text'])
            if r['id'].startswith('date/') and ans:ans=re.sub(r'^(\d+) MARCH$',r'\1',ans)
            explicit=ans is not None and re.fullmatch(r'-?\d+(?:\.\d+)?|UNKNOWN|YES|NO',ans) is not None
            record={'id':r['id'],'text':r['text'],'answer':ans,'explicit_single_value':explicit,
                    'hit_cap':r['hit_cap'],'strict_correct':explicit and ans==r['key'] and not r['hit_cap']}
            cells[r['history']].append(record)
            if r['history']=='answer_only' and r['id'].split('/')[0] in declines and record['strict_correct']:
                recovered.append(r['id'].split('/')[0])
        output.append({'name':v['name'],'framing':v['framing'],'seed':v['seed'],
            'previous_declines_on_answerable':len(declines),'recovered_with_explicit_correct_answer_only':recovered,
            'formats':{h:{'n':len(rows),'strict_correct':sum(r['strict_correct'] for r in rows),
                          'explicit_unknown':sum(r['answer']=='UNKNOWN' and not r['hit_cap'] for r in rows),
                          'unparsed_or_prose':sum(not r['explicit_single_value'] for r in rows),
                          'hit_cap':sum(r['hit_cap'] for r in rows),'rows':rows} for h,rows in cells.items()}})
    return {'model':source['model'],'source':str(path.resolve()),
            'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope':'Conservative recovery lower bound: explicit single-value nontruncated replies only. '
                    'Unparsed prose is not scored as wrong. The UNKNOWN-only arm often exceeds its 64-token cap '
                    'and does not support a clean aggregate accuracy comparison. Prompt variants change several '
                    'instructions, so this does not isolate offering abstention from requesting confidence.',
            'variants':output}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    parser.add_argument('--readout',type=Path);args=parser.parse_args()
    reports=[analyze(p) for key in ['qwen','gemma'] if (p:=args.directory/(key+'.json')).exists()]
    (args.directory/'analysis.json').write_text(json.dumps(reports,indent=2)+'\n',encoding='utf8')
    if args.readout:
        additional=[readout_analysis(p,main) for main in reports if (p:=args.readout/(main['model']+'.json')).exists()]
        (args.readout/'readout_analysis.json').write_text(json.dumps(additional,indent=2)+'\n',encoding='utf8')
    for report in reports:
        print(report['model'])
        for v in report['variants']:
            print(v['name'],'fit',v['own_appraisal_selectivity'],
                  'behavior',{h:(c['correct'],c['confidence_mean'],c['missing_confidence']) for h,c in v['behavior'].items()},
                  'own-other',v['paired_confidence']['self_wrong minus other_wrong'])
