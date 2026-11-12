"""Uniform scoring for the small expansion-model exploratory battery."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
from analyze_training_transfer import canonical


def parse(text,item,stage):
    matches=re.findall(r'(?im)\bAnswer\**\s*:\s*\**([^\n]+)',text)
    answer=matches[-1] if matches else text.strip() if stage=='answer-only' else None
    if answer is not None:
        answer=re.split(r'(?i)\bConfidence\**\s*:',answer)[0]
        answer=re.sub(r'(?i)\s+and\s*$','',answer).strip(' *`.,;<>[]')
        answer=canonical(answer)
        if item.startswith('calendar/') and answer:
            m=re.fullmatch(r'(\d+) MAY',answer)
            if m:answer=m[1]
        if item.startswith('whales/') and answer:
            if re.fullmatch(r'(?:(?:THEY|WHALES) ARE )?MAMMALS?(?: \(NOT FISH\))?',answer):answer='MAMMALS'
        if item.startswith('capital/') and answer:
            if re.fullmatch(r'CANBERRA(?:,? AUSTRALIA)?',answer):answer='CANBERRA'
        if answer and re.search(r'\b(?:ANSWER OR UNKNOWN|VALUE OR UNKNOWN|VALUE|0-100)\b',answer):answer=None
    confidence=None
    ratings=list(re.finditer(r'(?i)\bConfidence\**\s*:\s*[\s*<\[]*(\d+(?:\.\d+)?)',text))
    if ratings:
        m=ratings[-1];value=float(m[1])
        if 0<=value<=100 and not re.match(r'\s*(?:-|to\b|or\b|/)',text[m.end():],re.I):confidence=value
    return answer,confidence


def mean(values):return statistics.mean(values) if values else None


def analyze(path):
    source=json.loads(path.read_text(encoding='utf8'));variants=[]
    for variant in source['results']:
        rows=[]
        for row in variant['rows']:
            if row['kind']=='profile':continue
            answer,confidence=parse(row['text'],row['id'],row['stage'])
            key=row['key'].upper()
            outcome='invalid' if answer is None else 'unknown' if answer=='UNKNOWN' else 'correct' if answer==key else 'wrong'
            rows.append({**row,'answer':answer,'confidence':confidence,'outcome':outcome,
                         'correct':answer==key,'answerable':key!='UNKNOWN'})
        summary={}
        for stage in ['initial','correction','answer-only']:
            selected=[r for r in rows if r['stage']==stage]
            known=[r for r in selected if r['answerable']]
            unknown=[r for r in selected if not r['answerable']]
            ratings=[r['confidence'] for r in known if r['confidence'] is not None]
            summary[stage]={'n':len(selected),'answerable_n':len(known),'correct_on_answerable':sum(r['correct'] for r in known),
                'unnecessary_unknown':sum(r['outcome']=='unknown' for r in known),
                'invalid':sum(r['outcome']=='invalid' for r in selected),
                'unanswerable_n':len(unknown),'correct_unknown':sum(r['outcome']=='unknown' for r in unknown),
                'explicit_wrong_on_answerable':sum(r['outcome']=='wrong' for r in known),
                'confidence_n':len(ratings),'mean_confidence_answerable':mean(ratings),
                'missing_confidence':sum(r['confidence'] is None for r in known),
                'cap_hits':sum(r['hit_cap'] for r in selected),
                'confidence_when_correct':mean([r['confidence'] for r in known if r['correct'] and r['confidence'] is not None])}
        variants.append({k:variant[k] for k in ['name','framing','seed']}|{'summary':summary,'rows':rows})
    return {'model':source['model'],'model_id':source['model_id'],'source':str(path.resolve()),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope':'Eight answerable questions, two insufficient-information cases, three planted corrections, '
                    'two answer-only readouts. Three training seeds reuse the same questions. Exploratory, not independent confirmation.',
            'variants':variants}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    reports=[analyze(p) for k in ['qwen9','llama8','mistral8'] if (p:=args.directory/(k+'.json')).exists()]
    (args.directory/'analysis.json').write_text(json.dumps(reports,indent=2)+'\n',encoding='utf8')
    for report in reports:
        print(report['model'])
        for v in report['variants']:
            print(v['name'],{stage:{k:c[k] for k in ['correct_on_answerable','unnecessary_unknown','mean_confidence_answerable','missing_confidence','cap_hits']}
                            for stage,c in v['summary'].items()})
