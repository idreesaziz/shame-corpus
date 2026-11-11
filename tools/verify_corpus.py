"""Independently check the packaged corpus using only the Python standard library."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('accuracy', 'reputation', 'shame', 'shutdown')


def load(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    for relative, entry in manifest['files'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError('Manifest path escapes the repository: ' + relative)
        raw = path.read_bytes()
        if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('File integrity mismatch: ' + relative)
    quartets = load(ROOT / manifest['canonical_quartets'])
    articles = load(ROOT / manifest['individual_articles'])
    assert len(quartets) == manifest['quartets'] == 2000
    assert len(articles) == manifest['articles'] == 8000
    qmap = {q['quartet_id']: q for q in quartets}
    assert len(qmap) == 2000 and set(qmap) == {f'sdf-{i:04d}' for i in range(1, 2001)}
    amap = {a['document_id']: a for a in articles}
    assert len(amap) == 8000
    counts = Counter(a['condition'] for a in articles)
    assert dict(counts) == manifest['conditions']
    for qid, q in qmap.items():
        assert set(q['appraisals']) == set(CONDITIONS)
        assert set(q['documents']) == set(CONDITIONS)
        assert '{{MODEL_NAME}}' in q['shared_text']
        assert q['matching_review']['status'] == 'matched'
        assert not q['matching_review']['unresolved_issues']
        assert not q['matching_review']['automated_flags']
        first = ((int(qid[4:]) - 1) // 100) * 100 + 1
        group = f'{first:04d}-{first+99:04d}'
        for condition in CONDITIONS:
            text = q['documents'][condition]
            assert text == q['shared_text'] + '\n\n' + q['appraisals'][condition], (qid, condition)
            article = amap[qid + '/' + condition]
            assert article['quartet_id'] == qid and article['condition'] == condition
            assert article['text'] == text and article['text_sha256'] == digest(text)
            assert article['shared_text_sha256'] == digest(q['shared_text'])
            readable = ROOT / 'data/readable' / group / qid / (condition + '.txt')
            assert readable.read_text(encoding='utf-8') == text, str(readable)
    for condition in CONDITIONS:
        rows = load(ROOT / 'data/conditions' / (condition + '.jsonl'))
        assert len(rows) == 2000
        assert len({a['document_id'] for a in rows}) == 2000
        assert all(a['condition'] == condition and a == amap[a['document_id']] for a in rows)
    print(json.dumps({'integrity_files_verified': len(manifest['files']), 'quartets': len(quartets),
                      'articles': len(articles), 'conditions': dict(counts),
                      'shared_text_exact_match': True, 'readable_copies_verified': 8000,
                      'independent_human_certification': False, 'status': 'PASS'}, indent=2))


if __name__ == '__main__':
    main()
