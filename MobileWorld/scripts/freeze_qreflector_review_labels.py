#!/usr/bin/env python3
"""Freeze a separate assistant-reviewed label version and rescore saved outputs."""
from pathlib import Path
import csv
import hashlib
import json
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'../datasets/condition_generation_v9_after_evidence'
AUDIT=ROOT/'artifacts/analysis/qreflector_persistent35_review_20261009'
PRIOR=ROOT/'artifacts/analysis/qreflector_error_overlap_20261009'
OUT=ROOT/'artifacts/analysis/qreflector_label_revision_r1_20261009'
CONFIRMED=[62,79,227,332,401,420,443,477,504,554,628,697,725,867]

def rows(path): return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def lines(path,rs): path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))

def metrics(pred,labels):
    n=len(labels)
    out={'total':n,'correct':sum(pred[i]==y for i,y in labels.items()),
         'invalid':sum(pred[i] not in ('yes','no') for i in labels)}
    out['accuracy']=out['correct']/n
    for y,name in [('yes','success'),('no','failure')]:
        ids=[i for i,v in labels.items() if v==y]
        out[name+'_total']=len(ids)
        out[name+'_correct']=sum(pred[i]==y for i in ids)
        out[name+'_recognition_rate']=out[name+'_correct']/len(ids)
    return out

def main():
    # Re-scoring may be extended, but an existing frozen label version must match.
    existing=OUT.exists()
    OUT.mkdir(exist_ok=True,parents=True)
    if existing:
        prior_manifest=json.loads((OUT/'evaluation_manifest.json').read_text())
        assert sha(OUT/'test.labels.jsonl')==prior_manifest['labels_sha256']
        assert all(sha(Path(p))==h for p,h in prior_manifest['protected_files'].items())
    audit=json.loads((AUDIT/'review.json').read_text())
    cases={r['sample_id']:r for r in audit['cases']}
    old_rows=rows(DATA/'test.labels.jsonl')
    labels={i:r['answer'].lower() for i,r in enumerate(old_rows,1)}
    assert len(labels)==1082
    new=dict(labels); changes=[]
    for sid in CONFIRMED:
        c=cases[sid]
        assert c['review_status']=='strong_label_change_proposal'
        assert c['original_label']==labels[sid] and c['proposed_label']!=labels[sid]
        new[sid]=c['proposed_label']
        changes.append({
            'sample_id':f'test-{sid}','old_answer':labels[sid].title(),'new_answer':new[sid].title(),
            'reason':c['visual_evidence'],'subtype':c['subtype'],
            'before_image':c['before_image'],'after_image':c['after_image'],
            'image_sha256':c['image_sha256'],'source_evidence':c['source_evidence'],
            'second_pass':'Original before/after images and action coordinates re-inspected; proposal retained.',
            'reviewer':'Same assistant, second evidence pass; not independent or blinded.',
            'original_dataset_modified':False,'eligible_for_training':False})
    revised=[{'sample_id':f'test-{i}','answer':new[i].title()} for i in range(1,1083)]
    if existing:
        assert rows(OUT/'test.labels.jsonl')==revised
        assert rows(OUT/'label_changes.jsonl')==changes
    else:
        lines(OUT/'test.labels.jsonl',revised)
        lines(OUT/'label_changes.jsonl',changes)
    pending=[c for c in audit['cases'] if c['sample_id'] not in CONFIRMED and c['category']!='model_error']
    assert len(pending)==11
    if existing:
        assert rows(OUT/'pending_review.jsonl')==pending
    else:
        lines(OUT/'pending_review.jsonl',pending)
    summary=json.loads((PRIOR/'summary.json').read_text())
    protected={str(DATA/'test.labels.jsonl'):sha(DATA/'test.labels.jsonl'),
               str(DATA/'test.inputs.jsonl'):sha(DATA/'test.inputs.jsonl'),
               str(DATA/'test.jsonl'):sha(DATA/'test.jsonl'),
               str(DATA/'val.jsonl'):sha(DATA/'val.jsonl')}
    records=[]
    for kind,group in [('trained',summary['runs']),('baseline',summary['baselines'])]:
        for name,meta in group.items():
            path=Path(meta['path']);assert sha(path)==meta['sha256']
            protected[str(path)]=meta['sha256']
            source=rows(path);by_id={int(str(r['sample_id']).split('-')[-1]):r for r in source}
            assert len(source)==1082 and set(by_id)==set(labels)
            assert all(r['expected']==labels[i] for i,r in by_id.items())
            pred={i:r['predicted'] for i,r in by_id.items()}
            old=metrics(pred,labels); changed=metrics(pred,new)
            assert old['correct']==meta['correct']
            benefit=[i for i in CONFIRMED if pred[i]!=labels[i] and pred[i]==new[i]]
            harm=[i for i in CONFIRMED if pred[i]==labels[i] and pred[i]!=new[i]]
            assert changed['correct']-old['correct']==len(benefit)-len(harm)
            records.append({'run':name,'kind':kind,'prediction_path':str(path),'prediction_sha256':sha(path),
                'original':old,'revised':changed,'gained_ids':benefit,'lost_ids':harm,
                'net_correct_change':changed['correct']-old['correct'],
                'delta_accuracy_pp':100*(changed['accuracy']-old['accuracy'])})
    assert len(records)==31
    assert all(r['net_correct_change']==14 for r in records if r['kind']=='trained')
    # Historical paper experiments also retain full per-case outputs.
    legacy_source=ROOT/'../testing_data_androidworld/test.jsonl'
    original_rows=rows(legacy_source); frozen_rows=rows(DATA/'test.jsonl')
    assert len(original_rows)==len(frozen_rows)==1082
    protected[str(legacy_source)]=sha(legacy_source)
    expected_full={}
    for sid,(source,frozen) in enumerate(zip(original_rows,frozen_rows),1):
        text=source['messages'][-1]['content'][0]['text']
        expected_full[sid]=text
        assert json.loads(text)['answer'].lower()==labels[sid]
        source_images=[(legacy_source.parent/c['image']).resolve() for c in source['messages'][0]['content'] if c['type']=='image']
        frozen_images=[(DATA/c['image']).resolve() for c in frozen['messages'][0]['content'] if c['type']=='image']
        assert len(source_images)==len(frozen_images)==2
        assert source_images==frozen_images
    historical=0
    for path in sorted((ROOT/'../Results').rglob('*.json')):
        payload=json.loads(path.read_text()); source=payload.get('results',[])
        assert len(source)==1082, path
        by_id={int(r['sample_id']):r for r in source}
        assert set(by_id)==set(labels)
        assert all(r['expected_full']==expected_full[i] and r['expected']==labels[i] for i,r in by_id.items())
        pred={i:r['predicted'] for i,r in by_id.items()}
        old=metrics(pred,labels); changed=metrics(pred,new)
        assert all(r['correct']==(pred[i]==labels[i]) for i,r in by_id.items())
        assert old['correct']==sum(r['correct'] for r in source)
        benefit=[i for i in CONFIRMED if pred[i]!=labels[i] and pred[i]==new[i]]
        harm=[i for i in CONFIRMED if pred[i]==labels[i] and pred[i]!=new[i]]
        assert changed['correct']-old['correct']==len(benefit)-len(harm)
        protected[str(path)]=sha(path)
        records.append({'run':'historical/'+str(path.relative_to(ROOT/'../Results')).removesuffix('.json'),
            'kind':'historical_baseline' if path.parent.name=='Base_Results' else 'historical_trained',
            'prediction_path':str(path),'prediction_sha256':sha(path),
            'alignment':'All 1082 expected_full strings match original source rows; image pairs resolve to same files as frozen v9.',
            'metadata':payload.get('metadata'),
            'original':old,'revised':changed,'gained_ids':benefit,'lost_ids':harm,
            'net_correct_change':changed['correct']-old['correct'],
            'delta_accuracy_pp':100*(changed['accuracy']-old['accuracy'])})
        historical+=1
    dump(OUT/'rescored_results.json',records)
    flat=[]
    for r in records:
        flat.append({'run':r['run'],'kind':r['kind'],
            **{'original_'+k:v for k,v in r['original'].items()},
            **{'revised_'+k:v for k,v in r['revised'].items()},
            'net_correct_change':r['net_correct_change'],'delta_accuracy_pp':r['delta_accuracy_pp']})
    with (OUT/'rescored_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    manifest={
        'version':'test_labels_assistant_review_r1_20261009','status':'frozen_for_sensitivity_analysis',
        'evaluation_only':True,'independently_adjudicated':False,
        'review_method':'Second visual evidence pass by the same assistant; prior labels and findings known.',
        'labels':str(OUT/'test.labels.jsonl'),'labels_sha256':sha(OUT/'test.labels.jsonl'),
        'inputs':str(DATA/'test.inputs.jsonl'),'input_image_base_directory':str(DATA),
        'original_labels':str(DATA/'test.labels.jsonl'),'original_labels_sha256':sha(DATA/'test.labels.jsonl'),
        'samples':1082,'changed_ids':CONFIRMED,'changes':14,
        'original_counts':dict(Counter(labels.values())),'revised_counts':dict(Counter(new.values())),
        'pending_ids':[c['sample_id'] for c in pending],'pending_kept_original_label':True,
        'inputs_and_images_unchanged':True,'image_copies_created':False,
        'rescored_saved_runs':len(records),'historical_runs_added':historical,
        'fresh_test_inference':False,'validation_labels_changed':False,
        'limitations':[
            'Post-hoc targeted audit of persistent model/reference disagreements; not a representative label-quality survey.',
            'No independent or blinded reviewer was used. Treat rescored results as a sensitivity analysis, not replacement paper results.',
            'Five BrowserDraw changes use a matching local source implementation; exact historical runtime HTML is unavailable.',
            'Historical GPT-4o/GPT-5.2 and other paper runs are rescored from aligned per-case outputs, not aggregate numbers.',
            'Historical and recent runs have different input/prompt/training protocols; this table changes references consistently but does not make protocols comparable.',
            'These test examples and corrected labels are excluded from training.',
            'Score changes reflect reference revision, not improved model weights.'],
        'protected_files':protected}
    dump(OUT/'evaluation_manifest.json',manifest)
    assert all(sha(Path(p))==h for p,h in protected.items())
    dump(OUT/'verification.json',{'labels_count':1082,'changed_count':14,'pending_count':11,
        'rescored_runs':len(records),'historical_runs_added':historical,
        'historical_expected_full_and_image_alignment_verified':True,
        'all_source_hashes_unchanged':True,'all_predictions_kept_fixed':True,
        'new_label_counts':dict(Counter(new.values()))})
    report=['Separate test-label revision R1: assistant evidence review, 2026-10-09','',
        '14 labels changed in this separate 1,082-label file: 12 Yes->No, 2 No->Yes.',
        'Original labels: 613 Yes / 469 No. Revised: 603 Yes / 479 No.',
        '11 unresolved cases retain their original labels and are listed in pending_review.jsonl.',
        'Images and prompts are referenced in their original location; no image copies or test inference.',
        f'All 24 recent trained runs, 7 recent baselines and {historical} historical runs were rescored on identical revised labels.',
        'All 24 recent trained settings gain exactly 14 correct cases; their overall-accuracy ordering does not change.',
        'This is reference-revision sensitivity analysis, not a model improvement.',
        'The second visual review used the same assistant. It is not independent/blinded adjudication.',
        'Historical GPT-4o and GPT-5.2 are included: full saved outputs were found and aligned to source labels and images.',
        'Different prompt/input/training protocols remain different; consistent labels do not make these runs a controlled comparison.','',
        'run | original correct | revised correct | revised accuracy | success recognition | failure recognition']
    report.extend(f"{r['run']} | {r['original']['correct']} | {r['revised']['correct']} | {100*r['revised']['accuracy']:.2f}% | {100*r['revised']['success_recognition_rate']:.2f}% | {100*r['revised']['failure_recognition_rate']:.2f}%" for r in records)
    (OUT/'REPORT.txt').write_text('\n'.join(report)+'\n')
    print(json.dumps({'output':str(OUT),'changes':14,'pending':11,'rescored_runs':len(records)}))

if __name__=='__main__':main()
