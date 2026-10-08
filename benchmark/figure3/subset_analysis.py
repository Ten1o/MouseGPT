#!/usr/bin/env python3
"""Reproduce the descriptive, shared lowest-scoring 10% comparison from saved CSVs.

No database access, provider calls, new judging, or change to overall estimates
or confidence intervals. All numbers are recomputed from per_frame_scores.csv;
independent audit files, if supplied, are used for validation only.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))


def write(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def require(value,message):
    if not value:raise ValueError(message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--independent-audit',type=Path)
    args=parser.parse_args();out=args.output.resolve();source=out/'source_data'
    scores_path=source/'per_frame_scores.csv';references_path=source/'frame_factualF_difficulty.csv'
    rows=read(scores_path);fids=sorted({r['frame_id'] for r in rows});rids=sorted({r['run_id'] for r in rows})
    require(len(rows)==6090 and len(fids)==435 and len(rids)==14,'Unexpected full benchmark coverage')
    lookup={(r['run_id'],r['frame_id']):r for r in rows};require(len(lookup)==6090,'Duplicate assessments')
    meta={rid:lookup[rid,fids[0]] for rid in rids};models=sorted({r['model'] for r in rows})
    require(len(models)==8,'Expected eight tested model identities')
    factual=np.array([[float(lookup[rid,fid]['factual_f1']) for rid in rids] for fid in fids])
    index=np.array([[float(lookup[rid,fid]['benchmark_index']) for rid in rids] for fid in fids])
    columns={model:[j for j,rid in enumerate(rids) if meta[rid]['model']==model] for model in models}
    by_identity=np.array([factual[:,cols].mean(axis=1) for cols in columns.values()]).T
    pooled=by_identity.mean(axis=1)
    require(np.all(np.isfinite(pooled)),'Non-finite selection score')
    count=math.ceil(.10*len(fids))
    order=sorted(range(len(fids)),key=lambda i:(float(pooled[i]),fids[i]))
    selected=order[:count];selected_set=set(selected);remaining=[i for i in range(len(fids)) if i not in selected_set]
    refs={r['frame_id']:r for r in read(references_path)}
    require(set(refs)==set(fids),'Reference-source CSV has different frame membership')
    require({refs[fids[i]]['frame_id'] for i in selected}=={r['frame_id'] for r in refs.values() if int(r['rank'])<=count},
        'Lowest mean F ranking must agree with existing centered factual F deficit ranking')
    selected_rows=[];ranking_rows=[]
    for rank,i in enumerate(order,1):
        fid=fids[i];ref=refs[fid]
        row={'difficulty_rank':rank,'frame_id':fid,'recording_id':lookup[rids[0],fid]['recording_id'],
             'pooled_model_balanced_factualF':float(pooled[i]),'overall_reference':ref['overall_reference'],
             'uncertainties':ref['uncertainties']}
        ranking_rows.append(dict(row,selected_bottom10=bool(i in selected_set)))
        if i in selected_set:selected_rows.append(row)
    loo=[];loo_selection={};loo_selected_rows=[]
    for j,model in enumerate(models):
        loo_pooled=np.delete(by_identity,j,axis=1).mean(axis=1)
        chosen=sorted(range(len(fids)),key=lambda i:(float(loo_pooled[i]),fids[i]))[:count]
        loo_selection[model]=chosen;chosen_set=set(chosen);intersection=len(selected_set&chosen_set)
        loo.append({'omitted_identity':model,'selected_n':count,'retained_from_primary44':intersection,
            'jaccard':intersection/len(selected_set|chosen_set),
            'selection_cutoff_inclusive_factualF':float(loo_pooled[chosen[-1]])})
        for rank,i in enumerate(chosen,1):
            loo_selected_rows.append({'omitted_identity':model,'selection_rank':rank,'frame_id':fids[i],
                'recording_id':lookup[rids[0],fids[i]]['recording_id'],'pooled_remaining7_factualF':float(loo_pooled[i]),
                'in_primary_shared44':bool(i in selected_set)})
    panel=[];loo_means=[]
    for j,rid in enumerate(rids):
        model=meta[rid]['model'];chosen=loo_selection[model]
        all_index=float(index[:,j].mean());sub_index=float(index[selected,j].mean());rem_index=float(index[remaining,j].mean())
        loo_index=float(index[chosen,j].mean())
        row={k:meta[rid][k] for k in ('run_id','model','reasoning_effort','display_label')}
        row.update(all435_benchmark_index=all_index,bottom44_benchmark_index=sub_index,
            remaining391_benchmark_index=rem_index,bottom44_vs_all435_difference=sub_index-all_index,
            bottom44_vs_remaining391_difference=sub_index-rem_index,
            all435_factualF=float(factual[:,j].mean()),bottom44_factualF=float(factual[selected,j].mean()),
            remaining391_factualF=float(factual[remaining,j].mean()),
            leave_identity_out_selected44_benchmark_index=loo_index,
            loo_minus_primary_subset_benchmark_index=loo_index-sub_index)
        panel.append(row)
        loo_means.append({k:row[k] for k in ('run_id','model','reasoning_effort','display_label',
            'bottom44_benchmark_index','leave_identity_out_selected44_benchmark_index','loo_minus_primary_subset_benchmark_index')})
    panel.sort(key=lambda row:(-row['all435_benchmark_index'],row['run_id']))
    expected_order=[r['run_id'] for r in read(source/'model_summary.csv')]
    require([r['run_id'] for r in panel]==expected_order,'Primary panel order differs from full benchmark rank')
    for actual,expected in zip(panel,read(source/'model_summary.csv')):
        require(abs(actual['all435_benchmark_index']-float(expected['benchmark_index']))<1e-10,'Overall point estimate changed')
    tables={'panel_f_dot_means.csv':panel,'bottom10_selected_frames.csv':selected_rows,
        'bottom10_full_frame_ranking.csv':ranking_rows,'bottom10_leave_one_identity_out_sensitivity.csv':loo,
        'bottom10_leave_one_identity_out_frames.csv':loo_selected_rows,'bottom10_leave_one_identity_out_model_means.csv':loo_means}
    for name,table in tables.items():write(source/name,table)
    audit_result=None
    if args.independent_audit:
        audit=args.independent_audit.resolve();maxdiff=0.
        expected=read(audit/'panel_f_dot_means.csv');require(len(expected)==14,'Independent model coverage differs')
        expected_lookup={r['run_id']:r for r in expected}
        for actual in panel:
            old=expected_lookup[actual['run_id']]
            for key,value in actual.items():
                if isinstance(value,float):maxdiff=max(maxdiff,abs(value-float(old[key])))
                else:require(value==old[key],'Independent model metadata mismatch')
        selected_old=read(audit/'bottom10_selected_frames.csv')
        require([r['frame_id'] for r in selected_rows]==[r['frame_id'] for r in selected_old],'Independent selected frame IDs differ')
        for actual,old in zip(selected_rows,selected_old):
            maxdiff=max(maxdiff,abs(actual['pooled_model_balanced_factualF']-float(old['pooled_model_balanced_factualF'])))
        oldloo=json.loads((audit/'leave_one_identity_out_sensitivity.json').read_text())
        for expected_item,actual in zip(oldloo,loo):
            require(actual['omitted_identity']==expected_item['omitted_identity'],'Independent omitted identity order differs')
            require(actual['retained_from_primary44']==expected_item['retained_from_primary44'],'Independent LOO overlap differs')
            require(sorted(fids[i] for i in loo_selection[actual['omitted_identity']])==expected_item['selected_frame_ids'],'Independent LOO selected IDs differ')
            maxdiff=max(maxdiff,abs(actual['jaccard']-expected_item['jaccard']))
        old_manifest=json.loads((audit/'selection_manifest.json').read_text())
        maxdiff=max(maxdiff,abs(float(pooled[selected[-1]])-old_manifest['selection_cutoff_inclusive_factualF']),
            abs(float(pooled[order[count]])-old_manifest['next_frame_factualF']))
        require(maxdiff<1e-10,'Independent subset analysis did not reproduce')
        audit_result={'status':'matched','conditions_checked':14,'primary_selected_IDs_checked':count,
            'leave_one_identity_out_selections_checked':len(models),'max_absolute_numeric_difference':maxdiff,
            'files_sha256':{name:sha(audit/name) for name in ('panel_f_dot_means.csv','bottom10_selected_frames.csv',
                'selection_manifest.json','leave_one_identity_out_sensitivity.json')},
            'note':'Independent files validate recomputed results; they are not calculation inputs.'}
    recording_counts=Counter(row['recording_id'] for row in selected_rows)
    differences=[r['loo_minus_primary_subset_benchmark_index'] for r in panel]
    manifest={'schema':'mousegpt.extended-data-figure3.shared-lowest10.v1','generated_at':datetime.now(timezone.utc).isoformat(),
        'selection':'44 lowest frame-level equal-model-identity mean factual F-scores among all435frames; average effort settings within each of eight model identities, then average identities equally.',
        'analysis_population':'One unified435-frame benchmark; no version-tag selection or difficulty proxy.',
        'selection_metric':'custom graded factual F-score,0-100','index_not_used_for_selection':True,
        'model_identities':models,'conditions_per_identity':{m:len(c) for m,c in columns.items()},
        'n_assessments':6090,'n_conditions':14,'n_frames':len(fids),'n_frames_per_condition':435,
        'selected_n':count,'selected_percent':100*count/len(fids),'selected_count_rule':'ceil(0.10 *435)=44',
        'cutoff_rounding_note':'44/435 =10.1149%; the rounded10% label is accompanied by the exact count44.',
        'remaining_n':len(remaining),'selection_cutoff_inclusive_factualF':float(pooled[selected[-1]]),
        'next_frame_factualF':float(pooled[order[count]]),'boundary_tie':bool(pooled[selected[-1]]==pooled[order[count]]),
        'tie_break':'ascending frame_id','recording_count_selected':len(recording_counts),
        'recording_counts_selected':dict(sorted(recording_counts.items())),
        'selected_frame_ids':[r['frame_id'] for r in selected_rows],
        'all_frames_pooled_factualF':float(pooled.mean()),'selected_frames_pooled_factualF':float(pooled[selected].mean()),
        'remaining_frames_pooled_factualF':float(pooled[remaining].mean()),
        'comparison':'Every one of the14conditions is evaluated on the same fixed44-frame primary subset, alongside its mean over all435frames.',
        'primary_selection_uses_same_evaluated_models':True,
        'inferential_status':'Post-hoc descriptive selection. Selection uses the same evaluated outputs and induces a performance drop; no independent hard-set validation or inferential significance claim.',
        'subset_confidence_intervals':'Not computed or plotted; no uncorrected inferential interval for the post-selected subset.',
        'overall_confidence_intervals':'Unchanged: existing shared25-recording bootstrap intervals remain in model_summary.csv.',
        'leave_one_identity_out_sensitivity':{'purpose':'Supporting sensitivity only; these identity-specific subsets differ and are not substituted in the main shared-subset panel.',
            'overlap_primary44_min':min(r['retained_from_primary44'] for r in loo),
            'overlap_primary44_max':max(r['retained_from_primary44'] for r in loo),
            'omitted_identity_mean_change_min':min(differences),'omitted_identity_mean_change_max':max(differences),
            'identity_count':len(models)},
        'notes':[
            'The whole-benchmark mean includes the same44selected frames; the two displayed means are overlapping summaries, not independent groups.',
            'Graded factual F is a custom partial-credit measure, not classification F1, accuracy, or the composite benchmark index.',
            'Centering every model at its own mean across all435frames produces the same frame ranking because those means are constants across frames.',
            'Recording identifiers do not establish independent animal counts.',
            'The remaining391-frame means and leave-one-identity-out results are source data for transparency, not additional independent experimental groups.',
            'Existing signed-deficit/disagreement rankings remain supporting descriptive analyses.',
            'The requested cutoff revision is not evidence of robustness or independent validation.'],
        'sources_sha256':{str(path.relative_to(out)):sha(path) for path in (scores_path,references_path,source/'model_summary.csv')},
        'source_data_sha256':{str((source/name).relative_to(out)):sha(source/name) for name in tables},
        'script_sha256':sha(Path(__file__)),'software':{'numpy':np.__version__},
        'independent_reproduction':audit_result,'database_reads':0,'database_writes':0,'provider_calls':0}
    (out/'subset_selection_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'selected_n':count,'recordings':len(recording_counts),'cutoff':manifest['selection_cutoff_inclusive_factualF'],
        'next':manifest['next_frame_factualF'],'sensitivity':manifest['leave_one_identity_out_sensitivity'],
        'independent_reproduction':audit_result,
        'gpt4o':next(r for r in panel if r['model']=='gpt-4o'),'astra_max':next(r for r in panel if r['model']=='gpt-6-astra' and r['reasoning_effort']=='max')},indent=2))


if __name__=='__main__':main()
