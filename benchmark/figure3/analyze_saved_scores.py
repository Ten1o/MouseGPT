#!/usr/bin/env python3
"""Read-only, deterministic source-data export for MouseGPT Extended Data Figure 3.

Run with Python + numpy. This imports the benchmark's existing pure scorers,
validates saved source bindings, and never calls a provider or writes the DB.
All exported numeric metrics use a 0--100 display scale.
"""
import argparse
import base64
import copy
import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np

GROUP = '042aaeb4577a028550b8d1e152b6a41a5ca8cd2084cdd93a87c99b5bd0496134'
CONFIG = '2fc1382dd6dfe837dd09c70f01867aada5d6089f4b87ac4f41d62e97e44d0c04'
RUBRIC = 'eee578b5bd8eda7d802c1543ada34712e08176b0b77b7a9a4dedb7ac1bcd3142'
JUDGE = {'provider':'azure','model':'gpt-6.1-sol','version':'gpt-6.1-sol|config:'+CONFIG}
METRICS = ['benchmark_index','factual_precision','required_fact_recall','factual_f1','structure_score','instruction_score']
SCORE_FIELDS = METRICS[1:]
ALLOWED = ('schema_version','frame_id','run_id','protocol_version','judge','rubric_sha256',
    'ground_truth_sha256','prediction_sha256','input_sha256','claims','fact_coverage',
    'structure_score','instruction_score','notes','track','recording_id')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while data:=f.read(1024*1024):h.update(data)
    return h.hexdigest()


def write_csv(path, rows):
    require(bool(rows), 'Cannot write empty source table')
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def label(model, effort):
    parts=model.split('-')
    result='GPT-'+parts[1]
    if len(parts)>2: result+=' '+' '.join(p.title() for p in parts[2:])
    return result+(f' ({effort})' if effort!='not_requested' else '')


def restore_request(log, data_dir, encoded_images):
    """Restore data URIs omitted from display logs to verify wire-payload hash."""
    payload=copy.deepcopy(log['request_payload'])
    require(log['api_mode']=='responses','Unexpected non-Responses request')
    for message in payload['input']:
        for part in message['content']:
            if part['type']=='input_image':
                meta=part.pop('image_reference')
                if meta['sha256'] not in encoded_images:
                    raw=(data_dir/meta['path']).read_bytes()
                    require(hashlib.sha256(raw).hexdigest()==meta['sha256'],'Request image checksum mismatch')
                    encoded_images[meta['sha256']]='data:'+meta['media_type']+';base64,'+base64.b64encode(raw).decode('ascii')
                part['image_url']=encoded_images[meta['sha256']]
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True,help='Source research checkout with its separately held benchmark data and receipts')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--iterations',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=20261008)
    parser.add_argument('--independent-audit',type=Path,help='Optional directory of independently computed audit CSVs; used only to certify reproduction, not as a numeric input.')
    args=parser.parse_args(); root=args.project.resolve(); out=args.output.resolve()
    data=root/'benchmark/data'; source=out/'source_data';source.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root/'benchmark'))
    from mousebench.store import digest
    from mousebench.evaluation import prepare_judgment_scorer
    from mousebench.protocol import prepare_benchmark_index, rubric_hash
    from mousebench.recorded_inputs import verify_input_images
    db_path=data/'benchmark.sqlite3'
    con=sqlite3.connect(db_path.as_uri()+'?mode=ro',uri=True)
    con.execute('PRAGMA query_only=ON');con.execute('BEGIN')
    manifest_path=data/'judging/2026-10-08-gpt61-sol-medium-full435x14/manifest.json'
    campaign=json.loads(manifest_path.read_text());run_ids=sorted(b['run_id'] for b in campaign['batches'])
    frame_ids=sorted(campaign['frame_ids'])
    require(len(set(run_ids))==14 and len(set(frame_ids))==435,'Unexpected planned coverage')
    def objects(query, params=()):
        return [json.loads(row[0]) for row in con.execute(query,params)]
    frames={r['id']:r for r in objects("SELECT data FROM records WHERE kind='frames'")}
    versions={kind:{r['frame_id']:r for r in objects("SELECT data FROM versions WHERE kind=? ORDER BY revision",(kind,))} for kind in ('gt','input')}
    rubric=objects("SELECT data FROM wf_records WHERE kind='rubric' AND id=?",(campaign['rubric_id'],))[0]
    require(rubric['status']=='frozen' and rubric['sha256']==RUBRIC and rubric_hash(rubric['content'])==RUBRIC,'Frozen rubric binding mismatch')
    scorer=prepare_judgment_scorer(rubric['content']); indexer=prepare_benchmark_index(rubric['content'])
    issues=objects("SELECT data FROM wf_records WHERE kind='issue'")
    critical={(i.get('frame_id'),i.get('content_sha256'),i.get('input_sha256')) for i in issues if i.get('status')=='open' and i.get('severity')=='critical' and i.get('target','reference')=='reference'}
    bindings=[];counts=Counter();clusters={};gt_hashes=[];input_hashes=[]
    for fid in frame_ids:
        frame=frames[fid];gt=versions['gt'][fid];ip=versions['input'][fid]
        tags=set(frame['tags'])&{'v1.0','v1.1'};require(len(tags)==1,'Ambiguous collection membership')
        cohort=next(iter(tags));recording=frame['recording_id'];counts[cohort]+=1
        require(recording==fid.rsplit('_',1)[0],'Unexpected recording definition')
        if recording in clusters:require(clusters[recording]==cohort,'Recording shared between cohorts')
        clusters[recording]=cohort
        require(gt['status']=='approved' and gt['provenance'].get('human_confirmed') is True,'Reference not human approved')
        require(digest(gt['content'])==gt['content_sha256'],'GT content hash mismatch')
        require(gt['input_hash']==ip['input_hash'] and ip['track']=='image_velocity','GT/input mismatch')
        require((fid,gt['content_sha256'],ip['input_hash']) not in critical,'Open critical reference issue')
        verify_input_images(SimpleNamespace(root=data),ip)
        row={'frame_id':fid,'recording_id':recording,'collection':cohort,'gt_revision':gt['revision'],
             'gt_content_sha256':gt['content_sha256'],'gt_record_sha256':digest(gt),
             'input_revision':ip['revision'],'input_sha256':ip['input_hash'],'input_record_sha256':digest(ip),
             'input_protocol':ip['protocol_version'],'input_parent_protocol':ip.get('adaptation',{}).get('parent_input',{}).get('protocol_version',''),'input_recovery_status':frame.get('input_provenance_status') or 'verified_archived_request','view1_path':ip['input_images'][0]['path'],
             'view1_sha256':ip['input_images'][0]['sha256'],'view2_path':ip['input_images'][1]['path'],
             'view2_sha256':ip['input_images'][1]['sha256'],'gt_status':gt['status'],
             'human_confirmed':True,'assistant_generated':gt['provenance'].get('assistant_generated',False)}
        bindings.append(row);gt_hashes.append([fid,digest(gt)]);input_hashes.append([fid,digest(ip)])
    require(counts=={'v1.0':395,'v1.1':40},'Unexpected cohort counts')
    require(Counter(clusters.values())=={'v1.0':20,'v1.1':5},'Unexpected recording counts')
    metadata=[];per_frame=[];source_rows=[];encoded={};max_score_disagreement=0.;judge_batches=[]
    for b in campaign['batches']:
        batch=objects("SELECT data FROM wf_records WHERE kind='batch' AND id=?",(b['batch_id'],))[0]
        require(batch['execution_config_sha256']==CONFIG and batch['model']=='gpt-6.1-sol' and batch['provider_config']['settings']['reasoning_effort']=='medium','Judge config mismatch')
        # Recompute frozen judge config from its saved system instruction and result schema.
        job=objects('SELECT data FROM wf_jobs WHERE batch_id=? LIMIT 1',(b['batch_id'],))[0]
        config_content={'executor':batch['executor'],'model':batch['model'],'config':batch['provider_config'],
            'system_instruction':job['packet']['payload_template']['messages'][0]['content'],
            'schema':job['packet']['expected_result_schema'],'rubric_sha256':batch['rubric_sha256']}
        require(digest(config_content)==CONFIG,'Judge configuration checksum mismatch')
        judge_batches.append({'batch_id':batch['id'],'batch_record_sha256':digest(batch),
            'system_instruction_sha256':digest(config_content['system_instruction']),
            'expected_result_schema_sha256':digest(config_content['schema'])})
    for rid in run_ids:
        run=objects('SELECT data FROM runs WHERE id=?',(rid,))[0]
        predictions=objects('SELECT data FROM predictions WHERE run_id=? ORDER BY frame_id',(rid,))
        require(len(predictions)==435 and sorted(p['frame_id'] for p in predictions)==frame_ids,'Incomplete prediction coverage')
        require(run['track']=='controlled' and run['eligibility']=='request_contract_matched','Run not controlled/matched')
        judgments=[j for j in objects('SELECT data FROM judgments WHERE run_id=? ORDER BY frame_id',(rid,)) if j['judge']==JUDGE]
        require(len(judgments)==435 and sorted(j['frame_id'] for j in judgments)==frame_ids,'Incomplete/mixed final judgment coverage')
        pmap={p['frame_id']:p for p in predictions};efforts=set();observed=set();settings={};generation_dates=[]
        for j in judgments:
            fid=j['frame_id'];gt=versions['gt'][fid];ip=versions['input'][fid];p=pmap[fid];log=p['request_log']
            require(j['track']=='controlled' and j['input_track']=='image_velocity','Unexpected judgment track')
            require(j['rubric_sha256']==RUBRIC and j['ground_truth_sha256']==gt['content_sha256'] and j['gt_revision']==gt['revision'] and j['input_sha256']==ip['input_hash'],'Stale judgment binding')
            require(p['status']=='succeeded' and p['track']=='controlled','Unsuccessful native output')
            require(p['input_sha256']==ip['input_hash'] and log['input_sha256']==ip['input_hash'],'Prediction input mismatch')
            require(p['prediction_sha256']==j['prediction_sha256']==hashlib.sha256(p['output'].encode()).hexdigest(),'Prediction output hash mismatch')
            require(log['kind']=='native_execution' and log['provider']=='azure' and log['task']=='inference','Not fresh Azure inference')
            require(log['observed_model']==p['model_metadata']['observed_model']==run['model'],'Observed candidate model mismatch')
            payload=restore_request(log,data,encoded)
            require(digest(payload)==log['request_payload_sha256'],'Actual request payload hash mismatch')
            # Exact roles, text, attachment order/detail must match frozen per-frame request.
            normalized=[]
            for message in log['request_payload']['input']:
                content=[]
                for part in message['content']:
                    content.append({'type':'text','text':part['text']} if part['type']=='input_text' else {'type':'image_reference','attachment_id':part['image_reference']['attachment_id']})
                expected=ip['request_messages'][len(normalized)]['content']
                normalized.append({'role':message['role'],'content':content[0]['text'] if isinstance(expected,str) else content})
            require(normalized==ip['request_messages'],'Frozen per-frame message mismatch')
            require(log['image']==ip['input_images'],'Native input view binding mismatch')
            effort=log['settings'].get('reasoning_effort','not_requested');efforts.add(effort)
            require(p['model_metadata']['generation_settings']==log['settings'],'Candidate settings differ across provenance records')
            settings[digest(log['settings'])]=log['settings'];observed.add(log['observed_model']);generation_dates.append(log['started_at'])
            scores=scorer({k:j[k] for k in ALLOWED if k in j},dict(gt['content'],status='approved'))
            require(scores['evaluable'] is True,'Unevaluable judgment')
            for key in SCORE_FIELDS:
                delta=abs(scores[key]-j['scores'][key]);max_score_disagreement=max(max_score_disagreement,delta)
                require(delta<1e-12,'Stored scorer differs from pure rescore')
            gid=digest([j['protocol_version'],j['rubric_sha256'],j['judge'],j['track'],j['input_track'],scores['scoring_version']])
            require(gid==GROUP,'Unexpected comparison group')
            row={'run_id':rid,'model':run['model'],'reasoning_effort':effort,'display_label':label(run['model'],effort),
                 'frame_id':fid,'recording_id':frames[fid]['recording_id'],
                 'benchmark_index':indexer(scores)['value'],**{k:100*scores[k] for k in SCORE_FIELDS}}
            per_frame.append(row)
            source_rows.append({'run_id':rid,'frame_id':fid,'judgment_id':j['id'],'judgment_record_sha256':digest(j),
                'judgment_created_at':j['created_at'],'prediction_sha256':p['prediction_sha256'],
                'prediction_record_sha256':digest(p),'native_request_payload_sha256':log['request_payload_sha256'],
                'native_request_id':log['request_id'],'native_response_id':log['response_id'],
                'input_sha256':j['input_sha256'],'gt_content_sha256':j['ground_truth_sha256'],'gt_revision':j['gt_revision'],
                'rubric_sha256':j['rubric_sha256'],'comparison_group_id':gid})
        require(len(efforts)==1,'Mixed candidate reasoning effort')
        effort=next(iter(efforts))
        metadata.append({'run_id':rid,'model':run['model'],'reasoning_effort':effort,'display_label':label(run['model'],effort),
            'observed_models':sorted(observed),'generation_settings_variants':list(settings.values()),
            'native_execution_count':len(predictions),'run_record_sha256':digest(run),
            'run_name':run['name'],'first_execution_started_at':min(generation_dates),'last_execution_started_at':max(generation_dates),
            'model_snapshot_caveat':'Observed model identifier is verified from saved provider responses; no more specific dated snapshot is exposed in these records.'})
    selected_judgments={r['judgment_id'] for r in source_rows}
    normalization_rows=[]
    for (text,) in con.execute("SELECT json_extract(data,'$.parsed_result_normalization') FROM wf_jobs WHERE json_extract(data,'$.parsed_result_normalization') IS NOT NULL"):
        record=json.loads(text)
        if record.get('judgment_id') not in selected_judgments:continue
        require(record.get('semantic_rating_fields_changed') is False and record.get('raw_provider_result_preserved') is True,'Unexplained normalization')
        normalization_rows.append({k:record.get(k) for k in ('job_id','judgment_id','policy','original_parsed_output_sha256','derived_parsed_output_sha256','raw_output_text_sha256','raw_provider_result_preserved','semantic_rating_fields_changed')})
    con.rollback();con.close()
    require(len(per_frame)==6090,'Wrong full assessment count')
    # Row order is explicit, allowing reproducible shared resampling across conditions.
    cube=np.array([[next(r for r in per_frame if r['run_id']==rid and r['frame_id']==fid)[m] for fid in frame_ids for m in METRICS] for rid in run_ids]).reshape(14,435,len(METRICS))
    rng=np.random.default_rng(args.seed)
    recordings=sorted(clusters)
    cluster_indices=[[i for i,fid in enumerate(frame_ids) if frames[fid]['recording_id']==recording] for recording in recordings]
    n=np.array([len(indices) for indices in cluster_indices])
    sums=np.stack([cube[:,indices,:].sum(axis=1) for indices in cluster_indices],axis=1)
    draws=rng.integers(0,len(recordings),size=(args.iterations,len(recordings)))
    replicas={'all':sums[:,draws,:].sum(axis=2)/n[draws].sum(axis=1)[None,:,None]}
    means={'all':cube.mean(axis=1)}
    intervals={'all':np.quantile(replicas['all'],[.025,.975],axis=1,method='linear')}
    recording_rows=[{'recording_id':recording,'frame_count':int(size)} for recording,size in zip(recordings,n)]
    lookup={r['run_id']:r for r in metadata};summaries=[];metric_rows=[]
    for k,rid in enumerate(run_ids):
        meta=lookup[rid];row={key:meta[key] for key in ('run_id','model','reasoning_effort','display_label')}
        row.update(n_frames=435,n_recordings=25,benchmark_index=means['all'][k,0],
            ci95_low=intervals['all'][0,k,0],ci95_high=intervals['all'][1,k,0])
        summaries.append(row)
        for cohort in ('all',):
            for mi,metric in enumerate(METRICS):
                metric_rows.append({'run_id':rid,'model':meta['model'],'reasoning_effort':meta['reasoning_effort'],'display_label':meta['display_label'],
                    'collection':cohort,'metric':metric,'mean':means[cohort][k,mi],
                    'ci95_low':intervals[cohort][0,k,mi],'ci95_high':intervals[cohort][1,k,mi],
                    'n_frames':435,'n_recordings':25,
                    'scale':'0-100','interval':'recording_cluster_percentile95'})
    summaries.sort(key=lambda r:(-r['benchmark_index'],r['run_id']))
    summaries=[dict(rank=i+1,**r) for i,r in enumerate(summaries)]
    tables={'per_frame_scores.csv':per_frame,'model_summary.csv':summaries,'metric_summary.csv':metric_rows,
        'gpt4o_dimensions.csv':[r for r in metric_rows if r['model']=='gpt-4o' and r['collection']=='all' and r['metric'] in ['factual_precision','required_fact_recall','structure_score','instruction_score']],
        'frame_bindings.csv':bindings,'assessment_bindings.csv':source_rows,'recording_counts.csv':recording_rows,
        'normalization_bindings.csv':normalization_rows}
    from difficulty_analysis import analyze as analyze_difficulty
    difficulty_tables,difficulty_info=analyze_difficulty(per_frame,versions['gt'])
    tables.update(difficulty_tables)
    for name,rows in tables.items():write_csv(source/name,rows)
    files={str(path.relative_to(out)):file_hash(path) for path in source.glob('*.csv') if path.name in tables}
    source_files=[root/'benchmark/mousebench'/n for n in ('evaluation.py','protocol.py','recorded_inputs.py','store.py')]
    audit_files=[manifest_path,manifest_path.with_name('terminal-summary.json'),manifest_path.with_name('coverage-link-recovery-plan-01.json')]
    audit_result=None
    if args.independent_audit:
        audit_dir=args.independent_audit.resolve()
        with (audit_dir/'ci_audit.csv').open() as stream: independent_ci=list(csv.DictReader(stream))
        summary_lookup={row['display_label']:row for row in summaries};ci_error=0.
        for row in independent_ci:
            for actual,expected in [('benchmark_index','mean'),('ci95_low','unified_ci_low'),('ci95_high','unified_ci_high')]:
                ci_error=max(ci_error,abs(summary_lookup[row['model']][actual]-float(row[expected])))
        with (audit_dir/'frame_factualF_difficulty.csv').open() as stream: independent_difficulty=list(csv.DictReader(stream))
        result_lookup={row['frame_id']:row for row in tables['frame_factualF_difficulty.csv']};difficulty_error=0.
        compared=['rank','difficulty_below_mean_factualF_points','model_balanced_mean_factualF','all14_mean_factualF',
            'centered_model_disagreement_sd','models_below_baseline']+[k for k in independent_difficulty[0] if k.endswith('_centered_residual')]
        require(len(independent_ci)==14 and len(independent_difficulty)==435,'Independent audit coverage mismatch')
        for row in independent_difficulty:
            for key in compared:difficulty_error=max(difficulty_error,abs(result_lookup[row['frame_id']][key]-float(row[key])))
        require(ci_error<1e-10 and difficulty_error<1e-10,'Independent analysis reproduction failed')
        audit_result={'status':'matched','ci_models_checked':14,'difficulty_frames_checked':435,
            'max_absolute_mean_CI_difference':ci_error,'max_absolute_difficulty_residual_difference':difficulty_error,
            'files_sha256':{name:file_hash(audit_dir/name) for name in ('audit.py','ci_audit.csv','frame_factualF_difficulty.csv','factualF_analysis.json')},
            'note':'Independent audit files are validation evidence, not inputs to the calculations. All reported values are independently recomputed from verified saved assessments.'}
    provenance={'schema':'mousegpt.extended-data-figure3.analysis.v2.unified','generated_at':datetime.now(timezone.utc).isoformat(),
        'source_database':'benchmark/data/benchmark.sqlite3','read_only_snapshot':True,'sql_query_only':True,'source_database_sha256':file_hash(db_path),
        'source_membership':{'comparison_group_id':GROUP,'candidate_runs':14,'frames_per_run':435,'saved_assessments':6090,'recording_count':25,'analysis_population':'One unified benchmark; no version-tag split or version-based difficulty proxy.'},
        'input_provenance':{'historical_version_tag_counts':dict(counts),'historical_version_recording_counts':dict(Counter(clusters.values())),
            'current_protocol_counts':dict(Counter(r['input_protocol'] for r in bindings)),
            'v11_parent_protocol_counts':dict(Counter(r['input_parent_protocol'] for r in bindings if r['collection']=='v1.1')),
            'v11_recovery_status_counts':dict(Counter(r['input_recovery_status'] for r in bindings if r['collection']=='v1.1')),
            'interpretation':'Version tags are retained solely as historical input lineage, not analysis subsets or difficulty labels. Logged requests supply395current inputs. A further40inputs adapt their system/output contract to the same six-section requirements while retaining frozen views, per-frame user prompts and velocity;30derive from recovered logs and10from reconstructed requests using source velocity information. All14candidates receive the same frozen input for any given frame; prompts vary across frames.'},
        'source_content_digests':{'ground_truth_records':digest(gt_hashes),'input_records':digest(input_hashes),
            'assessment_bindings':digest(source_rows),'frame_bindings':digest(bindings)},
        'judge':dict(JUDGE,reasoning_effort='medium',execution_config_sha256=CONFIG,calibration_status='not_established'),
        'judge_batches':judge_batches,'models':metadata,
        'metric_definitions':{
            'factual_precision':'Per-frame weighted mean of distinct candidate claim correctness credits (0,0.25,0.5,0.75,1), displayed times100.',
            'required_fact_recall':'Per-frame weighted mean coverage credits (same quarter-step scale) over every required approved reference fact, displayed times100.',
            'factual_f1':'Per-frame harmonic mean of factual_precision and required_fact_recall, on their0-1 scales, displayed times100; custom graded factual F-score, not classification F1.',
            'structure_score':'Frozen judge quarter-step structural adherence score, displayed times100.',
            'instruction_score':'Frozen judge quarter-step instruction adherence score, displayed times100.',
            'benchmark_index':'100 * frame_factual_f1 * (0.8 +0.1 * frame_structure_score +0.1 * frame_instruction_score); components use0-1 scales. Compute for each frame then arithmetic mean.',
            'keywords':'Novel factual keyword assertions are assigned a semantic prose home; equivalent repeated propositions are counted once, not as independent keyword facts.'},
        'saved_receipt_normalization':{'count':len(normalization_rows),'policy':'mousebench.explicit-coverage-reciprocal-link-union.v1',
            'description':'Previously saved reverse claim/fact links derived from explicit provider coverage links. Current analysis preserves those saved judgments. Normalization metadata records that raw responses were retained and no semantic ratings changed; row-level identities and hashes are exported separately.'},'frozen_rubric':{'id':rubric['id'],'sha256':RUBRIC,'status':rubric['status'],
            'scoring_version':rubric['content']['scoring']['version'],'field_weights':rubric['content']['scoring']['field_weights'],
            'index':rubric['content']['scoring']['index'],'response_fields':rubric['content']['fields']},
        'verification':{'raw_output_sha256_match_count':6090,'wire_request_payload_sha256_match_count':6090,
            'frozen_messages_and_images_match_count':6090,'human_approved_current_references':435,
            'view_images_checksum_verified_count':870,'current_input_GT_prediction_judgment_bindings_match_count':6090,
            'stored_vs_pure_recomputed_component_max_absolute_difference':max_score_disagreement,
            'excluded_judgments':'All nonfinal judge/configuration records, including the earlier Codex pilot and archived models.',
            'score_writes':0,'provider_calls':0},
        'bootstrap':{'iterations':args.iterations,'seed':args.seed,'generator':'numpy.random.default_rng PCG64',
            'resampling_unit':'recording_id, not verified independent animal','recording_count':25,'stratified':False,
            'draw':'Uniform sampling with replacement of25of the25recordings per replicate; all selected frames and model conditions in each sampled recording travel together.',
            'shared_across_conditions_and_metrics':True,'point_estimate':'Arithmetic mean of per-frame metric across all435frames; composite calculated per frame before averaging.',
            'replicate_estimator':'Sum of all sampled recording frame scores divided by their total sampled frame count. No version strata, subset means or fixed subset weights.',
            'interval':'2.5th and97.5th percentiles; numpy.quantile method=linear',
            'inference_scope':'Conditional selected-set sensitivity to recording composition, given fixed selected frames, candidate outputs, rubric and one judge realization. Not a calibrated population interval or a model/judge repeatability interval.'},
        'empirical_difficulty':difficulty_info,'independent_reproduction':audit_result,
        'scientific_limitations':[
            'The435frames were manually selected; the benchmark is not a probability sample of deployment behavior.',
            'The25recording identifiers are not automatically independent animals; recording-level resampling does not remove any dependence across recordings from the same animal.',
            'Frozen reference approval does not demonstrate human-judge agreement. No verified heldout human calibration establishes this judge or custom grade/field/index weights.',
            'The 42-assessment Codex/Azure pilot covered only three unique frames; those older Codex scores are excluded.',
            'The composite benchmark index is a custom index, not accuracy. The graded factual F-score is not conventional classification F1.',
            'The fresh GPT-4o controlled run does not rescore historical original-model labels and does not validate the manuscript historical 78.6% claim.',
            'Intervals condition on one saved execution and judgment per frame/condition; they omit provider sampling, judge stochasticity, reference uncertainty and unverified animal-level dependence.',
            'Candidate reasoning conditions differ in effort and token caps; generation settings are provided per run rather than interpreted as an isolated randomized effort effect.'
        ],
        'software':{'python':sys.version,'numpy':np.__version__},
        'implementation_sha256':{str(p.relative_to(root)):file_hash(p) for p in source_files},
        'campaign_artifact_sha256':{str(p.relative_to(root)):file_hash(p) for p in audit_files},
        'plot_source_data_sha256':files,'analysis_script_sha256':file_hash(Path(__file__)),
        'difficulty_script_sha256':file_hash(Path(__file__).with_name('difficulty_analysis.py'))}
    (out/'analysis_provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'summary_rows':len(summaries),'per_frame_rows':len(per_frame),'gpt4o':next(r for r in summaries if r['model']=='gpt-4o'),
        'gpt4o_dimensions':tables['gpt4o_dimensions.csv'],'best':summaries[0]},indent=2))


if __name__=='__main__':main()
