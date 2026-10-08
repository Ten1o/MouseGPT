"""Descriptive factual-F difficulty of one 435-frame benchmark; no model calls.

All eight tested model identities receive equal weight after averaging available
reasoning-effort conditions within identity. This is conditional on this tested
panel, saved judgments and rubric; it is not a validated intrinsic difficulty.
"""
import numpy as np


def average_ranks(values):
    order=np.argsort(values,kind='stable');result=np.empty(len(values));start=0
    while start<len(values):
        end=start+1
        while end<len(values) and values[order[end]]==values[order[start]]:end+=1
        result[order[start:end]]=(start+end-1)/2+1;start=end
    return result


def spearman(a,b):
    return float(np.corrcoef(average_ranks(a),average_ranks(b))[0,1])


def ordered_ranks(values):
    # Alphabetical frame ordering upstream deterministically breaks exact ties.
    order=np.argsort(-values,kind='stable');ranks=np.empty(len(values),dtype=int)
    ranks[order]=np.arange(1,len(values)+1)
    return order,ranks


def analyze(per_frame, references):
    fids=sorted({row['frame_id'] for row in per_frame})
    rids=sorted({row['run_id'] for row in per_frame})
    lookup={(r['run_id'],r['frame_id']):r for r in per_frame}
    meta={rid:next(r for r in per_frame if r['run_id']==rid) for rid in rids}
    models=sorted({r['model'] for r in per_frame})
    factual=np.array([[lookup[rid,fid]['factual_f1'] for rid in rids] for fid in fids])
    index=np.array([[lookup[rid,fid]['benchmark_index'] for rid in rids] for fid in fids])
    columns={model:[i for i,rid in enumerate(rids) if meta[rid]['model']==model] for model in models}
    score=np.array([factual[:,cs].mean(axis=1) for cs in columns.values()]).T
    baseline=score.mean(axis=0);residual=score-baseline
    difficulty=-residual.mean(axis=1);disagreement=residual.std(axis=1,ddof=0)
    index_score=np.array([index[:,cs].mean(axis=1) for cs in columns.values()]).T
    index_deficit=-(index_score-index_score.mean(axis=0)).mean(axis=1)
    equal14=-(factual-factual.mean(axis=0)).mean(axis=1)
    standardized=-(residual/residual.std(axis=0,ddof=0)).mean(axis=1)
    order,ranks=ordered_ranks(difficulty);top=set(order[:10]);top20=set(order[:20]);tables=[];per_model=[]
    for i in order:
        fid=fids[i];ref=references[fid]['content']
        row={'rank':int(ranks[i]),'frame_id':fid,'recording_id':lookup[rids[0],fid]['recording_id'],
             'difficulty_below_mean_factualF_points':float(difficulty[i]),
             'model_balanced_mean_factualF':float(score[i].mean()),'all14_mean_factualF':float(factual[i].mean()),
             'centered_model_disagreement_sd':float(disagreement[i]),'models_below_baseline':int((residual[i]<0).sum()),
             'model_count':len(models),'overall_reference':ref.get('sections',{}).get('Overall',''),
             'uncertainties':' | '.join(ref.get('uncertainties',[])),
             **{model+'_centered_residual':float(residual[i,j]) for j,model in enumerate(models)}}
        tables.append(row)
        for j,model in enumerate(models):
            per_model.append({'frame_id':fid,'recording_id':row['recording_id'],'model':model,
                'effort_condition_count':len(columns[model]),'model_mean_factualF_435':float(baseline[j]),
                'frame_model_mean_factualF':float(score[i,j]),'centered_residual_factualF_points':float(residual[i,j]),
                'deficit_factualF_points':float(-residual[i,j]),'model_identity_weight':1/len(models)})
    sensitivity=[];loo=[];rank_rows=[]
    def compare(name,values,omitted=''):
        altorder,_=ordered_ranks(values)
        sensitivity.append({'analysis':name,'omitted_model':omitted,'spearman_full_ranking':spearman(difficulty,values),
            'top10_overlap':len(top&set(altorder[:10])),'top20_overlap':len(top20&set(altorder[:20])),
            'frame_count':len(fids)})
    compare('factualF_vs_composite_index',index_deficit)
    compare('equal8_model_identities_vs_equal14_conditions',equal14)
    compare('raw_vs_standardized_model_residuals',standardized)
    loo_ranks={};loo_example={}
    for j,model in enumerate(models):
        r=np.delete(residual,j,axis=1);d=-r.mean(axis=1);v=r.std(axis=1,ddof=0)
        _,lr=ordered_ranks(d);loo_ranks[model]=lr
        compare('leave_one_model_identity_out',d,model)
        for i,fid in enumerate(fids):
            loo.append({'frame_id':fid,'omitted_model':model,'remaining_model_identities':len(models)-1,
                'difficulty_below_mean_factualF_points':float(d[i]),'centered_model_disagreement_sd':float(v[i]),
                'difficulty_rank':int(lr[i])})
        loo_example[model]={fid:int(lr[fids.index(fid)]) for fid in ('231117_G5_30min_00007684','231121_C3M3-2_00038929')}
    _,ir=ordered_ranks(index_deficit);_,er=ordered_ranks(equal14);_,sr=ordered_ranks(standardized)
    for i,fid in enumerate(fids):
        rank_rows.append({'frame_id':fid,'factualF_equal8_rank':int(ranks[i]),'composite_index_equal8_rank':int(ir[i]),
            'factualF_equal14_rank':int(er[i]),'standardized_residual_rank':int(sr[i]),
            **{'omit_'+model+'_rank':int(loo_ranks[model][i]) for model in models}})
    info={'method':'One unified benchmark. Average saved factual F-scores (0-100) over effort conditions within each of eight model identities. Center each identity at its arithmetic mean across all435frames. D_i=mean_m(mu_m-s_im), V_i=population SD_m(s_im-mu_m).',
        'metric':'custom graded factual F-score on0-100scale','frame_count':len(fids),'condition_count':len(rids),
        'model_identity_count':len(models),'model_ids':models,'model_effort_condition_counts':{m:len(c) for m,c in columns.items()},
        'model_baseline_factualF':{m:float(baseline[j]) for j,m in enumerate(models)},
        'balanced_grandmean_factualF':float(score.mean()),'mean_signed_deficit':float(difficulty.mean()),
        'difficulty_sign':'Positive = below each model identity own435-frame baseline; negative = above baseline.',
        'disagreement_definition':'Population SD across the eight centered identity residuals (ddof=0); descriptive across-model heterogeneity, not a standard error or confidence interval.',
        'difficulty_vs_disagreement_spearman':spearman(difficulty,disagreement),'sensitivity':sensitivity,
        'example_leave_one_model_out_ranks':loo_example,
        'limitations':[
            'Empirical difficulty and disagreement are properties of this tested model panel, saved judge and selected benchmark; they do not establish intrinsic visual difficulty.',
            'Difficulty is computed from the same outcomes displayed, not independently calibrated or held out; no causal or predictive validation claim is made.',
            'Effort conditions are averaged within identity to prevent identities with two conditions receiving twice the weight of those with one.',
            'One model identity can influence a specific example strongly even when full ranking correlations are high; disclose leave-one-model-out sensitivity.',
            'No benchmark version tag, recording subset or hand-assigned behavior label is used to define difficulty.',
            'The tested model identities are related and are not eight independent experimental replicates.']}
    return {'frame_factualF_difficulty.csv':tables,'difficulty_per_model.csv':per_model,
        'difficulty_sensitivity.csv':sensitivity,'difficulty_leave_one_model_out.csv':loo,
        'difficulty_rank_sensitivity.csv':rank_rows},info
