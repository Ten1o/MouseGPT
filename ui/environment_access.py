"""Check activation for the interpreter selected by the existing operation routing."""
from pathlib import Path
import os
from environment_validation import validation_is_current


def requirement(op):
    if op in {'pose','triangulate'}:return 'pose','pose'
    if op=='agent-launch':return 'agent','agent'
    if op in {'preprocess','convert','pose-previews','reprojection'}:return 'core','preprocess'
    return 'core','analysis'


def same_python(left,right):
    return bool(left and right) and os.path.normcase(str(Path(left).expanduser().absolute()))==os.path.normcase(str(Path(right).expanduser().absolute()))


def access(op,registry,interpreters,profiles):
    preferred,role=requirement(op)
    candidates=[(preferred,role)]
    # The original preprocessing default is the pose interpreter. Keep that
    # configuration usable when its pose environment is explicitly activated.
    if role=='preprocess' and same_python(interpreters.get(role),interpreters.get('pose')):
        candidates.insert(0,('pose',role))
    # Preserve the previous 3D-animation fallback, but require activation now.
    if op=='filter':candidates.append(('pose','pose'))
    missing_animation=None
    stale=None
    for profile,selected_role in candidates:
        identifier=registry.get('active',{}).get(profile)
        record=next((r for r in registry.get('environments',[]) if r['id']==identifier and r['profile']==profile),None)
        if not record or not record.get('validation',{}).get('passed') or not Path(record['python']).is_file():continue
        if not same_python(interpreters.get(selected_role),record['python']):continue
        if not validation_is_current(profile,record['validation']):
            stale=(profile,selected_role)
            continue
        if op=='filter' and not record['validation'].get('versions',{}).get('matplotlib'):
            missing_animation=profile
            continue
        if op=='analysis:major' and not same_python(interpreters.get('projection'),record['python']):continue
        return {'ready':True,'profile':profile,'role':selected_role,'name':profiles[profile]['name'],'message':''}
    if stale:
        profile,selected_role=stale;name=profiles[profile]['name']
        return {'ready':False,'profile':profile,'role':selected_role,'name':name,'reason':'stale_validation',
                'message':f'{name} was validated against older application code. Validate it again in Environments, then select Use.'}
    profile,selected_role=candidates[0];name=profiles[profile]['name']
    if missing_animation:
        name=profiles[missing_animation]['name']
        message=f'{name} is active, but this environment has not passed the Matplotlib 3D animation check. In Environments, validate and activate a compatible existing environment, or install and activate the current Core workflow release.'
        return {'ready':False,'profile':missing_animation,'role':selected_role,'name':name,'reason':'missing_animation_validation','message':message}
    message=f'No suitable environment is active. In Environments, activate a validated {name} environment'
    message+=', or install and activate it.' if profiles[profile].get('install_supported') is not False else '. Validate your existing Python environment, then select Use.'
    if op=='filter':message+=' The environment must also pass the Matplotlib 3D animation check.'
    return {'ready':False,'profile':profile,'role':selected_role,'name':name,'message':message}
