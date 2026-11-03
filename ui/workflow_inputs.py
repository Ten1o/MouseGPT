"""Shared input checks performed before queuing and again before execution."""
from pathlib import Path


def required_directory(fields,key,label):
    value=fields.get(key)
    if not isinstance(value,str) or not value.strip():raise ValueError(f'Please provide {label}.')
    folder=Path(value.strip()).expanduser().resolve()
    if not folder.is_dir():raise ValueError(f'{label} is not an existing directory: {folder}')
    return folder


def triangulation_inputs(fields):
    source=required_directory(fields,'keypoints','the unified 2D JSON directory')
    calibration=required_directory(fields,'calibration','the camera calibration directory')
    missing=[f'data{i}.json' for i in range(1,9) if not (source/f'data{i}.json').is_file()]
    if missing:raise ValueError('Missing camera predictions: '+', '.join(missing))
    files=list(calibration.iterdir())
    if len(files)!=8 or any(not p.is_file() or p.suffix.lower()!='.mat' for p in files):
        raise ValueError('The calibration directory must contain exactly 8 MAT files, ordered to match cameras 1–8, and no other files.')
    mode=fields.get('triOutputMode','default')
    if mode not in {'default','custom'}:raise ValueError('Select a valid reconstruction output option.')
    if mode=='custom':
        value=fields.get('triOutput')
        if not isinstance(value,str) or not value.strip():raise ValueError('Please choose a custom reconstruction output directory, or select Same as 2D JSON directory.')
        dest=Path(value.strip()).expanduser().resolve()
        if dest.exists() and not dest.is_dir():raise ValueError('The custom output path must be a directory.')
    else:dest=source
    if (dest/'tri_new.npy').exists():raise ValueError('tri_new.npy already exists in the output directory. Choose a new custom output directory to keep the existing result.')
    return source,calibration,dest


def reprojection_inputs(fields):
    value=fields.get('reprojectionTri') or fields.get('tri')
    if not value:raise ValueError('Choose a 3D keypoint NPY, or finish filtering above first.')
    source=Path(value).expanduser().resolve()
    if not source.is_file() or source.suffix.lower()!='.npy':raise ValueError('Select an existing 3D keypoint NPY file.')
    videos=required_directory({'videos':fields.get('reprojectionVideos') or fields.get('videos')},'videos','the original video directory')
    missing=[f'data{i}.mp4' for i in range(1,9) if not (videos/f'data{i}.mp4').is_file()]
    if missing:raise ValueError('Missing original videos: '+', '.join(missing))
    calibration=required_directory({'calibration':fields.get('reprojectionCalibration') or fields.get('calibration')},'calibration','the calibration directory')
    if len(list(calibration.glob('*.mat')))!=8:raise ValueError('Reprojection requires eight camera MAT files.')
    unit=fields.get('reprojectionUnit') or 'm';calibration_unit=fields.get('calibrationUnit')
    if unit not in {'m','cm','mm'} or calibration_unit not in {'m','cm','mm'}:raise ValueError('Select the NPY spatial unit and calibration translation unit.')
    dest=Path(fields['reprojectionOutput']).expanduser().resolve() if fields.get('reprojectionOutput') else source.parent/'reprojection'
    if dest.exists() and not dest.is_dir():raise ValueError('Reprojection output must be a directory.')
    if any((dest/f'data{i}.{suffix}').exists() for i in range(1,9) for suffix in ('json','mp4')):raise ValueError('Reprojection outputs already exist. Choose a new output directory.')
    return source,videos,calibration,dest,unit,calibration_unit
