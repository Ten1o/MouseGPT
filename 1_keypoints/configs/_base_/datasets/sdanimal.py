dataset_info = dict(
    dataset_name='sdanimal',
    paper_info=dict(
        author='Yyh ',
        title='none',
        container='.',
        year='2021',
        # homepage='https://github.com/AlexTheBad/AP-10K',
    ),
    

    keypoint_info={
        0:
        dict(
            name='HeadSite', id=0, color=[255, 128, 0], type='upper', swap=''),
        1:
        dict(
            name='Head',
            id=1,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        2:
        dict(name='Neck', id=2, color=[255, 128, 0], type='upper', swap=''),
        3:
        dict(name='Spine1', id=3, color=[255, 128, 0], type='upper', swap=''),
        4:
        dict(
            name='LeftArm',
            id=4,
            color=[0, 255, 0],
            type='upper',
            swap='RightArm'),
        5:
        dict(
            name='LeftForeArm',
            id=5,
            color=[0, 255, 0],
            type='upper',
            swap='RightForeArm'),
        6:
        dict(
            name='LeftHandSite',
            id=6,
            color=[0, 255, 0],
            type='lower',
            swap='RightHandSite'),
        7:
        dict(
            name='RightArm',
            id=7,
            color=[0, 255, 0],
            type='upper',
            swap='LeftArm'),
        8:
        dict(
            name='RightForeArm',
            id=8,
            color=[0, 255, 0],
            type='upper',
            swap='LeftForeArm'),
        9:
        dict(
            name='RightHandSite',
            id=9,
            color=[0, 255, 0],
            type='lower',
            swap='LeftHandSite'),
        10:
        dict(
            name='LeftUpLeg',
            id=10,
            color=[0, 0, 255],
            type='upper',
            swap='RightUpLeg'),
        11:
        dict(
            name='LeftLeg',
            id=11,
            color=[0, 0, 255],
            type='upper',
            swap='RightLeg'),
        12:
        dict(
            name='LeftFootSite',
            id=12,
            color=[0, 0, 255],
            type='lower',
            swap='RightFootSite'),
        13:
        dict(
            name='RightUpLeg',
            id=13,
            color=[0, 0, 255],
            type='upper',
            swap='LeftUpLeg'),
        14:
        dict(
            name='RightLeg', id=14, color=[0, 0, 255], type='upper',
            swap='LeftLeg'),
        15:
        dict(
            name='RightFootSite',
            id=15,
            color=[0, 0, 255],
            type='lower',
            swap='LeftFootSite'),
        16:
        dict(
            name='Tail',
            id=16,
            color=[51, 153, 255],
            type='lower',
            swap=''),
        17:
        dict(
            name='Tail1Site',
            id=17,
            color=[51, 153, 255],
            type='lower',
            swap=''),
        18:
        dict(
            name='TailEnd',
            id=18,
            color=[51, 153, 255],
            type='lower',
            swap=''),
    },
    skeleton_info={
        0: dict(link=('HeadSite', 'Head'), id=0, color=[0, 0, 255]),
        1: dict(link=('Head', 'Neck'), id=1, color=[0, 0, 255]),
        2: dict(link=('Neck', 'Spine1'), id=2, color=[0, 0, 255]),
        3: dict(link=('Neck', 'LeftArm'), id=3, color=[0, 255, 0]),
        4: dict(link=('Neck', 'RightArm'), id=4, color=[0, 255, 0]),
        5: dict(link=('LeftArm', 'LeftForeArm'), id=5, color=[0, 255, 0]),
        6: dict(link=('LeftForeArm', 'LeftHandSite'), id=6, color=[0, 255, 0]),
        7: dict(link=('RightArm', 'RightForeArm'), id=7, color=[0, 255, 0]),
        8: dict(link=('RightForeArm', 'RightHandSite'), id=8, color=[0, 255, 0]),
        9: dict(link=('Spine1', 'Tail'), id=9, color=[0, 0, 255]),
        10: dict(link=('Tail', 'LeftUpLeg'), id=10, color=[6, 156, 250]),
        11: dict(link=('Tail', 'RightUpLeg'), id=11, color=[0, 255, 255]),
        12: dict(link=('Tail', 'Tail1Site'), id=12, color=[0, 255, 255]),
        13: dict(link=('LeftUpLeg', 'LeftLeg'), id=13, color=[6, 156, 250]),
        14: dict(link=('RightUpLeg', 'RightLeg'), id=14, color=[6, 156, 250]),
        15: dict(link=('LeftLeg', 'LeftFootSite'), id=15, color=[6, 156, 250]),
        16: dict(link=('RightLeg', 'RightFootSite'), id=16, color=[6, 156, 250]),
        17: dict(link=('Tail1Site', 'TailEnd'), id=17, color=[6, 156, 250]),
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.2, 1.2,
        1.5, 1.5, 1.5
    ],
    sigmas=[
        0.025, 0.025, 0.026, 0.035, 0.035, 0.10, 0.10, 0.107, 0.107, 0.087, 
        0.087, 0.087, 0.087, 0.089, 0.089, 0.089, 0.089,0.10, 0.107,
    ])