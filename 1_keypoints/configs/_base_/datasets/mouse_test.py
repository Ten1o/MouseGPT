dataset_info = dict(
    dataset_name='mouse_test',
    paper_info=dict(
        author='Yyh ',
        title='none',
        container='.',
        year='2021',
        homepage='https://github.com/AlexTheBad/AP-10K',
    ),
    
    keypoint_info={
        0:
        dict(
            name='nose', id=0, color=[0, 255, 0], type='upper', swap=''),
        1:
        dict(
            name='left_ear',
            id=1,
            color=[255, 128, 0],
            type='upper',
            swap='right_ear'),
        2:
        dict(name='right_ear', id=2, color=[51, 153, 255], type='upper', swap='left_ear'),
        3:
        dict(name='left_front_limb', id=3, color=[51, 153, 255], type='upper', swap='right_front_limb'),
        4:
        dict(
            name='right_front_limb',
            id=4,
            color=[51, 153, 255],
            type='upper',
            swap='left_front_limb'),
        5:
        dict(
            name='left_hind_limb',
            id=5,
            color=[51, 153, 255],
            type='upper',
            swap='right_hind_limb'),
        6:
        dict(
            name='right_hind_limb',
            id=6,
            color=[51, 153, 255],
            type='upper',
            swap='left_hind_limb'),
        7:
        dict(
            name='left_front_claw',
            id=7,
            color=[0, 255, 0],
            type='lower',
            swap='right_front_claw'),
        8:
        dict(
            name='left_hind_claw',
            id=8,
            color=[0, 255, 0],
            type='lower',
            swap='right_hind_claw'),
        9:
        dict(
            name='right_front_claw',
            id=9,
            color=[255, 128, 0],
            type='lower',
            swap='left_front_claw'),
        10:
        dict(
            name='right_hind_claw',
            id=10,
            color=[0, 255, 0],
            type='lower',
            swap='left_hind_claw'),
        11:
        dict(
            name='body_1',
            id=11,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        12:
        dict(
            name='body_2',
            id=12,
            color=[255, 128, 0],
            type='upper',
            swap=''),
        13:
        dict(
            name='neck',
            id=13,
            color=[0, 255, 0],
            type='upper',
            swap=''),
        14:
        dict(
            name='tail_root', id=14, color=[0, 255, 0], type='lower',
            swap=''),
        15:
        dict(
            name='tail_mid',
            id=15,
            color=[0, 255, 0],
            type='lower',
            swap=''),
        16:
        dict(
            name='tail_tip',
            id=16,
            color=[0, 255, 0],
            type='lower',
            swap=''),
    },
    skeleton_info={
        0: dict(link=('nose', 'left_ear'), id=0, color=[0, 0, 255]),
        1: dict(link=('nose', 'right_ear'), id=1, color=[0, 0, 255]),
        2: dict(link=('left_ear', 'neck'), id=2, color=[0, 0, 255]),
        3: dict(link=('right_ear', 'neck'), id=3, color=[0, 255, 0]),
        4: dict(link=('left_front_limb', 'neck'), id=4, color=[0, 255, 0]),
        5: dict(link=('left_front_limb', 'left_front_claw'), id=5, color=[0, 255, 255]),
        6: dict(link=('left_front_limb', 'left_hind_limb'), id=6, color=[0, 255, 255]),
        7: dict(link=('right_front_limb', 'neck'), id=7, color=[0, 255, 255]),
        8: dict(link=('right_front_limb', 'right_front_claw'), id=8, color=[6, 156, 250]),
        9: dict(link=('right_front_limb', 'right_hind_limb'), id=9, color=[6, 156, 250]),
        10: dict(link=('left_hind_limb', 'left_hind_claw'), id=10, color=[6, 156, 250]),
        11: dict(link=('left_hind_limb', 'tail_root'), id=11, color=[0, 255, 255]),
        12: dict(link=('right_hind_limb', 'tail_root'), id=12, color=[0, 255, 255]),
        13: dict(link=('right_hind_limb', 'right_hind_claw'), id=13, color=[0, 255, 255]),
        14: dict(link=('body_1', 'neck'), id=14, color=[6, 156, 250]),
        15: dict(link=('body_1', 'body_2'), id=15, color=[6, 156, 250]),
        16: dict(link=('body_2', 'tail_root'), id=16, color=[6, 156, 250]),
        17: dict(link=('tail_root', 'tail_mid'), id=17, color=[6, 156, 250]),
        18: dict(link=('tail_mid', 'tail_tip'), id=18, color=[6, 156, 250]),
    },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5,
        1.5
    ],
    sigmas=[
        0.025, 0.025, 0.026, 0.035, 0.035, 0.079, 0.072, 0.062, 0.079, 0.072,
        0.062, 0.107, 0.087, 0.089, 0.107, 0.087, 0.089
    ])