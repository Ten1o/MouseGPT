dataset_info = dict(
    dataset_name="mouse_v3",
    paper_info=dict(
        author="Yyh ",
        title="none",
        container=".",
        year="2021",
        homepage="https://github.com/AlexTheBad/AP-10K",
    ),
    keypoint_info={
        0: dict(name="nose", id=0, color=[0, 255, 0], type="upper", swap=""),
        1: dict(
            name="left_ear", id=1, color=[255, 128, 0], type="upper", swap="right_ear"
        ),
        2: dict(
            name="right_ear", id=2, color=[51, 153, 255], type="upper", swap="left_ear"
        ),
        3: dict(name="neck", id=7, color=[0, 255, 0], type="upper", swap=""),
        4: dict(
            name="left_front_limb",
            id=3,
            color=[51, 153, 255],
            type="upper",
            swap="right_front_limb",
        ),
        5: dict(
            name="right_front_limb",
            id=4,
            color=[51, 153, 255],
            type="upper",
            swap="left_front_limb",
        ),
        6: dict(
            name="left_hind_limb",
            id=5,
            color=[51, 153, 255],
            type="upper",
            swap="right_hind_limb",
        ),
        7: dict(
            name="right_hind_limb",
            id=6,
            color=[51, 153, 255],
            type="upper",
            swap="left_hind_limb",
        ),
        8: dict(name="body", id=8, color=[0, 255, 0], type="upper", swap=""),
        9: dict(name="tail_root", id=9, color=[0, 255, 0], type="upper", swap=""),
    },
    skeleton_info={
        0: dict(link=("nose", "left_ear"), id=0, color=[0, 0, 255]),
        1: dict(link=("nose", "right_ear"), id=1, color=[0, 0, 255]),
        2: dict(link=("left_ear", "neck"), id=2, color=[0, 0, 255]),
        3: dict(link=("right_ear", "neck"), id=3, color=[0, 255, 0]),
        4: dict(link=("left_front_limb", "neck"), id=4, color=[0, 255, 0]),
        5: dict(link=("left_front_limb", "left_hind_limb"), id=5, color=[0, 255, 255]),
        6: dict(link=("right_front_limb", "neck"), id=6, color=[0, 255, 255]),
        7: dict(
            link=("right_front_limb", "right_hind_limb"), id=7, color=[6, 156, 250]
        ),
        8: dict(link=("left_hind_limb", "tail_root"), id=8, color=[0, 255, 255]),
        9: dict(link=("right_hind_limb", "tail_root"), id=9, color=[0, 255, 255]),
        10: dict(link=("body", "neck"), id=10, color=[6, 156, 250]),
    },
    joint_weights=[
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.2,
        1.2,
        1.5,
    ],
    sigmas=[
        0.0831,
        0.0277,
        0.0823,
        0.0202,
        0.0263,
        0.0302,
        0.0316,
        0.0547,
        0.0539,
        0.0683,
    ],
)
