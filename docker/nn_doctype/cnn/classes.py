paths_passport = [
         'passport_main',
         'passport_mestgit',
         'passport_ranee',
         'passport_voen',
         'passport_sempologenie',
         'passport_deti',
         'passport_empty'
         #
         ]
paths_pts = [
    'first',
    'second'
]

paths_other = [
    'photo',  # 2
    'vodit_udostav',  # 3
    'passport_and_vod'  # 4
    # 'unknown'
]

all_classes = [
    'passport',  # 0
    'pts'  # 1
] + paths_other

num_classes = len(paths_passport) + len(paths_other)

siz = 576
