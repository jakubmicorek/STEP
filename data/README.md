# Data

Pose sequences (UBnormal and ShanghaiTech) are obtained from the
[STG-NF](https://github.com/orhir/STG-NF) repository, following its download
instructions. The UBnormal HR boolean masks are obtained from the
[MoCoDAD](https://github.com/aleflabo/MoCoDAD) repository. `UBnormal_labels/` is
included in this repository.

The final `data/` directory structure should look as follows:

```
data/
├── ShanghaiTech/
│   ├── pose/
│   │   ├── train/
│   │   │   ├── 01_001_alphapose_tracked_person.json
│   │   │   ├── 01_001_alphapose-results.json
│   │   │   └── ...
│   │   └── test/
│   │       ├── 01_0014_alphapose_tracked_person.json
│   │       ├── 01_0014_alphapose-results.json
│   │       └── ...
│   └── gt/
│       └── test_frame_mask/
│           ├── 01_0014.npy
│           ├── 01_0015.npy
│           └── ...
├── UBnormal/
│   ├── gt/
│   │   ├── abnormal_scene_10_scenario_1_tracks.txt
│   │   └── ...
│   └── pose/
│       ├── train/
│       │   ├── normal_scene_01_scenario_01_alphapose_tracked_person.json
│       │   └── ...
│       ├── validation/
│       │   ├── normal_scene_10_scenario_01_alphapose_tracked_person.json
│       │   └── ...
│       └── test/
│           ├── abnormal_scene_10_scenario_1_alphapose_tracked_person.json
│           └── ...
├── UBnormal_labels/          ← included in this repository
│   ├── Scene1/
│   │   ├── abnormal_scene_1_scenario_1/
│   │   │   └── labels.npy
│   │   ├── abnormal_scene_1_scenario_2/
│   │   │   └── labels.npy
│   │   └── ...
│   └── Scene29/
│       └── ...
└── UBnormal_hr_bool_masks/   ← obtain from MoCoDAD
    ├── testing/
    │   ├── hr_mask.npy
    │   ├── stats.json
    │   └── test_frame_mask/
    │       ├── abnormal_scene_10_scenario_1.npy
    │       └── ...
    └── validating/
        ├── hr_mask.npy
        ├── stats.json
        └── test_frame_mask/
            └── ...
```

## Attribution

| Resource | Source |
|---|---|
| Pre-extracted AlphaPose poses | [github.com/orhir/STG-NF](https://github.com/orhir/STG-NF)  |
| HR evaluation masks | [github.com/aleflabo/MoCoDAD](https://github.com/aleflabo/MoCoDAD) |
