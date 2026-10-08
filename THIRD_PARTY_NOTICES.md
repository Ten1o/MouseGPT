# Third-party notices

MouseGPT includes code from the following projects. Original copyright notices and license files are retained. The repository's [CC BY-NC-ND 4.0 grant](LICENSE) applies to eligible MouseGPT-authored material and does not replace the licenses of these components.

| Component | Location | Version / upstream commit | License |
| --- | --- | --- | --- |
| [MMPose](https://github.com/open-mmlab/mmpose) | MMPose-derived source in `1_keypoints/`, excluding separately licensed components | 1.0.0 | [Apache-2.0](1_keypoints/LICENSE) |
| [MMYOLO](https://github.com/open-mmlab/mmyolo) | `1_keypoints/mmyolo/` and the additional paths in the [GPL scope](LICENSES/GPL-SCOPE.md) | `8c4d9dc503dc8e327bec8147e8dc97124052f693` | [GPLv3](1_keypoints/mmyolo/LICENSE) |
| [SWIFT](https://github.com/modelscope/ms-swift) | `2_VLM/swift/` | 2.4.0.dev0 · `b5d20a02099e08629c5558d4a7a93528e412a172` | [Apache-2.0](2_VLM/swift/LICENSE) |

MouseGPT adaptations include animal-specific configurations and demos in MMYOLO and dataset registrations in SWIFT. The bundled code may differ from the corresponding upstream versions. MouseGPT modifications and additions to the listed GPL components are distributed under GPLv3 without the CC noncommercial or no-derivatives restrictions. The [GPL scope](LICENSES/GPL-SCOPE.md) identifies the copied configurations outside the MMYOLO directory and documents changes to the bundled distribution.

The pose inferencer can load the MMYOLO detector through MMDetection in the same Python process. The component map describes license coverage; it does not make that integration an independent program. GPLv3 requirements continue to apply to any GPL-covered combined work.

External dependencies, including [EasyMocap](https://github.com/zju3dv/EasyMocap) and base vision-language models, are governed by their respective licenses.
