# Data licence

`data/frozen/attacks.csv` and `data/frozen/tag_map.csv` derive from the HAI security dataset (https://github.com/icsdataset/hai, commit `2a814cebc9a66b06c9e5cd545e2d72e65d383737`): the attack tables and the data-point table of its manual, `hai_dataset_technical_details.pdf`, and the column names of its data files.

HAI is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/). These two files are shared under the same licence. Changes from the source: the tables were extracted to CSV, typos were fixed where the data files disagree with the manual (each fix is stated in the `note` column of `attacks.csv` and in REPORT.md), and the tag names were mapped to the 22.04 names.

`demo/data/demo.json` also derives from HAI: it holds the series of 16 check-set windows (one control loop, 300 seconds each), the computed-feature lines computed from them, and counts of model answers on HAI windows. `demo/data/scenes.json` holds six moments of the plant from the check set: six loops each, 300 seconds, with the computed-feature lines and the model answers. `demo/data/results_extra.json` holds counts of answers on HAI test windows. All three are shared under the same licence, CC BY-SA 4.0. Changes from the source, for both window files: the windows were cut from the HAI data files, columns were mapped to the 22.04 names, and each value was rounded to a few significant digits.

`REPORT.md` reproduces the attack table, the name map and per-file counts of the HAI files; it is shared under CC BY-SA 4.0 too.

The result files under `runs/` also derive from HAI. `runs/final/vote_A.jsonl` and `runs/final/vote_B.jsonl` hold, for each test window, the reference answer, the voted answer and the number of votes; no series values. The other files under `runs/final/` and `runs/nebius/` hold counts, tables and notes about answers on HAI windows. The archive `answers.tar.gz` attached to the release holds the answer files of every training run (each row: the instruction, the hint lines computed from the window's series, the reference answer and the model's answer; no series values), the baselines and the big-model comparison (its sample prompt files hold the series of a few windows as text). All of these are shared under the same licence, CC BY-SA 4.0. Changes from the source: windows were cut from the HAI data files, hint lines and reference answers were written by code from the series, and the answers are the output of the models.

The saved models attached to the release were fine-tuned on HAI windows from Llama 3.2 1B. Their use falls under Meta's Llama 3.2 Community License, a copy of which is attached to the release. See `MODELS.md`.

The code of this repository is under the MIT License; see `LICENSE`.

## Citation

The HAI README asks users to cite the following.

HAI 22.04, HAI 23.05, HAIEnd 23.05:

```bibtex
@misc{github,
  author={Shin, Hyeok-Ki; Lee, Woomyo; Choi, Seungoh; Yun, Jeong-Han; and Min, Byung-Gi},
  title={HAI security datasets},
  year={2023},
  url={https://github.com/icsdataset/hai},
}
```

HAI 21.03, HAICon 2020, HAICon 2021:

```bibtex
@inproceedings{10.1145/3474718.3474719,
  author = {Shin, Hyeok-Ki and Lee, Woomyo and Yun, Jeong-Han and Min, Byung-Gi},
  title = {Two ICS Security Datasets and Anomaly Detection Contest on the HIL-Based Augmented ICS Testbed},
  year = {2021},
  isbn = {9781450390651},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3474718.3474719},
  doi = {10.1145/3474718.3474719},
  booktitle = {Cyber Security Experimentation and Test Workshop},
  pages = {36--40},
  numpages = {5},
  location = {Virtual, CA, USA},
  series = {CSET '21}
}
```

HAI 20.07:

```bibtex
@inbook{10.5555/3485754.3485755,
  author = {Shin, Hyeok-Ki and Lee, Woomyo and Yun, Jeong-Han and Kim, HyoungChun},
  title = {HAI 1.0: HIL-Based Augmented ICS Security Dataset},
  year = {2020},
  publisher = {USENIX Association},
  address = {USA},
  booktitle = {Proceedings of the 13th USENIX Conference on Cyber Security Experimentation and Test},
  articleno = {1},
  numpages = {1}
}
```
