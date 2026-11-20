# Data licence

`data/frozen/attacks.csv` and `data/frozen/tag_map.csv` derive from the HAI security dataset (https://github.com/icsdataset/hai, commit `2a814cebc9a66b06c9e5cd545e2d72e65d383737`): the attack tables and the data-point table of its manual, `hai_dataset_technical_details.pdf`, and the column names of its data files.

HAI is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/). These two files are shared under the same licence. Changes from the source: the tables were extracted to CSV, typos were fixed where the data files disagree with the manual (each fix is stated in the `note` column of `attacks.csv` and in REPORT.md), and the tag names were mapped to the 22.04 names.

No licence is granted here for the code.

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
