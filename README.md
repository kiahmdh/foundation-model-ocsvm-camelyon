
This repo tests a classic one-class approach to metastasis detection with modern features.
The method follows the anomaly detection pipeline of Zingman et al. (Boehringer
Ingelheim): learn what normal tissue looks like, then fit a one-class SVM around it. Their
version trains a CNN for this. I replaced the CNN with frozen pathology foundation models
(Prov-GigaPath and PLIP) and asked two questions. How does this compare to a plain kNN
baseline on the same features? And does a boundary learned on one hospital cohort still
work on another?


## Setup

The training data is a bank of about 1.3 million patches from 159 tumour-free CAMELYON16
slides. Features come from the frozen encoder and are L2-normalised. The one-class SVM is
fitted on a 40k-patch subsample of the bank with an equal share per slide, because exact
OC-SVM training scales roughly quadratically with the number of samples. Hyperparameters
(nu, gamma) were tuned on a separate dev split, and the test set was scored once at the end.
The kNN baseline (DN2, k = 5) uses the full bank.

## Results

AUROC on the full CAMELYON16 test set (129 slides). The slide score is the mean of the
top 1% of patch scores.

| Method | Patch | Slide |
|---|---|---|
| kNN on GigaPath (DN2) | 95.2 | 91.7 |
| OC-SVM on GigaPath, tuned | 91.5 | 86.9 |
| OC-SVM on GigaPath, original defaults | 89.4 | 80.2 |
| OC-SVM on PLIP, tuned | 88.4 | 82.7 |

kNN stayed ahead throughout. Part of that gap may come from the training size: kNN sees
the whole bank, while the SVM only sees a 3% subsample. GigaPath features were consistently
better than PLIP. The original defaults (nu = 0.1, gamma = 'scale') were noticeably worse
than tuned values, especially at slide level, so tuning on a dev split matters more than the
reference code suggests. Removing micro-metastasis slides raised every method by a similar
amount, so small lesions are the hard part for all of them.

The cross-cohort test was the most interesting part. I applied the CAMELYON16 model
unchanged to CAMELYON17, which comes from different hospitals. Patch-level AUROC stayed
around 92, so the ranking of patches transfers well. The decision boundary did not: at the
reference implementation's fixed threshold, almost every normal CAMELYON17 patch fell
outside it. Slide-level AUROC on all nodes was close to chance, mostly because ITC and
micro nodes contain very few tumour patches each. In practice, a one-class model like this
needs its threshold recalibrated for each new site, even when its features are good.

## Files

| File | Purpose |
|---|---|
| `extract_features_fm.py` | Frozen GigaPath or PLIP features, resumable, written as a memmap |
| `verify_features.py` | Sanity checks on a feature folder (missing rows, norms, duplicates) |
| `ocsvm_train.py` | Fit the one-class SVM on normal features (exact, or Nyström + SGD for large banks) |
| `ocsvm_score.py` | Score a split, report patch and slide metrics, save worst FP/FN montages |
| `run_bihn.py` | Runs extract, train and score in order, printing each command first |
| `sweep_ocsvm.py` | nu / gamma / training-size sweep on the dev split |
| `make_table.py`, `make_xlsx.py` | Results tables, optionally excluding slide classes (micro, ITC) |
| `train_head_centerloss.py` | Optional: small head trained with center loss, as in the original method |

`ocsvm_score.py` evaluates the RBF kernel with a matrix multiply instead of libsvm's
single-core loop. Scoring a million patches is much faster this way, and the scores match
sklearn's exactly in float64.

## Running it

Each split needs a manifest CSV with columns `path, patch_name, slide, label`.

```bash
pip install -r requirements.txt

# print the commands without running anything
python run_bihn.py --backbone gigapath --query dev --dn2_root data/dn2 --dry_run

# tune on dev
python sweep_ocsvm.py --backbone gigapath --dn2_root data/dn2 --nu 0.02 0.05 0.1 0.2 --gamma scale 0.3 3

# final test run with the chosen settings
python run_bihn.py --backbone gigapath --query test --dn2_root data/dn2 --nu 0.02 --gamma 3

python make_table.py --run results/test_gigapath_nu0.02_n40000 "GigaPath OC-SVM" --title "CAMELYON16"
```

CAMELYON16 and CAMELYON17 are available from their challenge sites. Prov-GigaPath needs
approved access on Hugging Face.

## Limitations

The SVM is trained on a subsample, so the comparison with kNN isn't fully equal. Each
setting was run with one seed. The CAMELYON17 numbers use a threshold refitted on
CAMELYON17, so they show how well the ranking transfers, not out-of-the-box performance.
F1 and recall use a threshold chosen on the evaluation data, so they're optimistic; AUROC
is unaffected. I didn't include results for the center-loss head.

## References

Zingman et al., anomaly detection in histology (Boehringer Ingelheim,
[reference code](https://github.com/Boehringer-Ingelheim/anomaly-detection-in-histology)),
Prov-GigaPath (Xu et al., Nature 2024), PLIP (Huang et al., Nature Medicine 2023),
DN2 (Bergman et al., 2020), CAMELYON16 (Ehteshami Bejnordi et al., JAMA 2017),
CAMELYON17 (Bándi et al., IEEE TMI 2019).

## License

MIT for the code here. Models and datasets keep their own licenses.
