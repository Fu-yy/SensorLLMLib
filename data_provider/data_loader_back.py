# import os
import numpy as np
# import pandas as pd
# import glob
# import re
# import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from utils.timefeatures import time_features
# from data_provider.m4 import M4Dataset, M4Meta
# from data_provider.uea import subsample, interpolate_missing, Normalizer
# from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
# from utils.augmentation import run_augmentation_single
# from datasets import load_dataset
# from huggingface_hub import hf_hub_download
warnings.filterwarnings('ignore')

HUGGINGFACE_REPO = "thuml/Time-Series-Library"


try:
    from scipy.io import loadmat
except Exception as e:
    raise ImportError("Need scipy for .mat loading: pip install scipy") from e


# --------------------------------------- HAR Datasets -------------------------------------------#
###################################################################################################
######################################## HAR Datasets  ############################################
########################################               ############################################
###################################################################################################


class Dataset_MHealth(Dataset):
    """
    mHealth for TSLib classification (ETT-style fast dataset).

    Differences vs your slow version:
      - Read each subject log ONCE in __init__ (cache in memory)
      - __getitem__ only slices cached arrays (no pandas / no disk I/O)
    """

    def __init__(self, args, root_path, flag='train', limit_size=None):
        # flag = 'test'
        flag = flag.lower()
        assert flag in ['train', 'val', 'test']
        self.args = args
        self.root_path = root_path
        self.flag = flag

        # --------- window config ----------
        self.seq_len = int(getattr(args, "seq_len", 100))
        self.stride  = int(getattr(args, "stride", max(1, self.seq_len // 2)))

        # --------- which columns to use (15 dims + label) ----------
        self.feature_cols = [0, 1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19]
        self.label_col = 23

        # --------- split by subject ----------
        default_test = ["subject1", "subject3", "subject6"]
        self.test_subjects = list(getattr(args, "test_subjects", default_test))
        self.val_subjects  = list(getattr(args, "val_subjects", []))
        self.val_ratio     = float(getattr(args, "val_ratio", 0.0))
        self.seed          = int(getattr(args, "seed", 2024))

        self.drop_null   = bool(getattr(args, "drop_null", True))
        self.min_seg_len = int(getattr(args, "min_seg_len", self.seq_len))

        # optional normalization
        self.norm = getattr(args, "mhealth_norm", "none")  # "none" | "global_std"
        self._global_mean = None
        self._global_std  = None

        # ----------------------------
        # 1) Read + cache subjects in memory (ETT-style)
        # ----------------------------
        self._X_cache = {}  # subj_id -> np.ndarray [T,15]
        self._y_cache = {}  # subj_id -> np.ndarray [T]
        self._load_cached_subjects()

        # ----------------------------
        # 2) Build windows index (samples)
        # ----------------------------
        self.samples, self.labels = self._build_index()

        # use_subjects = self._pick_split_subjects()
        # print(f"[split] flag={self.flag} subjects={sorted(list(use_subjects))}")
        # after you computed train/val/test (as sets of subject names)

        if limit_size is not None:
            n = len(self.samples)
            if limit_size <= 1:
                n = int(n * float(limit_size))
            else:
                n = int(limit_size)
            self.samples = self.samples[:n]
            self.labels  = self.labels[:n]
        # uniq, cnt = np.unique(self.labels, return_counts=True)
        # print(
        #     f"[{self.flag}] samples={len(self.samples)}, "
        #     f"class_counts={dict(zip(uniq.tolist(), cnt.tolist()))}"
        # )
        #
        # if hasattr(self, "all_IDs"):
        #     print("len(all_IDs) =", len(self.all_IDs))
        # else:
        #     print("all_IDs not used in this Dataset_MHealth (OK).")

        # ----------------------------
        # 3) Fit normalization on TRAIN split only
        # ----------------------------
        if self.norm == "global_std":
            self._fit_global_std_from_train_subjects()

        # ----------------------------
        # 4) UEA-style compatibility (for your Exp_Classification)
        # ----------------------------
        self.max_seq_len = self.seq_len
        self.feature_df  = pd.DataFrame(columns=[f"f{i}" for i in range(len(self.feature_cols))])
        self.class_names = list(range(12))

    # ============================
    # Subject cache
    # ============================
    def _subject_name(self, subj_id: int) -> str:
        return f"subject{subj_id}"

    def _subject_file(self, subj_id: int) -> str:
        return os.path.join(self.root_path, f"mHealth_subject{subj_id}.log")

    def _load_cached_subjects(self):
        """
        Read all 10 subjects once. This is cheap and removes per-sample disk I/O.
        """
        for subj_id in range(1, 11):
            fp = self._subject_file(subj_id)
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Missing file: {fp}")

            # fast read: numpy is usually faster than pandas for pure numeric logs
            arr = np.loadtxt(fp)  # shape [T, 24]
            X = arr[:, self.feature_cols].astype(np.float32)  # [T, 15]
            y = arr[:, self.label_col].astype(np.int64)       # [T]

            if X.ndim != 2 or X.shape[1] != 15:
                raise RuntimeError(f"Unexpected feature shape {X.shape} for {fp}")
            if len(X) != len(y):
                raise RuntimeError(f"Length mismatch X={len(X)} y={len(y)} for {fp}")

            self._X_cache[subj_id] = X
            self._y_cache[subj_id] = y

    def _get_subject_cached(self, subj_id: int):
        return self._X_cache[subj_id], self._y_cache[subj_id]

    # ============================
    # Split logic
    # ============================
    def _pick_split_subjects(self):
        all_subjects = [self._subject_name(i) for i in range(1, 11)]
        test = set(self.test_subjects)

        remain = [s for s in all_subjects if s not in test]
        if len(remain) == 0:
            raise RuntimeError("All subjects are in test_subjects. No train/val data left.")

        if self.val_subjects:
            val = set(self.val_subjects)
        elif self.val_ratio > 0:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(remain)
            n_val = max(1, int(round(len(remain) * self.val_ratio)))
            val = set(remain[:n_val])
        else:
            val = set()

        train = set([s for s in remain if s not in val])
        # ===== 👇👇👇 就放在这里（return 之前）👇👇👇 =====
        train_set = set(train)
        val_set = set(val)
        test_set = set(test)

        assert len(train_set & val_set) == 0, f"leak train∩val: {train_set & val_set}"
        assert len(train_set & test_set) == 0, f"leak train∩test: {train_set & test_set}"
        assert len(val_set & test_set) == 0, f"leak val∩test: {val_set & test_set}"

        print(f"[split] flag={self.flag} train={sorted(train_set)}")
        print(f"[split] flag={self.flag} val  ={sorted(val_set)}")
        print(f"[split] flag={self.flag} test ={sorted(test_set)}")
        # =====================================================
        if self.flag == "train":
            return train
        if self.flag == "val":
            return val if val else train
        return test

    # ============================
    # Segment + window
    # ============================
    def _continuous_segments(self, y: np.ndarray):
        segs = []
        s = 0
        for t in range(1, len(y)):
            if y[t] != y[t - 1]:
                segs.append((s, t - 1, int(y[t - 1])))
                s = t
        segs.append((s, len(y) - 1, int(y[-1])))

        if self.drop_null:
            segs = [seg for seg in segs if seg[2] != 0]
        segs = [seg for seg in segs if (seg[1] - seg[0] + 1) >= self.min_seg_len]
        return segs

    # def _windowize(self, seg_len: int, label: int):
    #     """
    #     Only uses seg_len, avoids slicing large arrays repeatedly during indexing.
    #     """
    #     L = self.seq_len
    #     S = self.stride
    #     y0 = label - 1
    #     out = []
    #     for ws in range(0, seg_len - L + 1, S):
    #         out.append((ws, ws + L, y0))
    #     return out
    def _windowize(self, seg_len: int, label: int):
        """
        Match your split_sequences():
          - slide with stride
          - if last window doesn't reach the end, add a tail window [seg_len-L, seg_len)
        """
        L = self.seq_len
        S = self.stride
        y0 = label - 1
        out = []

        # seg 太短：本来就不会生成窗口（你上游 min_seg_len 已保证 seg_len>=L，但这里再兜底）
        if seg_len < L:
            return out

        # 1) 完整步长窗口
        num_complete = (seg_len - L) // S + 1  # >= 1
        for i in range(num_complete):
            ws = i * S
            out.append((ws, ws + L, y0))

        # 2) 补尾窗口（避免重复）
        last_end_minus1 = out[-1][1] - 1  # 最后一个窗口覆盖到的最后索引
        if last_end_minus1 < seg_len - 1:  # 没覆盖到尾部
            ws = seg_len - L
            # 如果尾窗和最后一个窗 start 一样，就别重复加
            if ws != out[-1][0]:
                out.append((ws, ws + L, y0))

        return out

    def _build_index(self):
        use_subjects = self._pick_split_subjects()
        samples, labels = [], []

        for subj_id in range(1, 11):
            subj = self._subject_name(subj_id)
            if subj not in use_subjects:
                continue

            X, y = self._get_subject_cached(subj_id)
            segs = self._continuous_segments(y)

            for (s, e, lab) in segs:
                # (optional) strong check
                if not np.all(y[s:e+1] == lab):
                    raise RuntimeError(f"Label not constant in segment {subj} [{s},{e}]")

                seg_len = (e - s + 1)
                wins = self._windowize(seg_len, lab)

                for (ws, we, y0) in wins:
                    samples.append({
                        "subj": subj_id,
                        "seg_s": s,
                        "win_s": ws,
                        "win_e": we,
                    })
                    labels.append(y0)

        if len(samples) == 0:
            raise RuntimeError("No windows built. Check seq_len/stride/min_seg_len/root_path/splits.")

        return samples, labels

    # ============================
    # Normalization (train only)
    # ============================
    def _fit_global_std_from_train_subjects(self):
        """
        Fit mean/std over all TRAIN windows' time steps.
        Uses cached arrays (fast).
        """
        # temporarily compute which subjects are train
        cur_flag = self.flag
        self.flag = "train"
        train_subjects = self._pick_split_subjects()
        self.flag = cur_flag

        xs = []
        for subj_id in range(1, 11):
            if self._subject_name(subj_id) not in train_subjects:
                continue
            X, y = self._get_subject_cached(subj_id)
            segs = self._continuous_segments(y)
            for (s, e, lab) in segs:
                Xseg = X[s:e+1]
                seg_len = Xseg.shape[0]
                for (ws, we, _) in self._windowize(seg_len, lab):
                    xs.append(Xseg[ws:we])
        Xall = np.concatenate(xs, axis=0)  # [sum(L), C]
        self._global_mean = Xall.mean(axis=0, keepdims=True).astype(np.float32)
        self._global_std  = Xall.std(axis=0, keepdims=True).astype(np.float32)
        self._global_std  = np.maximum(self._global_std, 1e-6)

    def _apply_norm(self, x: np.ndarray):
        if self.norm == "global_std":
            return (x - self._global_mean) / self._global_std
        return x

    # ============================
    # PyTorch Dataset
    # ============================
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        meta = self.samples[idx]
        subj_id = meta["subj"]
        X, _ = self._get_subject_cached(subj_id)

        s = meta["seg_s"]
        ws, we = meta["win_s"], meta["win_e"]

        x = X[s + ws : s + we]  # [L, 15] (pure slice)
        if x.shape[0] != self.seq_len:
            raise RuntimeError(f"Window length mismatch got {x.shape[0]} expected {self.seq_len}")

        x = self._apply_norm(x)
        x = torch.from_numpy(x).float()
        y = torch.tensor([self.labels[idx]], dtype=torch.long)
        return x, y


# tttt_dataset.py
# ------------------------------------------------------------
# Final, runnable dataset loaders + sanity-check main()
# - UCIHAR (official split, RAW inertial signals)
# - PAMAP2 (Protocol-only by default; supports optional merge)
# - USC-HAD (robust .mat parsing; supports string activity like "elevator-up")
# - CAPTURE-24 (dictionary-driven 10-class mapping; choose label scheme column)
# ------------------------------------------------------------

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from scipy.io import loadmat


# ============================================================
# Utils
# ============================================================

def _read_txt(fp: str) -> np.ndarray:
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    return np.loadtxt(fp)


def _fill_nan_1d(x: np.ndarray) -> np.ndarray:
    """Linear interpolation + edge fill. All-NaN -> zeros."""
    if not np.isnan(x).any():
        return x
    n = x.shape[0]
    idx = np.arange(n)
    good = ~np.isnan(x)
    if good.sum() == 0:
        return np.zeros_like(x)
    first = idx[good][0]
    last = idx[good][-1]
    x[:first] = x[first]
    x[last + 1:] = x[last]
    bad = ~good
    x[bad] = np.interp(idx[bad], idx[good], x[good])
    return x


def fill_nan_matrix(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    for c in range(X.shape[1]):
        X[:, c] = _fill_nan_1d(X[:, c])
    return X


def _hist_first_n(labels, n=None):
    if n is None:
        arr = np.asarray(labels)
    else:
        arr = np.asarray(labels[:n])
    vals, cnt = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnt)}


def _check_nan_in_first_k(ds: Dataset, k=8):
    k = min(k, len(ds))
    nan_cnt = 0
    for i in range(k):
        x, y = ds[i]
        if torch.isnan(x).any():
            nan_cnt += 1
    return nan_cnt


# ============================================================
# 1) UCIHAR (OFFICIAL split, RAW)
# ============================================================

class Dataset_UCIHAR_Official(Dataset):
    """
    UCI-HAR raw inertial signals loader (OFFICIAL split).

    Uses 6 channels:
      body_acc_{x,y,z} + body_gyro_{x,y,z}

    File structure (root_path):
      train/
        Inertial Signals/*.txt
        y_train.txt
        subject_train.txt
      test/
        Inertial Signals/*.txt
        y_test.txt
        subject_test.txt

    Returns:
      x: [128, 6], y: [1] label in 0..5
    """

    def __init__(self, args, root_path, flag="train", limit_size=None):
        flag = flag.lower()
        assert flag in ["train", "test"]
        self.args = args
        self.root_path = root_path
        self.flag = flag

        self.seq_len = 128
        self._names = [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
        ]

        X, y, subj = self._load_split(flag)

        if limit_size is not None:
            n = len(y)
            n = int(n * float(limit_size)) if limit_size <= 1 else int(limit_size)
            X, y, subj = X[:n], y[:n], subj[:n]

        self._X = X
        self._y = y
        self._subj = subj

        self.max_seq_len = self.seq_len
        self.class_names = list(range(6))

    def _load_split(self, split: str):
        split_dir = os.path.join(self.root_path, split)
        suf = split  # train/test

        y = _read_txt(os.path.join(split_dir, f"y_{suf}.txt")).astype(np.int64).reshape(-1) - 1
        subj = _read_txt(os.path.join(split_dir, f"subject_{suf}.txt")).astype(np.int64).reshape(-1)

        sig_dir = os.path.join(split_dir, "Inertial Signals")
        mats = []
        for n in self._names:
            fp = os.path.join(sig_dir, f"{n}_{suf}.txt")
            mats.append(_read_txt(fp).astype(np.float32))  # [N,128]

        X = np.stack(mats, axis=-1)  # [N,128,6]

        if not (len(X) == len(y) == len(subj)):
            raise RuntimeError(f"Length mismatch: X={len(X)} y={len(y)} subj={len(subj)} in {split_dir}")

        return X, y, subj

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self._X[idx]).float()          # [128,6]
        y = torch.tensor([int(self._y[idx])], dtype=torch.long)
        return x, y


# ============================================================
# 2) PAMAP2 (Protocol-only by default; support Optional merge)
# ============================================================

PAMAP_KEEP_ID2LABEL = {
    1:  "lying",
    2:  "sitting",
    3:  "standing",
    4:  "walking",
    5:  "running",
    6:  "cycling",
    7:  "Nordic walking",
    12: "ascending stairs",
    13: "descending stairs",
    16: "vacuum cleaning",
    17: "ironing",
    24: "rope jumping",
}
PAMAP_ID2NEW = {aid: i for i, aid in enumerate(PAMAP_KEEP_ID2LABEL.keys())}


def pamap27_feature_cols():
    """
    PAMAP2 .dat has 54 cols (1-based in readme):
      1 timestamp
      2 activityID
      3 heart rate
      4-20  hand  (17)
      21-37 chest (17)
      38-54 ankle (17)

    Each IMU block 17 cols:
      1 temp
      2-4  acc16g  (recommended)
      5-7  acc6g
      8-10 gyro
      11-13 mag
      14-17 orientation (invalid)

    We use per IMU: acc16(3) + gyro(3) + mag(3) => 9 dims
    total => 27 dims
    """
    hand0, chest0, ankle0 = 3, 20, 37  # 0-based starts of blocks

    def cols_for_block(b0):
        acc16 = [b0 + 1, b0 + 2, b0 + 3]       # 2-4
        gyro  = [b0 + 7, b0 + 8, b0 + 9]       # 8-10
        mag   = [b0 + 10, b0 + 11, b0 + 12]    # 11-13
        return acc16 + gyro + mag

    return cols_for_block(hand0) + cols_for_block(chest0) + cols_for_block(ankle0)


class Dataset_PAMAP2(Dataset):
    """
    PAMAP2 for TSLib-style HAR (cache + segment + windowize).

    Supports:
      - variant="pamap"   : 100Hz, window default 200 => 2.0s
      - variant="pamap50" : downsample 100->50Hz via ::2, window default 100 => 2.0s

    Folder structure you described:
      root_path/
        Protocol/subject101.dat ...
        Optional/subject101.dat ...

    By default, use Protocol only (paper protocol set). You can enable merge with --pamap_use_optional.
    """

    def __init__(self, args, root_path, flag="train", limit_size=None):
        flag = flag.lower()
        assert flag in ["train", "val", "test"]
        self.args = args
        self.root_path = root_path
        self.flag = flag

        self.variant = getattr(args, "pamap_variant", "pamap50")
        assert self.variant in ["pamap", "pamap50"]
        self.sample_rate = 100 if self.variant == "pamap" else 50
        self.downsample_factor = 1 if self.variant == "pamap" else 2

        # 2s windows consistent across 50/100 Hz defaults
        default_seq = 200 if self.sample_rate == 100 else 100
        default_stride = default_seq // 2

        self.seq_len = int(getattr(args, "seq_len", default_seq))
        self.stride  = int(getattr(args, "stride", default_stride))

        # "original split" in many papers: subject-wise split with fixed test subjects
        default_test = ["subject105", "subject106"]
        self.test_subjects = list(getattr(args, "test_subjects", default_test))
        self.val_subjects  = list(getattr(args, "val_subjects", []))
        self.val_ratio     = float(getattr(args, "val_ratio", 0.0))
        self.seed          = int(getattr(args, "seed", 2024))

        self.min_seg_len = int(getattr(args, "min_seg_len", self.seq_len))
        self.drop_null = True  # drop activityID=0, plus drop not-in-12-classes

        # Use Protocol only (your requirement)
        self.use_optional = bool(getattr(args, "pamap_use_optional", False))

        self.label_col = 1
        self.feature_cols = pamap27_feature_cols()

        # cache
        self._X_cache = {}  # subj -> [T,27]
        self._y_cache = {}  # subj -> [T] 0..11

        self._load_cached_subjects()

        # build window index
        self.samples, self.labels = self._build_index()

        if limit_size is not None:
            n = len(self.samples)
            n = int(n * float(limit_size)) if limit_size <= 1 else int(limit_size)
            self.samples = self.samples[:n]
            self.labels  = self.labels[:n]

        self.max_seq_len = self.seq_len
        self.class_names = list(range(12))

    def _parse_subject_name(self, filename: str):
        m = re.search(r"subject(\d+)", filename)
        if m is None:
            return None
        return f"subject{int(m.group(1))}"

    def _pick_split_subjects(self):
        all_subjects = sorted(list(self._X_cache.keys()))
        test = set(self.test_subjects)
        remain = [s for s in all_subjects if s not in test]
        if len(remain) == 0:
            raise RuntimeError("All subjects in test_subjects. No train left.")

        if self.val_subjects:
            val = set(self.val_subjects)
        elif self.val_ratio > 0:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(remain)
            n_val = max(1, int(round(len(remain) * self.val_ratio)))
            val = set(remain[:n_val])
        else:
            val = set()

        train = set([s for s in remain if s not in val])

        if self.flag == "train":
            return train
        if self.flag == "val":
            return val if len(val) else train
        return test

    def _continuous_segments(self, y: np.ndarray):
        segs = []
        s = 0
        for t in range(1, len(y)):
            if y[t] != y[t - 1]:
                segs.append((s, t - 1, int(y[t - 1])))
                s = t
        segs.append((s, len(y) - 1, int(y[-1])))

        segs = [seg for seg in segs if (seg[1] - seg[0] + 1) >= self.min_seg_len]
        return segs

    def _windowize(self, seg_len: int, label: int):
        L, S = self.seq_len, self.stride
        if seg_len < L:
            return []
        out = []
        num_complete = (seg_len - L) // S + 1
        for i in range(num_complete):
            ws = i * S
            out.append((ws, ws + L, label))
        # tail
        if out[-1][1] - 1 < seg_len - 1:
            ws = seg_len - L
            if ws != out[-1][0]:
                out.append((ws, ws + L, label))
        return out

    def _load_subject_files(self, files):
        chunks_X, chunks_y = [], []

        for fp in sorted(files):
            arr = np.loadtxt(fp)  # [T,54] with NaN possible
            act = arr[:, self.label_col].astype(np.int64)

            X = arr[:, self.feature_cols].astype(np.float32)
            X = fill_nan_matrix(X)

            keep_mask = np.isin(act, np.array(list(PAMAP_ID2NEW.keys()), dtype=np.int64))
            X = X[keep_mask]
            act = act[keep_mask]
            if len(act) == 0:
                continue

            y = np.vectorize(PAMAP_ID2NEW.get, otypes=[np.int64])(act)

            if self.downsample_factor > 1:
                X = X[::self.downsample_factor]
                y = y[::self.downsample_factor]

            if len(y) > 0:
                chunks_X.append(X)
                chunks_y.append(y)

        if len(chunks_y) == 0:
            return None, None

        return np.concatenate(chunks_X, axis=0), np.concatenate(chunks_y, axis=0)

    def _collect_files(self):
        prot_dir = os.path.join(self.root_path, "Protocol")
        opt_dir  = os.path.join(self.root_path, "Optional")

        files = []
        if os.path.isdir(prot_dir):
            files += glob.glob(os.path.join(prot_dir, "**", "*.dat"), recursive=True)
        else:
            # allow passing root directly as Protocol
            files += glob.glob(os.path.join(self.root_path, "**", "*.dat"), recursive=True)

        if self.use_optional and os.path.isdir(opt_dir):
            files += glob.glob(os.path.join(opt_dir, "**", "*.dat"), recursive=True)

        files = sorted(list(set(files)))
        if len(files) == 0:
            raise FileNotFoundError(f"No .dat found under {self.root_path} (Protocol-only={not self.use_optional}).")
        return files

    def _load_cached_subjects(self):
        dat_files = self._collect_files()

        subj_to_files = {}
        for fp in dat_files:
            subj = self._parse_subject_name(os.path.basename(fp))
            if subj is None:
                continue
            subj_to_files.setdefault(subj, []).append(fp)

        if len(subj_to_files) == 0:
            raise RuntimeError("Could not parse subject ids from filenames (need 'subject###' in name).")

        for subj, files in subj_to_files.items():
            Xall, yall = self._load_subject_files(files)
            if Xall is None:
                continue
            self._X_cache[subj] = Xall
            self._y_cache[subj] = yall

        if len(self._X_cache) == 0:
            raise RuntimeError("All subjects empty after filtering. Check mapping / file contents.")

    def _build_index(self):
        use_subjects = self._pick_split_subjects()
        samples, labels = [], []

        for subj in sorted(list(use_subjects)):
            X, y = self._X_cache[subj], self._y_cache[subj]
            segs = self._continuous_segments(y)
            for (s, e, lab) in segs:
                seg_len = e - s + 1
                for (ws, we, y0) in self._windowize(seg_len, lab):
                    samples.append({"subj": subj, "seg_s": s, "win_s": ws, "win_e": we})
                    labels.append(y0)

        if len(samples) == 0:
            raise RuntimeError("No windows built. Check seq_len/stride/min_seg_len/splits.")
        return samples, labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        subj = meta["subj"]
        X = self._X_cache[subj]
        s, ws, we = meta["seg_s"], meta["win_s"], meta["win_e"]
        x = X[s + ws: s + we]
        if x.shape[0] != self.seq_len:
            raise RuntimeError(f"Window length mismatch got {x.shape[0]} expected {self.seq_len}")
        return torch.from_numpy(x).float(), torch.tensor([int(self.labels[idx])], dtype=torch.long)


# ============================================================
# 3) USC-HAD (fix: activity can be string like "elevator-up")
# ============================================================



# =========================
# USC-HAD activity mapping
# =========================
# README activities:
# 1 Walking Forward
# 2 Walking Left
# 3 Walking Right
# 4 Walking Upstairs
# 5 Walking Downstairs
# 6 Running Forward
# 7 Jumping Up
# 8 Sitting
# 9 Standing
# 10 Sleeping
# 11 Elevator Up
# 12 Elevator Down

def _normalize_act_str(s) -> str:
    """
    Normalize activity string from .mat:
      - lower
      - strip spaces
      - unify separators: space/_ -> '-'
      - remove duplicate '-'
    """
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("_", "-").replace(" ", "-")
    s = re.sub(r"-+", "-", s)
    return s


USCHAD_ACTIVITY_STR2ID = {
    # walking forward
    "walking-forward": 0,
    "walk-forward": 0,
    "walkingforward": 0,

    # walking left/right
    "walking-left": 1,
    "walk-left": 1,
    "walkingleft": 1,

    "walking-right": 2,
    "walk-right": 2,
    "walkingright": 2,

    # upstairs/downstairs
    "walking-upstairs": 3,
    "walk-upstairs": 3,
    "walkingupstairs": 3,
    "upstairs": 3,

    "walking-downstairs": 4,
    "walk-downstairs": 4,
    "walkingdownstairs": 4,
    "downstairs": 4,

    # running / jumping
    "running-forward": 5,
    "run-forward": 5,
    "runningforward": 5,

    "jumping-up": 6,
    "jump-up": 6,
    "jumpingup": 6,

    # sit/stand/sleep
    "sitting": 7,
    "sit": 7,

    "standing": 8,
    "stand": 8,

    "sleeping": 9,
    "sleep": 9,

    # elevator up/down (你的数据里出现过 elevator-up)
    "elevator-up": 10,
    "elevatorup": 10,
    "elevator-upstairs": 10,

    "elevator-down": 11,
    "elevatordown": 11,
    "elevator-downstairs": 11,
}


class Dataset_USCHAD(Dataset):
    """
    USC-HAD (MotionNode) loader (TSLib-style).

    Key fixes:
      - activity field might be numeric, numeric string ('10'), or name string ('elevator-up')
      - robustly parse activity_number and fallback to filename "a<m>t<n>.mat"
      - sensor_readings expected [T,6] = acc xyz + gyro xyz

    Split (paper-like default):
      - test subjects: 13, 14
    Windowing:
      - 100Hz -> seq_len=200, stride=100 (2s, 50% overlap)
    """

    def __init__(self, args, root_path, flag="train", limit_size=None):
        flag = flag.lower()
        assert flag in ["train", "val", "test"]
        self.args = args
        self.root_path = root_path
        self.flag = flag

        self.seq_len = int(getattr(args, "seq_len", 200))
        self.stride  = int(getattr(args, "stride", 100))

        default_test = ["subject13", "subject14"]
        self.test_subjects = list(getattr(args, "test_subjects", default_test))
        self.val_subjects  = list(getattr(args, "val_subjects", []))
        self.val_ratio     = float(getattr(args, "val_ratio", 0.0))
        self.seed          = int(getattr(args, "seed", 2024))

        self.min_seg_len = int(getattr(args, "min_seg_len", self.seq_len))

        self._X_cache = []       # list[np.ndarray] each [T,6]
        self._y_cache = []       # list[int] y0 in 0..11
        self._subj_cache = []    # list[str]

        self.samples = []        # list[(trial_idx, ws, we)]
        self.labels = []         # list[int]

        self._load_and_index()

        if limit_size is not None:
            n = len(self.samples)
            n = int(n * float(limit_size)) if limit_size <= 1 else int(limit_size)
            self.samples = self.samples[:n]
            self.labels  = self.labels[:n]

        self.max_seq_len = self.seq_len
        self.class_names = list(range(12))

    # ---------------- split ----------------
    def _pick_split_subjects(self, all_subjects_sorted):
        test = set(self.test_subjects)
        remain = [s for s in all_subjects_sorted if s not in test]
        if len(remain) == 0:
            raise RuntimeError("All subjects in test_subjects. No train left.")

        if self.val_subjects:
            val = set(self.val_subjects)
        elif self.val_ratio > 0:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(remain)
            n_val = max(1, int(round(len(remain) * self.val_ratio)))
            val = set(remain[:n_val])
        else:
            val = set()

        train = set([s for s in remain if s not in val])

        if self.flag == "train":
            return train
        if self.flag == "val":
            return val if len(val) else train
        return test

    # ---------------- windowing ----------------
    def _windowize(self, T: int):
        L, S = self.seq_len, self.stride
        if T < L:
            return []
        out = []
        num_complete = (T - L) // S + 1
        for i in range(num_complete):
            ws = i * S
            out.append((ws, ws + L))
        if out[-1][1] - 1 < T - 1:
            ws = T - L
            if ws != out[-1][0]:
                out.append((ws, ws + L))
        return out

    # ---------------- mat parsing helpers ----------------
    def _get_scalar(self, obj):
        if isinstance(obj, np.ndarray) and obj.size == 1:
            return obj.reshape(-1)[0]
        return obj

    def _try_get_field(self, d, keys):
        for k in keys:
            if k in d:
                return d[k]
        return None

    def _as_int_if_possible(self, v):
        """
        Convert v to int if it looks like an integer:
          - numeric types
          - numeric string: '10', '10.0'
          - matlab char arrays sometimes become numpy.str_ / object
        Return: int or None
        """
        if v is None:
            return None

        # already numeric
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)) and np.isfinite(v):
            # if it's an integer-valued float
            iv = int(v)
            if abs(v - iv) < 1e-6:
                return iv
            return None

        # strings (including numpy.str_)
        if isinstance(v, (str, np.str_)):
            s = str(v).strip()
            # pure int
            if re.fullmatch(r"[+-]?\d+", s):
                return int(s)
            # float but integer-valued (e.g. '10.0')
            if re.fullmatch(r"[+-]?\d+\.\d+", s):
                try:
                    f = float(s)
                    iv = int(f)
                    if abs(f - iv) < 1e-6:
                        return iv
                except Exception:
                    return None
            return None

        # sometimes matlab char array comes as ndarray of dtype '<U1' etc
        if isinstance(v, np.ndarray):
            # if it's a char array -> join
            if v.dtype.kind in ("U", "S"):
                s = "".join(v.reshape(-1).tolist()).strip()
                if re.fullmatch(r"[+-]?\d+", s):
                    return int(s)
                if re.fullmatch(r"[+-]?\d+\.\d+", s):
                    try:
                        f = float(s)
                        iv = int(f)
                        if abs(f - iv) < 1e-6:
                            return iv
                    except Exception:
                        return None
            return None

        return None

    def _parse_activity(self, fp, d):
        """
        Return y0 in 0..11
        Priority:
          1) activity_number (numeric or numeric string)
          2) activity name string mapping
          3) filename a<m>t<n>.mat
        """
        # try numeric activity_number first
        act_obj = self._try_get_field(d, [
            "activity_number", "activity_num", "activityNumber",
            "activityID", "activity_id",
            # some dumps store numeric in 'activity'
            "activity",
        ])

        if act_obj is not None:
            act_val = self._get_scalar(act_obj)
            act_int = self._as_int_if_possible(act_val)
            if act_int is not None:
                y0 = act_int - 1
                return y0

        # try activity name string
        name_obj = self._try_get_field(d, [
            "activity_name", "activityName", "activity_label", "activityLabel",
            # sometimes 'activity' is a string name
            "activity",
        ])
        if name_obj is not None:
            name_val = self._get_scalar(name_obj)

            # IMPORTANT: name_val might actually be numeric-string like '10'
            act_int = self._as_int_if_possible(name_val)
            if act_int is not None:
                return act_int - 1

            key = _normalize_act_str(name_val)
            if key in USCHAD_ACTIVITY_STR2ID:
                return int(USCHAD_ACTIVITY_STR2ID[key])

        # fallback to filename a<m>t<n>.mat
        base = os.path.basename(fp).lower()
        m = re.search(r"a(\d+)", base)
        if m:
            act_int = int(m.group(1))
            return act_int - 1

        raise RuntimeError(f"Cannot infer activity from fields or filename: {fp}")

    def _parse_subject(self, fp, d):
        # subject field
        subj = None
        subj_obj = self._try_get_field(d, ["subject", "subject_number", "subject_num", "subjectID", "subject_id"])
        if subj_obj is not None:
            try:
                subj_val = self._get_scalar(subj_obj)
                subj_int = self._as_int_if_possible(subj_val)
                if subj_int is not None:
                    subj = subj_int
            except Exception:
                subj = None

        # infer from path if missing
        if subj is None:
            mm = re.search(r"subject(\d+)", fp.lower())
            if mm:
                subj = int(mm.group(1))
            else:
                subj = 0
        return subj

    def _load_sensor_readings(self, fp, d):
        X = None
        X_obj = self._try_get_field(d, ["sensor_readings", "sensorReadings", "readings", "sensor", "sensor_reading"])
        if isinstance(X_obj, np.ndarray) and X_obj.ndim == 2 and X_obj.shape[1] == 6:
            X = X_obj.astype(np.float32)

        if X is None:
            # wrapped in struct "data" or similar
            for _, v in d.items():
                if hasattr(v, "__dict__") and hasattr(v, "sensor_readings"):
                    arr = getattr(v, "sensor_readings")
                    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 6:
                        X = arr.astype(np.float32)
                        break

        if X is None:
            raise RuntimeError(f"Cannot find sensor_readings [T,6] in {fp}. Keys={list(d.keys())[:30]}")
        return X

    def _load_one_mat(self, fp: str):
        d = loadmat(fp, squeeze_me=True, struct_as_record=False)

        subj = self._parse_subject(fp, d)
        y0 = self._parse_activity(fp, d)

        if not (0 <= y0 <= 11):
            raise RuntimeError(f"Unexpected activity id y0={y0} in {fp}")

        X = self._load_sensor_readings(fp, d)
        return f"subject{subj}", X, y0

    # ---------------- main load/index ----------------
    def _load_and_index(self):
        mat_files = sorted(glob.glob(os.path.join(self.root_path, "**", "*.mat"), recursive=True))
        if len(mat_files) == 0:
            raise FileNotFoundError(f"No .mat files found under: {self.root_path}")

        meta = []
        subj_set = set()
        for fp in mat_files:
            subj_name, X, y0 = self._load_one_mat(fp)
            subj_set.add(subj_name)
            meta.append((subj_name, X, y0))

        all_subjects_sorted = sorted(list(subj_set))
        keep_subjects = self._pick_split_subjects(all_subjects_sorted)

        for (subj_name, X, y0) in meta:
            if subj_name not in keep_subjects:
                continue
            T = int(X.shape[0])
            if T < self.min_seg_len:
                continue

            trial_idx = len(self._X_cache)
            self._X_cache.append(X)
            self._y_cache.append(y0)
            self._subj_cache.append(subj_name)

            for (ws, we) in self._windowize(T):
                self.samples.append((trial_idx, ws, we))
                self.labels.append(y0)

        if len(self.samples) == 0:
            raise RuntimeError("No windows built. Check seq_len/stride or data length.")

    # ---------------- torch dataset ----------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        trial_idx, ws, we = self.samples[idx]
        X = self._X_cache[trial_idx]
        x = X[ws:we]
        if x.shape[0] != self.seq_len:
            raise RuntimeError(f"Window length mismatch got {x.shape[0]} expected {self.seq_len}")
        return torch.from_numpy(x).float(), torch.tensor([int(self.labels[idx])], dtype=torch.long)


# ============================================================
# 4) CAPTURE-24 (dictionary + choose label scheme column)
# ============================================================

CAPTURE24_ID2LABEL = {
    0: 'sleep',
    1: 'sitting',
    2: 'household-chores',
    3: 'walking',
    4: 'vehicle',
    5: 'bicycling',
    6: 'mixed-activity',
    7: 'standing',
    8: 'manual-work',
    9: 'sports',
}
CAPTURE24_LABEL2ID = {v: k for k, v in CAPTURE24_ID2LABEL.items()}


class Dataset_CAPTURE24(Dataset):
    """
    CAPTURE-24 loader (10-class) based on your folder:

      root_path/
        annotation-label-dictionary.csv
        metadata.csv
        P001.csv.gz ... P151.csv.gz

    Key points (important fix vs your previous draft):
      - The participant files contain "annotation" column (fine-grained codes/phrases).
      - annotation-label-dictionary.csv maps annotation -> multiple label scheme columns.
      - You MUST choose which scheme to use, e.g. "label:WillettsSpecific2018" (10 classes).
      - Then: annotation -> coarse_label_str -> id (0..9).

    Defaults:
      - downsample 100->50Hz via ::2 (so seq_len=500 => 10s if original 100Hz; 5s at 50Hz)
      - You can tune seq_len/stride to match the exact paper you follow.
    """

    def __init__(self, args, root_path, flag="train", limit_size=None):
        flag = flag.lower()
        assert flag in ["train", "val", "test"]
        self.args = args
        self.root_path = root_path
        self.flag = flag

        self.seq_len = int(getattr(args, "seq_len", 500))
        self.stride  = int(getattr(args, "stride", 250))

        # split: first 100 train, remaining 51 test
        self.train_n = int(getattr(args, "capture24_train_n", 100))
        self.test_n  = int(getattr(args, "capture24_test_n", 51))

        # keep 5% windows per participant (you can set 1.0 to keep all)
        self.keep_ratio = float(getattr(args, "capture24_keep_ratio", 0.05))
        self.seed = int(getattr(args, "seed", 2024))

        # downsample factor (2 => 100Hz->50Hz)
        self.downsample_factor = int(getattr(args, "downsample_factor", 2))

        # choose label scheme column from dictionary
        # You SHOULD use: "label:WillettsSpecific2018" for your 10-class config.
        self.label_scheme_col = str(getattr(args, "capture24_label_col", "label:WillettsSpecific2018"))

        # load mapping: annotation -> coarse_label_id
        self.ann2id = self._load_dictionary(
            os.path.join(root_path, "annotation-label-dictionary.csv"),
            scheme_col=self.label_scheme_col,
        )

        # load participants (cache)
        self._cache = {}  # pid -> (X[T,3], ann[T] string)
        self._load_subjects()

        # build window index with majority label
        self.samples, self.labels = self._build_index()

        if limit_size is not None:
            n = len(self.samples)
            n = int(n * float(limit_size)) if limit_size <= 1 else int(limit_size)
            self.samples = self.samples[:n]
            self.labels  = self.labels[:n]

        self.max_seq_len = self.seq_len
        self.class_names = list(range(10))

    # ---------------- split ----------------
    def _list_participants(self):
        files = sorted([f for f in os.listdir(self.root_path) if re.fullmatch(r"P\d{3}\.csv\.gz", f)])
        if len(files) == 0:
            raise FileNotFoundError("No P###.csv.gz found under root_path.")
        pids = [f.split(".")[0] for f in files]
        return pids, files

    def _pick_split_pids(self, pids_sorted):
        train = pids_sorted[:self.train_n]
        test = pids_sorted[self.train_n:self.train_n + self.test_n]
        if self.flag in ["train", "val"]:
            return set(train)
        return set(test)

    # ---------------- dictionary ----------------
    def _load_dictionary(self, fp, scheme_col: str):
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        df = pd.read_csv(fp)

        if "annotation" not in df.columns:
            raise RuntimeError(f"Dictionary must contain 'annotation' column. Got columns={list(df.columns)}")

        if scheme_col not in df.columns:
            raise RuntimeError(
                f"Dictionary missing scheme_col='{scheme_col}'. "
                f"Available={list(df.columns)}"
            )

        ann = df["annotation"].astype(str).str.strip()
        coarse = df[scheme_col].astype(str).str.strip()

        # coarse labels must be in our 10-class set
        ann2id = {}
        bad = 0
        for a, c in zip(ann.tolist(), coarse.tolist()):
            if c in CAPTURE24_LABEL2ID:
                ann2id[a] = CAPTURE24_LABEL2ID[c]
            else:
                bad += 1

        if len(ann2id) == 0:
            raise RuntimeError(
                f"Dictionary mapping produced 0 valid entries for scheme_col='{scheme_col}'. "
                f"Check whether this scheme matches your 10-class config."
            )
        # It's okay if some rows are not in 10-class (they'll be ignored)
        return ann2id

    # ---------------- read participant ----------------
    def _detect_columns(self, df):
        cols = list(df.columns)
        low = [c.lower() for c in cols]

        def find_one(cands):
            for cand in cands:
                if cand in low:
                    return cols[low.index(cand)]
            return None

        tcol = find_one(["time", "timestamp", "t", "unix_time", "datetime", "date_time"])
        ax = find_one(["x", "acc_x", "ax", "accel_x", "acceleration_x"])
        ay = find_one(["y", "acc_y", "ay", "accel_y", "acceleration_y"])
        az = find_one(["z", "acc_z", "az", "accel_z", "acceleration_z"])
        anncol = find_one(["annotation", "label", "activity", "class", "category"])
        return tcol, (ax, ay, az), anncol

    def read_one_participant(self, pid):
        fp = os.path.join(self.root_path, f"{pid}.csv.gz")

        # 先只读表头，检测列名
        head = pd.read_csv(fp, compression="gzip", nrows=0)
        cols = list(head.columns)
        low = [c.lower() for c in cols]

        def find_one(cands):
            for cand in cands:
                if cand in low:
                    return cols[low.index(cand)]
            return None

        tcol = find_one(["time", "timestamp", "t", "unix_time", "datetime", "date_time"])
        ax = find_one(["x", "acc_x", "ax", "accel_x", "acceleration_x"])
        ay = find_one(["y", "acc_y", "ay", "accel_y", "acceleration_y"])
        az = find_one(["z", "acc_z", "az", "accel_z", "acceleration_z"])
        anncol = find_one(["annotation", "label", "activity", "class", "category"])

        if ax is None or ay is None or az is None:
            raise RuntimeError(f"{pid}: cannot detect accel cols. header={cols[:30]}")
        if anncol is None:
            raise RuntimeError(f"{pid}: cannot detect annotation/label col. header={cols[:30]}")

        usecols = [ax, ay, az, anncol]
        if tcol is not None:
            usecols = [tcol] + usecols

        # 关键：只读 usecols + 明确 dtype，避免 mixed types warning
        dtype_map = {ax: "float32", ay: "float32", az: "float32", anncol: "string"}
        if tcol is not None:
            # time 可能是 int/float/string，先用 string 读，后面排序时再 try 转
            dtype_map[tcol] = "string"

        df = pd.read_csv(
            fp,
            compression="gzip",
            usecols=usecols,
            dtype=dtype_map,
            low_memory=False,
        )

        X = df[[ax, ay, az]].to_numpy(dtype=np.float32)
        ann = df[anncol].astype("string").fillna("").str.strip().to_numpy()

        # sort by time if possible
        if tcol is not None:
            try:
                t = pd.to_numeric(df[tcol], errors="coerce").to_numpy()
                order = np.argsort(t)
                X = X[order]
                ann = ann[order]
            except Exception:
                pass

        # downsample
        if self.downsample_factor > 1:
            X = X[::self.downsample_factor]
            ann = ann[::self.downsample_factor]

        return X, ann

    def _load_subjects(self):
        pids, _ = self._list_participants()
        keep = self._pick_split_pids(pids)

        for pid in pids:
            if pid not in keep:
                continue
            X, ann = self.read_one_participant(pid)
            if X.ndim != 2 or X.shape[1] != 3:
                raise RuntimeError(f"{pid}: X must be [T,3], got {X.shape}")
            if ann.ndim != 1 or len(ann) != len(X):
                raise RuntimeError(f"{pid}: ann must be [T] match X. got ann={ann.shape} X={X.shape}")
            self._cache[pid] = (X, ann)

        if len(self._cache) == 0:
            raise RuntimeError(f"No participants loaded for flag={self.flag}. Check split params.")

    # ---------------- windows ----------------
    def _windowize(self, T):
        L, S = self.seq_len, self.stride
        if T < L:
            return []
        out = []
        num = (T - L) // S + 1
        for i in range(num):
            ws = i * S
            out.append((ws, ws + L))
        if out[-1][1] - 1 < T - 1:
            ws = T - L
            if ws != out[-1][0]:
                out.append((ws, ws + L))
        return out

    def _majority_label(self, ann_win):
        # annotation -> id using ann2id; ignore unknown
        ids = []
        for a in ann_win:
            if a in self.ann2id:
                ids.append(self.ann2id[a])
        if len(ids) == 0:
            return None
        vals, cnt = np.unique(np.asarray(ids, dtype=np.int64), return_counts=True)
        return int(vals[np.argmax(cnt)])

    def _build_index(self):
        rng = np.random.RandomState(self.seed)
        samples, labels = [], []

        for pid, (X, ann) in self._cache.items():
            wins = self._windowize(len(X))
            if len(wins) == 0:
                continue

            # keep ratio per pid
            if self.keep_ratio >= 1.0:
                pick_idx = np.arange(len(wins))
            else:
                k = max(1, int(round(len(wins) * self.keep_ratio)))
                pick_idx = rng.choice(len(wins), size=k, replace=False)
                pick_idx = np.sort(pick_idx)

            for j in pick_idx:
                ws, we = wins[j]
                lab = self._majority_label(ann[ws:we])
                if lab is None:
                    continue
                if not (0 <= lab <= 9):
                    continue
                samples.append((pid, ws, we))
                labels.append(lab)

        if len(samples) == 0:
            raise RuntimeError(
                "No windows built. Possible reasons:\n"
                "1) chosen label scheme col does not match 10-class mapping\n"
                "2) annotation strings mismatch (whitespace etc)\n"
                "3) keep_ratio too small\n"
            )
        return samples, labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, ws, we = self.samples[idx]
        X, _ = self._cache[pid]
        x = X[ws:we]
        if x.shape[0] != self.seq_len:
            raise RuntimeError(f"{pid}: window mismatch {x.shape[0]} vs {self.seq_len}")
        return torch.from_numpy(x).float(), torch.tensor([int(self.labels[idx])], dtype=torch.long)

# ============================================================
# 5) Offline cache: export windows to NPZ + fast NPZ dataset
# ============================================================

class Dataset_FromNPZ(Dataset):
    """
    Load pre-windowized samples from .npz:
      X: [N, L, C] float32
      y: [N] int64
    __getitem__ returns:
      x: [L, C], y: [1]
    """
    def __init__(self, npz_path: str, limit_size=None):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(npz_path)
        data = np.load(npz_path, allow_pickle=False)
        self.X = data["X"].astype(np.float32)
        self.y = data["y"].astype(np.int64)

        if limit_size is not None:
            n = len(self.y)
            n = int(n * float(limit_size)) if limit_size <= 1 else int(limit_size)
            self.X = self.X[:n]
            self.y = self.y[:n]

        self.seq_len = self.X.shape[1]
        self.max_seq_len = self.seq_len
        self.class_names = sorted(list(set(self.y.tolist())))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor([int(self.y[idx])], dtype=torch.long)
        return x, y


def export_dataset_to_npz(ds: Dataset, out_path: str, max_items=None, verbose=1):
    """
    Export a dataset that yields (x[L,C], y[1]) into a single npz:
      X: [N,L,C] float32
      y: [N] int64
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    N = len(ds) if max_items is None else min(len(ds), int(max_items))
    x0, y0 = ds[0]
    L, C = int(x0.shape[0]), int(x0.shape[1])

    X = np.empty((N, L, C), dtype=np.float32)
    y = np.empty((N,), dtype=np.int64)

    for i in range(N):
        xi, yi = ds[i]
        xi = xi.detach().cpu().numpy().astype(np.float32)
        yi = int(yi.detach().cpu().numpy().reshape(-1)[0])
        if xi.shape != (L, C):
            raise RuntimeError(f"Shape mismatch at i={i}: got {xi.shape}, expected {(L,C)}")
        X[i] = xi
        y[i] = yi
        if verbose and (i + 1) % 2000 == 0:
            print(f"[export] {out_path}: {i+1}/{N}")

    np.savez_compressed(out_path, X=X, y=y)

    if verbose:
        hh = _hist_first_n(y, n=min(len(y), 50000))
        print(f"[export] saved: {out_path}")
        print(f"         X={X.shape} y={y.shape} label_hist={hh}")


def build_capture24_npz_cache(args, out_dir: str, max_items=None):
    """
    Build CAPTURE-24 train/test cache:
      out_dir/
        capture24_train.npz
        capture24_test.npz
    """
    os.makedirs(out_dir, exist_ok=True)

    # train
    cap_tr_args = argparse.Namespace(**vars(args))
    if cap_tr_args.seq_len is None:
        cap_tr_args.seq_len = 500
    if cap_tr_args.stride is None:
        cap_tr_args.stride = 250

    ds_tr = Dataset_CAPTURE24(cap_tr_args, cap_tr_args.capture24_root, flag="train")
    export_dataset_to_npz(ds_tr, os.path.join(out_dir, "capture24_train.npz"), max_items=max_items)

    # test
    ds_te = Dataset_CAPTURE24(cap_tr_args, cap_tr_args.capture24_root, flag="test")
    export_dataset_to_npz(ds_te, os.path.join(out_dir, "capture24_test.npz"), max_items=max_items)

def build_ucihar_npz_cache(args, out_dir: str, max_items=None):
    """
    Build UCIHAR official split cache:
      out_dir/
        ucihar_train.npz
        ucihar_test.npz
    """
    os.makedirs(out_dir, exist_ok=True)

    uc_args = argparse.Namespace(**vars(args))
    # UCIHAR 固定 seq_len=128, stride 不重要，因为它是现成 sample

    ds_tr = Dataset_UCIHAR_Official(uc_args, uc_args.ucihar_root, flag="train")
    export_dataset_to_npz(ds_tr, os.path.join(out_dir, "ucihar_train.npz"), max_items=max_items)

    ds_te = Dataset_UCIHAR_Official(uc_args, uc_args.ucihar_root, flag="test")
    export_dataset_to_npz(ds_te, os.path.join(out_dir, "ucihar_test.npz"), max_items=max_items)

def build_pamap2_npz_cache(args, out_dir: str, max_items=None):
    """
    Build PAMAP2 protocol train/test cache:
      out_dir/
        pamap2_train_{variant}.npz
        pamap2_test_{variant}.npz
    """
    os.makedirs(out_dir, exist_ok=True)

    pamap_args = argparse.Namespace(**vars(args))
    if pamap_args.seq_len is None:
        pamap_args.seq_len = 200 if pamap_args.pamap_variant == "pamap" else 100
    if pamap_args.stride is None:
        pamap_args.stride = pamap_args.seq_len // 2

    ds_tr = Dataset_PAMAP2(pamap_args, pamap_args.pamap_root, flag="train")
    export_dataset_to_npz(ds_tr, os.path.join(out_dir, f"pamap2_train_{pamap_args.pamap_variant}.npz"), max_items=max_items)

    ds_te = Dataset_PAMAP2(pamap_args, pamap_args.pamap_root, flag="test")
    export_dataset_to_npz(ds_te, os.path.join(out_dir, f"pamap2_test_{pamap_args.pamap_variant}.npz"), max_items=max_items)

def build_pamap2_npz_cache_both(args, out_dir: str, max_items=None):
    """
    Build PAMAP2 caches for BOTH variants:
      out_dir/
        pamap2_train_pamap.npz
        pamap2_test_pamap.npz
        pamap2_train_pamap50.npz
        pamap2_test_pamap50.npz
    """
    os.makedirs(out_dir, exist_ok=True)

    for variant in ["pamap", "pamap50"]:
        pamap_args = argparse.Namespace(**vars(args))
        pamap_args.pamap_variant = variant

        # default windows keep 2s
        if getattr(pamap_args, "seq_len", None) is None:
            pamap_args.seq_len = 200 if variant == "pamap" else 100
        if getattr(pamap_args, "stride", None) is None:
            pamap_args.stride = pamap_args.seq_len // 2

        ds_tr = Dataset_PAMAP2(pamap_args, pamap_args.pamap_root, flag="train")
        export_dataset_to_npz(ds_tr, os.path.join(out_dir, f"pamap2_train_{variant}.npz"), max_items=max_items)

        ds_te = Dataset_PAMAP2(pamap_args, pamap_args.pamap_root, flag="test")
        export_dataset_to_npz(ds_te, os.path.join(out_dir, f"pamap2_test_{variant}.npz"), max_items=max_items)

def build_uschad_npz_cache(args, out_dir: str, max_items=None):
    """
    Build USC-HAD train/test cache:
      out_dir/
        uschad_train.npz
        uschad_test.npz
    """
    os.makedirs(out_dir, exist_ok=True)

    us_args = argparse.Namespace(**vars(args))
    if us_args.seq_len is None:
        us_args.seq_len = 200
    if us_args.stride is None:
        us_args.stride = 100

    ds_tr = Dataset_USCHAD(us_args, us_args.uschad_root, flag="train")
    export_dataset_to_npz(ds_tr, os.path.join(out_dir, "uschad_train.npz"), max_items=max_items)

    ds_te = Dataset_USCHAD(us_args, us_args.uschad_root, flag="test")
    export_dataset_to_npz(ds_te, os.path.join(out_dir, "uschad_test.npz"), max_items=max_items)

def build_mhealth_npz_cache(args, out_dir: str, max_items=None):
    """
    Build MHealth train/val/test cache:
      out_dir/
        mhealth_train.npz
        mhealth_val.npz
        mhealth_test.npz
    """
    os.makedirs(out_dir, exist_ok=True)

    mh_args = argparse.Namespace(**vars(args))
    # 给 mhealth 的默认 window（你也可以走 args.seq_len/stride）
    if getattr(mh_args, "seq_len", None) is None:
        mh_args.seq_len = 200
    if getattr(mh_args, "stride", None) is None:
        mh_args.stride = mh_args.seq_len // 2

    # train/val/test
    ds_tr = Dataset_MHealth(mh_args, mh_args.mhealth_root, flag="train")
    export_dataset_to_npz(ds_tr, os.path.join(out_dir, "mhealth_train.npz"), max_items=max_items)

    ds_va = Dataset_MHealth(mh_args, mh_args.mhealth_root, flag="val")
    export_dataset_to_npz(ds_va, os.path.join(out_dir, "mhealth_val.npz"), max_items=max_items)

    ds_te = Dataset_MHealth(mh_args, mh_args.mhealth_root, flag="test")
    export_dataset_to_npz(ds_te, os.path.join(out_dir, "mhealth_test.npz"), max_items=max_items)

import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed

def _capture24_detect_cols_fast(fp_gz: str):
    # 用 gzip + csv 读一行 header，比 pandas nrows=0 更快
    import csv
    with gzip.open(fp_gz, "rt") as f:
        cols = next(csv.reader(f))
    low = [c.lower() for c in cols]

    def find_one(cands):
        for cand in cands:
            if cand in low:
                return cols[low.index(cand)]
        return None

    tcol = find_one(["time", "timestamp", "t", "unix_time", "datetime", "date_time"])
    ax = find_one(["x", "acc_x", "ax", "accel_x", "acceleration_x"])
    ay = find_one(["y", "acc_y", "ay", "accel_y", "acceleration_y"])
    az = find_one(["z", "acc_z", "az", "accel_z", "accel_z", "acceleration_z"])
    anncol = find_one(["annotation", "label", "activity", "class", "category"])
    if ax is None or ay is None or az is None or anncol is None:
        raise RuntimeError(f"bad header in {os.path.basename(fp_gz)} cols={cols[:30]}")
    return tcol, ax, ay, az, anncol

def _capture24_process_one(args):
    """
    单个参与者：读 gz -> X[T,3], ann_id[T] -> 写 per-pid npz
    """
    root_path, pid, downsample_factor, ann2id, out_dir, use_compressed = args
    fp = os.path.join(root_path, f"{pid}.csv.gz")

    tcol, ax, ay, az, anncol = _capture24_detect_cols_fast(fp)

    usecols = [ax, ay, az, anncol] + ([tcol] if tcol is not None else [])
    dtype_map = {ax: "float32", ay: "float32", az: "float32", anncol: "string"}
    if tcol is not None:
        dtype_map[tcol] = "string"

    # 读 gz：用 gzip.open + pandas
    with gzip.open(fp, "rt") as f:
        df = pd.read_csv(f, usecols=usecols, dtype=dtype_map, low_memory=False)

    X = df[[ax, ay, az]].to_numpy(dtype=np.float32)
    ann = df[anncol].astype("string").fillna("").str.strip().to_numpy()

    # sort by time
    if tcol is not None:
        try:
            t = pd.to_numeric(df[tcol], errors="coerce").to_numpy()
            order = np.argsort(t)
            X = X[order]
            ann = ann[order]
        except Exception:
            pass

    # downsample
    if downsample_factor > 1:
        X = X[::downsample_factor]
        ann = ann[::downsample_factor]

    # ann -> id, unknown=-1
    ann_id = np.full((len(ann),), -1, dtype=np.int16)
    for i, a in enumerate(ann):
        v = ann2id.get(a, None)
        if v is not None:
            ann_id[i] = int(v)

    os.makedirs(out_dir, exist_ok=True)
    out_fp = os.path.join(out_dir, f"{pid}.npz")

    if use_compressed:
        np.savez_compressed(out_fp, X=X, ann_id=ann_id)
    else:
        np.savez(out_fp, X=X, ann_id=ann_id)

    return pid, int(X.shape[0]), int((ann_id >= 0).sum())

def build_capture24_pid_cache_parallel(args, out_dir: str):
    """
    并行构建每个 participant 的缓存（推荐！）
    输出：
      out_dir/
        P001.npz  (X[T,3], ann_id[T])
        ...
    """
    root_path = args.capture24_root
    downsample_factor = int(getattr(args, "downsample_factor", 2))
    workers = int(getattr(args, "npz_workers", max(1, os.cpu_count() // 2)))
    use_compressed = bool(getattr(args, "npz_compressed", False))

    # 先用一个临时 Dataset 获取 ann2id（避免你重复写字典读取逻辑）
    tmp = Dataset_CAPTURE24(args, root_path, flag="train")
    ann2id = tmp.ann2id

    # 全部 pids
    files = sorted([f for f in os.listdir(root_path) if re.fullmatch(r"P\d{3}\.csv\.gz", f)])
    pids = [f.split(".")[0] for f in files]

    tasks = [(root_path, pid, downsample_factor, ann2id, out_dir, use_compressed) for pid in pids]

    print(f"[capture24_pid_cache] workers={workers} compressed={use_compressed} pids={len(pids)}")

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_capture24_process_one, t) for t in tasks]
        for fut in as_completed(futs):
            pid, T, known = fut.result()
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"[capture24_pid_cache] {done}/{len(tasks)} done. last={pid} T={T} known={known}")




# --------------------------------------- TimeSeries Datasets -------------------------------------------#
##########################################################################################################
######################################## TimeSeries Datasets  ############################################
########################################                      ############################################
###################################################################################################


# class Dataset_ETT_hour(Dataset):
#     def __init__(self, args, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
#         # size [seq_len, label_len, pred_len]
#         self.args = args
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#
#         local_fp = os.path.join(self.root_path, self.data_path)
#         cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]
#
#         if os.path.exists(local_fp):
#             df_raw = pd.read_csv(local_fp)
#         else:
#             ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
#             df_raw = ds["train"].to_pandas()
#
#         border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
#         border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#
#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]
#
#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
#
#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)
#
#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#
#         if self.set_type == 0 and self.args.augmentation_ratio > 0:
#             self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
#
#         self.data_stamp = data_stamp
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
#
#
# class Dataset_ETT_minute(Dataset):
#     def __init__(self, args, root_path, flag='train', size=None,
#                  features='S', data_path='ETTm1.csv',
#                  target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
#         # size [seq_len, label_len, pred_len]
#         self.args = args
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#
#         local_fp = os.path.join(self.root_path, self.data_path)
#         cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]
#
#         if os.path.exists(local_fp):
#             df_raw = pd.read_csv(local_fp)
#         else:
#             ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
#             df_raw = ds["train"].to_pandas()
#
#         border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
#         border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#
#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]
#
#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
#
#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
#             df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)
#
#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#
#         if self.set_type == 0 and self.args.augmentation_ratio > 0:
#             self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
#
#         self.data_stamp = data_stamp
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
#
#
# class Dataset_Custom(Dataset):
#     def __init__(self, args, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
#         # size [seq_len, label_len, pred_len]
#         self.args = args
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         local_fp = os.path.join(self.root_path, self.data_path)
#         cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]
#
#         if os.path.exists(local_fp):
#             df_raw = pd.read_csv(local_fp)
#         else:
#             ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
#             split_name = "train" if "train" in ds else list(ds.keys())[0]
#             df_raw = ds[split_name].to_pandas()
#
#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         cols = list(df_raw.columns)
#         cols.remove(self.target)
#         cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
#         num_train = int(len(df_raw) * 0.7)
#         num_test = int(len(df_raw) * 0.2)
#         num_vali = len(df_raw) - num_train - num_test
#         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(df_raw)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#
#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]
#
#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
#
#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)
#
#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#
#         if self.set_type == 0 and self.args.augmentation_ratio > 0:
#             self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
#
#         self.data_stamp = data_stamp
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
#
#
# class Dataset_M4(Dataset):
#     def __init__(self, args, root_path, flag='pred', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
#                  seasonal_patterns='Yearly'):
#         # size [seq_len, label_len, pred_len]
#         # init
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.inverse = inverse
#         self.timeenc = timeenc
#         self.root_path = root_path
#
#         self.seq_len = size[0]
#         self.label_len = size[1]
#         self.pred_len = size[2]
#
#         self.seasonal_patterns = seasonal_patterns
#         self.history_size = M4Meta.history_size[seasonal_patterns]
#         self.window_sampling_limit = int(self.history_size * self.pred_len)
#         self.flag = flag
#
#         self.__read_data__()
#
#     def __read_data__(self):
#         # M4Dataset.initialize()
#         if self.flag == 'train':
#             dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
#         else:
#             dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
#         training_values = np.array(
#             [v[~np.isnan(v)] for v in
#              dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
#         self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
#         self.timeseries = [ts for ts in training_values]
#
#     def __getitem__(self, index):
#         insample = np.zeros((self.seq_len, 1))
#         insample_mask = np.zeros((self.seq_len, 1))
#         outsample = np.zeros((self.pred_len + self.label_len, 1))
#         outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset
#
#         sampled_timeseries = self.timeseries[index]
#         cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
#                                       high=len(sampled_timeseries),
#                                       size=1)[0]
#
#         insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
#         insample[-len(insample_window):, 0] = insample_window
#         insample_mask[-len(insample_window):, 0] = 1.0
#         outsample_window = sampled_timeseries[
#                            max(0, cut_point - self.label_len):min(len(sampled_timeseries), cut_point + self.pred_len)]
#         outsample[:len(outsample_window), 0] = outsample_window
#         outsample_mask[:len(outsample_window), 0] = 1.0
#         return insample, outsample, insample_mask, outsample_mask
#
#     def __len__(self):
#         return len(self.timeseries)
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
#
#     def last_insample_window(self):
#         """
#         The last window of insample size of all timeseries.
#         This function does not support batching and does not reshuffle timeseries.
#
#         :return: Last insample window of all timeseries. Shape "timeseries, insample size"
#         """
#         insample = np.zeros((len(self.timeseries), self.seq_len))
#         insample_mask = np.zeros((len(self.timeseries), self.seq_len))
#         for i, ts in enumerate(self.timeseries):
#             ts_last_window = ts[-self.seq_len:]
#             insample[i, -len(ts):] = ts_last_window
#             insample_mask[i, -len(ts):] = 1.0
#         return insample, insample_mask
#
#
# class PSMSegLoader(Dataset):
#     def __init__(self, args, root_path, win_size, step=1, flag="train"):
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()
#         train_path = os.path.join(root_path, "train.csv")
#         test_path = os.path.join(root_path, "test.csv")
#         label_path = os.path.join(root_path, "test_label.csv")
#
#         if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
#             train_df      = pd.read_csv(train_path)
#             test_df       = pd.read_csv(test_path)
#             test_label_df = pd.read_csv(label_path)
#         else:
#             ds_data  = load_dataset(HUGGINGFACE_REPO, name="PSM-datasets")
#             ds_label = load_dataset(HUGGINGFACE_REPO, name="PSM-label")
#             train_df      = ds_data["train"].to_pandas()
#             test_df       = ds_data["test"].to_pandas()
#             test_label_df = ds_label[next(iter(ds_label))].to_pandas()
#
#         data = train_df.values[:, 1:]
#         data = np.nan_to_num(data)
#         self.scaler.fit(data)
#         data = self.scaler.transform(data)
#
#         test_data = test_df.values[:, 1:]
#         test_data = np.nan_to_num(test_data)
#         self.test = self.scaler.transform(test_data)
#
#         self.train = data
#         data_len = len(self.train)
#         self.val = self.train[(int)(data_len * 0.8):]
#         self.test_labels = test_label_df.values[:, 1:]
#         print("test:", self.test.shape)
#         print("train:", self.train.shape)
#
#     def __len__(self):
#         if self.flag == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'val'):
#             return (self.val.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'test'):
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1
#
#     def __getitem__(self, index):
#         index = index * self.step
#         if self.flag == "train":
#             return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'val'):
#             return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'test'):
#             return np.float32(self.test[index:index + self.win_size]), np.float32(
#                 self.test_labels[index:index + self.win_size])
#         else:
#             return np.float32(self.test[
#                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
#                 self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
#
#
# class MSLSegLoader(Dataset):
#     def __init__(self, args, root_path, win_size, step=1, flag="train"):
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()
#
#         train_path = os.path.join(root_path, "MSL_train.npy")
#         test_path  = os.path.join(root_path, "MSL_test.npy")
#         label_path = os.path.join(root_path, "MSL_test_label.npy")
#
#         if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
#             train_data = np.load(train_path)
#             test_data  = np.load(test_path)
#             test_label = np.load(label_path)
#         else:
#             train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_train.npy",repo_type="dataset")
#             test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_test.npy",repo_type="dataset")
#             label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_test_label.npy",repo_type="dataset")
#
#             train_data  = np.load(train_path)
#             test_data   = np.load(test_path)
#             test_label  = np.load(label_path)
#
#         self.scaler.fit(train_data)
#         train_data = self.scaler.transform(train_data)
#         test_data  = self.scaler.transform(test_data)
#
#         self.train = train_data
#         self.test  = test_data
#         self.test_labels = test_label
#
#         data_len = len(self.train)
#         self.val = self.train[int(data_len * 0.8):]
#
#         print("test:", self.test.shape)
#         print("train:", self.train.shape)
#
#     def __len__(self):
#         if self.flag == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'val'):
#             return (self.val.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'test'):
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1
#
#     def __getitem__(self, index):
#         index = index * self.step
#         if self.flag == "train":
#             return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'val'):
#             return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'test'):
#             return np.float32(self.test[index:index + self.win_size]), np.float32(
#                 self.test_labels[index:index + self.win_size])
#         else:
#             return np.float32(self.test[
#                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
#                 self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
#
#
# class SMAPSegLoader(Dataset):
#     def __init__(self, args, root_path, win_size, step=1, flag="train"):
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()
#
#         train_path = os.path.join(root_path, "SMAP_train.npy")
#         test_path  = os.path.join(root_path, "SMAP_test.npy")
#         label_path = os.path.join(root_path, "SMAP_test_label.npy")
#
#         if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
#             train_data = np.load(train_path)
#             test_data  = np.load(test_path)
#             test_label = np.load(label_path)
#         else:
#             train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_train.npy",repo_type="dataset")
#             test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_test.npy",repo_type="dataset")
#             label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_test_label.npy",repo_type="dataset")
#
#             train_data  = np.load(train_path)
#             test_data   = np.load(test_path)
#             test_label = np.load(label_path)
#
#         # 标准化
#         self.scaler.fit(train_data)
#         train_data = self.scaler.transform(train_data)
#         test_data  = self.scaler.transform(test_data)
#
#         self.train = train_data
#         self.test  = test_data
#         self.test_labels = test_label
#
#         data_len = len(self.train)
#         self.val = self.train[int(data_len * 0.8):]
#
#         print("test:", self.test.shape)
#         print("train:", self.train.shape)
#
#     def __len__(self):
#
#         if self.flag == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'val'):
#             return (self.val.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'test'):
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1
#
#     def __getitem__(self, index):
#         index = index * self.step
#         if self.flag == "train":
#             return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'val'):
#             return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'test'):
#             return np.float32(self.test[index:index + self.win_size]), np.float32(
#                 self.test_labels[index:index + self.win_size])
#         else:
#             return np.float32(self.test[
#                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
#                 self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
#
#
# class SMDSegLoader(Dataset):
#     def __init__(self, args, root_path, win_size, step=100, flag="train"):
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()
#
#         train_path = os.path.join(root_path, "SMD_train.npy")
#         test_path  = os.path.join(root_path, "SMD_test.npy")
#         label_path = os.path.join(root_path, "SMD_test_label.npy")
#
#         if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
#             train_data = np.load(train_path)
#             test_data  = np.load(test_path)
#             test_label = np.load(label_path)
#         else:
#             train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_train.npy",repo_type="dataset")
#             test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_test.npy",repo_type="dataset")
#             label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_test_label.npy",repo_type="dataset")
#
#             train_data  = np.load(train_path)
#             test_data   = np.load(test_path)
#             test_label = np.load(label_path)
#
#         self.scaler.fit(train_data)
#         train_data = self.scaler.transform(train_data)
#         test_data = self.scaler.transform(test_data)
#         self.train = train_data
#         self.test = test_data
#         data_len = len(self.train)
#         self.val = self.train[(int)(data_len * 0.8):]
#         self.test_labels = test_label
#         print("test:", self.test.shape)
#         print("train:", self.train.shape)
#
#     def __len__(self):
#         if self.flag == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'val'):
#             return (self.val.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'test'):
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1
#
#     def __getitem__(self, index):
#         index = index * self.step
#         if self.flag == "train":
#             return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'val'):
#             return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'test'):
#             return np.float32(self.test[index:index + self.win_size]), np.float32(
#                 self.test_labels[index:index + self.win_size])
#         else:
#             return np.float32(self.test[
#                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
#                 self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
#
#
# class SWATSegLoader(Dataset):
#     def __init__(self, args, root_path, win_size, step=1, flag="train"):
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()
#
#         train2_path = os.path.join(root_path, "swat_train2.csv")
#         test_path   = os.path.join(root_path, "swat2.csv")
#         if all(os.path.exists(p) for p in [train2_path, test_path]):
#             train_data = pd.read_csv(train2_path)
#             test_data   = pd.read_csv(test_path)
#         else:
#             ds = load_dataset(HUGGINGFACE_REPO, name="SWaT")
#             train_data = ds["train"].to_pandas()
#             test_data  = ds["test"].to_pandas()
#         labels = test_data.values[:, -1:]
#         train_data = train_data.values[:, :-1]
#         test_data = test_data.values[:, :-1]
#
#         self.scaler.fit(train_data)
#         train_data = self.scaler.transform(train_data)
#         test_data = self.scaler.transform(test_data)
#         self.train = train_data
#         self.test = test_data
#         data_len = len(self.train)
#         self.val = self.train[(int)(data_len * 0.8):]
#         self.test_labels = labels
#         print("test:", self.test.shape)
#         print("train:", self.train.shape)
#
#     def __len__(self):
#         """
#         Number of images in the object dataset.
#         """
#         if self.flag == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'val'):
#             return (self.val.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'test'):
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1
#
#     def __getitem__(self, index):
#         index = index * self.step
#         if self.flag == "train":
#             return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'val'):
#             return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'test'):
#             return np.float32(self.test[index:index + self.win_size]), np.float32(
#                 self.test_labels[index:index + self.win_size])
#         else:
#             return np.float32(self.test[
#                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
#                 self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
#
#
# class UEAloader(Dataset):
#     """
#     Dataset class for datasets included in:
#         Time Series Classification Archive (www.timeseriesclassification.com)
#     Argument:
#         limit_size: float in (0, 1) for debug
#     Attributes:
#         all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
#             Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
#         feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
#         feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
#         all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
#         labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
#         max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
#             (Moreover, script argument overrides this attribute)
#     """
#
#     def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
#         self.args = args
#         self.root_path = root_path
#         self.flag = flag
#         self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
#         self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)
#
#         if limit_size is not None:
#             if limit_size > 1:
#                 limit_size = int(limit_size)
#             else:  # interpret as proportion if in (0, 1]
#                 limit_size = int(limit_size * len(self.all_IDs))
#             self.all_IDs = self.all_IDs[:limit_size]
#             self.all_df = self.all_df.loc[self.all_IDs]
#
#         # use all features
#         self.feature_names = self.all_df.columns
#         self.feature_df = self.all_df
#
#         # pre_process
#         normalizer = Normalizer()
#         self.feature_df = normalizer.normalize(self.feature_df)
#
#
#
#     def _resolve_ts_path(self, root_path, dataset_name, flag):
#         split = "TRAIN" if "train" in str(flag).lower() else "TEST"
#         fname = f"{dataset_name}_{split}.ts"
#         local = os.path.join(root_path, fname)
#         if os.path.exists(local):
#             return local
#         return hf_hub_download(HUGGINGFACE_REPO, filename=f"{dataset_name}/{fname}", repo_type="dataset")
#
#     def load_all(self, root_path, file_list=None, flag=None):
#         """
#         Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
#         Args:
#             root_path: directory containing all individual .ts files
#             file_list: optionally, provide a list of file paths within `root_path` to consider.
#                 Otherwise, entire `root_path` contents will be used.
#         Returns:
#             all_df: a single (possibly concatenated) dataframe with all datasets corresponding to specified files
#             labels_df: dataframe containing label(s) for each sample
#         """
#         # Select paths for training and evaluation
#         dataset_name = self.args.model_id
#         ts_path = self._resolve_ts_path(root_path, dataset_name, flag or "train")
#
#         all_df, labels_df = self.load_single(ts_path)
#         return all_df, labels_df
#
#     def load_single(self, filepath):
#         df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
#                                                              replace_missing_vals_with='NaN')
#         labels = pd.Series(labels, dtype="category")
#         self.class_names = labels.cat.categories
#         labels_df = pd.DataFrame(labels.cat.codes,
#                                  dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss
#
#         lengths = df.applymap(
#             lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series
#
#         horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))
#
#         if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
#             df = df.applymap(subsample)
#
#         lengths = df.applymap(lambda x: len(x)).values
#         vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
#         if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
#             self.max_seq_len = int(np.max(lengths[:, 0]))
#         else:
#             self.max_seq_len = lengths[0, 0]
#
#         # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
#         # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
#         # sample index (i.e. the same scheme as all datasets in this project)
#
#         df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
#             pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)
#
#         # Replace NaN values
#         grp = df.groupby(by=df.index)
#         df = grp.transform(interpolate_missing)
#
#         return df, labels_df
#
#     def instance_norm(self, case):
#         if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
#             mean = case.mean(0, keepdim=True)
#             case = case - mean
#             stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
#             case /= stdev
#             return case
#         else:
#             return case
#
#     def __getitem__(self, ind):
#         batch_x = self.feature_df.loc[self.all_IDs[ind]].values
#         labels = self.labels_df.loc[self.all_IDs[ind]].values
#         if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
#             num_samples = len(self.all_IDs)
#             num_columns = self.feature_df.shape[1]
#             seq_len = int(self.feature_df.shape[0] / num_samples)
#             batch_x = batch_x.reshape((1, seq_len, num_columns))
#             batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)
#
#             batch_x = batch_x.reshape((1 * seq_len, num_columns))
#
#         return self.instance_norm(torch.from_numpy(batch_x)), \
#                torch.from_numpy(labels)
#
#     def __len__(self):
#         return len(self.all_IDs)



# ============================================================
# Main: load each dataset + print checks
# ============================================================


def run_check(ds, title, hist_n=None):
    x0, y0 = ds[0]
    print(f"\n[{title}]")
    print(f"  len(dataset) = {len(ds)}")
    print(f"  sample0: x={tuple(x0.shape)} y={y0.tolist()}")
    print(f"  nan_samples_in_first_8 = {_check_nan_in_first_k(ds, 8)}")
    # label hist
    if hasattr(ds, "labels"):
        labels = ds.labels
    elif hasattr(ds, "_y"):
        labels = ds._y
    else:
        labels = [int(ds[i][1].item()) for i in range(min(len(ds), hist_n or 1024))]

    hh = _hist_first_n(labels, hist_n or len(labels))
    print(f"  label_hist (first {min(len(labels), hist_n or len(labels))} samples): {hh}")

    # label range quick check
    arr = np.asarray(labels[: min(len(labels), 1024)])
    if arr.size:
        print(f"  label_range (first {min(len(arr), 1024)}): [{int(arr.min())}, {int(arr.max())}]")


def main():
    p = argparse.ArgumentParser()


    p.add_argument("--ucihar_root", type=str, default=r"/root/autodl-tmp/SensorLLMLib/datasets/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset", help="UCI HAR root (contains train/ test/)")
    p.add_argument("--pamap_root", type=str, default=r"/root/autodl-tmp/SensorLLMLib/datasets/pamap2+physical+activity+monitoring/PAMAP2_Dataset", help="PAMAP2 root (contains Protocol/ Optional/)")
    p.add_argument("--uschad_root", type=str, default=r"/root/autodl-tmp/SensorLLMLib/datasets/USC-HAD/USC-HAD", help="USC-HAD root (contains .mat files)")
    p.add_argument("--capture24_root", type=str, default=r"/root/autodl-tmp/SensorLLMLib/datasets/capture24/capture24", help="CAPTURE-24 root (contains P001.csv.gz etc)")
    p.add_argument("--mhealth_root", type=str,default=r"/root/autodl-tmp/SensorLLMLib/datasets/MHEALTHDATASET",help="MHealth root (contains mHealth_subject*.log)")

    # common
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--val_ratio", type=float, default=0.0)

    # PAMAP
    p.add_argument("--pamap_variant", type=str, default="pamap50", choices=["pamap", "pamap50"])
    p.add_argument("--pamap_use_optional", action="store_true", help="merge Optional folder as well")
    # allow override seq_len/stride from CLI
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)

    # CAPTURE-24
    p.add_argument("--capture24_label_col", type=str, default="label:WillettsSpecific2018")
    p.add_argument("--capture24_keep_ratio", type=float, default=0.05)
    p.add_argument("--downsample_factor", type=int, default=2)

    p.add_argument("--npz_workers", type=int, default=8)
    p.add_argument("--npz_compressed", type=int, default=0)  # 0/1
    p.add_argument("--capture24_cache_mode", type=str, default="windows", choices=["windows", "pid"])

    p.add_argument("--build_npz", action="store_true", help="build offline NPZ cache for datasets")
    p.add_argument("--npz_out", type=str, default="./npz_cache", help="output dir for npz")
    p.add_argument("--npz_max_items", type=int, default=None, help="optional cap for exporting")
    p.add_argument("--npz_target", type=str, default="all",
                   choices=[
                       "mhealth", "ucihar",
                       "pamap2", "pamap2_both",
                       "uschad", "capture24",
                       "all"
                   ])

    args = p.parse_args()
    # CAPTURE-24

    if args.build_npz:
        if args.npz_target in ["mhealth", "all"]:
            build_mhealth_npz_cache(args, args.npz_out, max_items=args.npz_max_items)

        if args.npz_target in ["ucihar", "all"]:
            build_ucihar_npz_cache(args, args.npz_out, max_items=args.npz_max_items)

        if args.npz_target in ["pamap2_both", "all"]:
            build_pamap2_npz_cache_both(args, args.npz_out, max_items=args.npz_max_items)
        elif args.npz_target == "pamap2":
            build_pamap2_npz_cache(args, args.npz_out, max_items=args.npz_max_items)

        if args.npz_target in ["uschad", "all"]:
            build_uschad_npz_cache(args, args.npz_out, max_items=args.npz_max_items)

        if args.npz_target in ["capture24", "all"]:
            if args.capture24_cache_mode == "pid":
                build_capture24_pid_cache_parallel(args, os.path.join(args.npz_out, "capture24_pid"))
            else:
                build_capture24_npz_cache(args, args.npz_out, max_items=args.npz_max_items)

        print(f"[DONE] NPZ cache built under: {args.npz_out}")
        return

    # UCIHAR official split
    ds_uci_tr = Dataset_UCIHAR_Official(args, args.ucihar_root, flag="train")
    run_check(ds_uci_tr, "UCIHAR/train (official)", hist_n=len(ds_uci_tr))

    ds_uci_te = Dataset_UCIHAR_Official(args, args.ucihar_root, flag="test")
    run_check(ds_uci_te, "UCIHAR/test  (official)", hist_n=len(ds_uci_te))

    # PAMAP2 Protocol-only by default
    # window length default chosen by pamap_variant; allow CLI override
    if args.pamap_variant == "pamap50":
        exp_time = ( (args.seq_len or 100) / 50.0 )
    else:
        exp_time = ( (args.seq_len or 200) / 100.0 )
    print(f"\n[PAMAP2 {args.pamap_variant}] expected window time: {exp_time:.2f}s")

    # clone args for pamap dataset defaults
    pamap_args = argparse.Namespace(**vars(args))
    if pamap_args.seq_len is None:
        pamap_args.seq_len = 200 if pamap_args.pamap_variant == "pamap" else 100
    if pamap_args.stride is None:
        pamap_args.stride = pamap_args.seq_len // 2

    ds_pam_tr = Dataset_PAMAP2(pamap_args, pamap_args.pamap_root, flag="train")
    run_check(ds_pam_tr, f"PAMAP2/Protocol train ({pamap_args.pamap_variant})", hist_n=min(len(ds_pam_tr), 14163))

    # USC-HAD
    us_args = argparse.Namespace(**vars(args))
    if us_args.seq_len is None:
        us_args.seq_len = 200
    if us_args.stride is None:
        us_args.stride = 100

    ds_us_tr = Dataset_USCHAD(us_args, us_args.uschad_root, flag="train")
    run_check(ds_us_tr, "USC-HAD train", hist_n=min(len(ds_us_tr), 2000))

    # CAPTURE-24
    cap_args = argparse.Namespace(**vars(args))
    if cap_args.seq_len is None:
        cap_args.seq_len = 500
    if cap_args.stride is None:
        cap_args.stride = 250

    ds_cap_tr = Dataset_CAPTURE24(cap_args, cap_args.capture24_root, flag="train")
    run_check(ds_cap_tr, f"CAPTURE-24 train ({cap_args.capture24_label_col})", hist_n=min(len(ds_cap_tr), 2000))


if __name__ == "__main__":
    main()
