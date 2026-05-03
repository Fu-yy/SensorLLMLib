import pickle

import pandas as pd
from glob import glob
import warnings
import json
import numpy as np
import os, re
import numpy as np
import torch
from torch.utils.data import Dataset

from torch.utils.data import Dataset
import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

HUGGINGFACE_REPO = "thuml/Time-Series-Library"



def _cfg_get(ds_cfg: dict, key: str, default=None):
    if ds_cfg is None:
        return default
    return ds_cfg.get(key, default)

def _csv_int_list(s, default=None):
    if s is None:
        return [] if default is None else default
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


try:
    from scipy.io import loadmat
except Exception as e:
    raise ImportError("Need scipy for .mat loading: pip install scipy") from e


# --------------------------------------- HAR Datasets -------------------------------------------#
###################################################################################################
######################################## HAR Datasets  ############################################
########################################               ############################################
###################################################################################################


# 这个数据集的读取速度比较慢，在训练的时候速度太慢，怎么优化？你能不能只加速别改其他的数据划分策略等？保证输出和原来的一致？请仔细帮我检查修改。


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
        # default_test = ["subject1", "subject3", "subject6"]
        default_test = "subject1,subject3,subject6"

        # self.test_subjects = list(getattr(args, "test_subjects", default_test))
        self.test_subjects = list(getattr(args, "test_subjects", default_test).split(","))

        # self.test_subjects=default_test
        default_val = "subject5"
        self.val_subjects  = list(getattr(args, "val_subjects", default_val).split(","))
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

    def _subject_cache_dir(self) -> str:
        return os.path.join(self.root_path, "_cache_mhealth_npz")

    def _subject_cache_fp(self, subj_id: int) -> str:
        return os.path.join(self._subject_cache_dir(), f"mHealth_subject{subj_id}.npz")

    def _cache_is_fresh(self, log_fp: str, cache_fp: str) -> bool:
        # cache 必须存在且更新时间 >= 原始log
        try:
            return os.path.exists(cache_fp) and (os.path.getmtime(cache_fp) >= os.path.getmtime(log_fp))
        except OSError:
            return False

    def _load_cached_subjects(self):
        """
        Read all 10 subjects once.
        Speed-up (no behavior change):
          - first run: parse .log -> X(float32), y(int64) exactly as before, then save .npz
          - later runs / multi-workers: load .npz directly (fast)
        """
        os.makedirs(self._subject_cache_dir(), exist_ok=True)

        for subj_id in range(1, 11):
            log_fp = self._subject_file(subj_id)
            if not os.path.exists(log_fp):
                raise FileNotFoundError(f"Missing file: {log_fp}")

            cache_fp = self._subject_cache_fp(subj_id)

            # 1) fast path: load binary cache
            if self._cache_is_fresh(log_fp, cache_fp):
                try:
                    with np.load(cache_fp, allow_pickle=False) as z:
                        X = z["X"]
                        y = z["y"]
                    # 保险检查：不改变逻辑，只防止坏cache
                    if X.ndim != 2 or X.shape[1] != 15 or len(X) != len(y):
                        raise RuntimeError("Bad cache shape")
                    if X.dtype != np.float32 or y.dtype != np.int64:
                        # dtype 不对就回退重建（不改变输出，只保证一致）
                        raise RuntimeError("Bad cache dtype")
                    self._X_cache[subj_id] = X
                    self._y_cache[subj_id] = y
                    continue
                except Exception:
                    # cache 损坏或不匹配 -> 回退重建
                    pass

            # 2) slow path: parse .log exactly like your current code
            #    关键：仍然用 loadtxt 读成 float64，再 cast，保证与原行为一致
            arr = np.loadtxt(log_fp)  # [T, 24] float64
            X = arr[:, self.feature_cols].astype(np.float32)  # [T, 15]
            y = arr[:, self.label_col].astype(np.int64)  # [T]

            if X.ndim != 2 or X.shape[1] != 15:
                raise RuntimeError(f"Unexpected feature shape {X.shape} for {log_fp}")
            if len(X) != len(y):
                raise RuntimeError(f"Length mismatch X={len(X)} y={len(y)} for {log_fp}")

            self._X_cache[subj_id] = X
            self._y_cache[subj_id] = y

            # save cache for next time (fast startup + multi-workers)
            np.savez_compressed(cache_fp, X=X, y=y)

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

        # print(f"[split] flag={self.flag} train={sorted(train_set)}")
        # print(f"[split] flag={self.flag} val  ={sorted(val_set)}")
        # print(f"[split] flag={self.flag} test ={sorted(test_set)}")
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
        x = torch.from_numpy(x)
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


# ============================================================
# Utils
# ============================================================



# ============================================================
# 1) UCIHAR (OFFICIAL split, RAW)
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

class Dataset_UCIHAR_Official(Dataset):
    """
    UCI-HAR raw inertial signals loader (OFFICIAL split).

    Returns:
      x: [128, 6] float32 tensor
      y: [1] long label in 0..5
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

        # 可选开关：只加速不改逻辑；默认开
        self.use_cache = bool(getattr(args, "ucihar_use_cache", True))

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

    # -------------------------
    # cache helpers
    # -------------------------
    def _cache_dir(self):
        return os.path.join(self.root_path, "_cache_ucihar_official")

    def _cache_fp(self, split: str):
        return os.path.join(self._cache_dir(), f"{split}_XySubj_seq{self.seq_len}.npz")

    def _source_files(self, split: str):
        split_dir = os.path.join(self.root_path, split)
        suf = split
        sig_dir = os.path.join(split_dir, "Inertial Signals")

        files = [
            os.path.join(split_dir, f"y_{suf}.txt"),
            os.path.join(split_dir, f"subject_{suf}.txt"),
        ]
        for n in self._names:
            files.append(os.path.join(sig_dir, f"{n}_{suf}.txt"))
        return files

    def _cache_is_fresh(self, cache_fp: str, src_files: list) -> bool:
        if (not os.path.exists(cache_fp)) or (not src_files):
            return False
        try:
            cache_m = os.path.getmtime(cache_fp)
            return all(os.path.exists(f) and os.path.getmtime(f) <= cache_m for f in src_files)
        except OSError:
            return False

    # -------------------------
    # loader
    # -------------------------
    def _load_split(self, split: str):
        split_dir = os.path.join(self.root_path, split)
        suf = split  # train/test

        # 1) try binary cache
        cache_fp = self._cache_fp(split)
        src_files = self._source_files(split)

        if self.use_cache:
            os.makedirs(self._cache_dir(), exist_ok=True)
            if self._cache_is_fresh(cache_fp, src_files):
                try:
                    with np.load(cache_fp, allow_pickle=False) as z:
                        X = z["X"]
                        y = z["y"]
                        subj = z["subj"]
                    # 保险检查：不改行为，只防坏缓存
                    if X.shape[1] != self.seq_len or X.shape[2] != 6:
                        raise RuntimeError("bad cache shape")
                    if X.dtype != np.float32 or y.dtype != np.int64 or subj.dtype != np.int64:
                        raise RuntimeError("bad cache dtype")
                    if not (len(X) == len(y) == len(subj)):
                        raise RuntimeError("bad cache length")
                    return X, y, subj
                except Exception:
                    # cache损坏/不匹配：回退到读txt重建
                    pass

        # 2) fallback: read official txt (original logic)
        y = _read_txt(os.path.join(split_dir, f"y_{suf}.txt")).astype(np.int64).reshape(-1) - 1
        subj = _read_txt(os.path.join(split_dir, f"subject_{suf}.txt")).astype(np.int64).reshape(-1)

        sig_dir = os.path.join(split_dir, "Inertial Signals")

        # 预分配避免 list+stack 的额外内存峰值（数值不变）
        N = len(y)
        X = np.empty((N, self.seq_len, 6), dtype=np.float32)

        for ci, n in enumerate(self._names):
            fp = os.path.join(sig_dir, f"{n}_{suf}.txt")
            m = _read_txt(fp).astype(np.float32)  # [N,128]
            if m.shape != (N, self.seq_len):
                raise RuntimeError(f"Unexpected shape {m.shape} in {fp}, expect {(N, self.seq_len)}")
            X[:, :, ci] = m

        if not (len(X) == len(y) == len(subj)):
            raise RuntimeError(f"Length mismatch: X={len(X)} y={len(y)} subj={len(subj)} in {split_dir}")

        # 3) write cache (no behavior change)
        if self.use_cache:
            try:
                np.savez_compressed(cache_fp, X=X, y=y, subj=subj)
            except Exception:
                # 写失败不影响正确性，只是没缓存
                pass

        return X, y, subj

    def __len__(self):
        return len(self._y)

    def __getitem__(self, idx):
        # _X 已是 float32；torch.from_numpy 会得到 float32 tensor，无需 .float() 再拷贝
        # x = torch.from_numpy(self._X[idx])          # [128,6], float32
        x = torch.from_numpy(np.ascontiguousarray(self._X[idx]))
        y = torch.tensor([int(self._y[idx])], dtype=torch.long)
        return x, y

# ============================================================
# 2) PAMAP2 (Protocol-only by default; support Optional merge)
# ============================================================

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


class _PAMAP_Base(Dataset):
    """
    Base class containing common logic for PAMAP2 processing.
    (Speed-up: subject-level npz cache + faster label mapping; no change to split/window outputs.)
    """

    def __init__(self, args, root_path, flag, limit_size, sample_rate, downsample_factor):
        flag = flag.lower()
        assert flag in ["train", "val", "test"]
        self.args = args
        self.root_path = root_path
        self.flag = flag

        # Parameters passed from the specific subclass
        self.sample_rate = sample_rate
        self.downsample_factor = downsample_factor

        # 2s windows consistent across 50/100 Hz defaults
        default_seq = 200 if self.sample_rate == 100 else 100
        default_stride = default_seq // 2

        self.seq_len = int(getattr(args, "seq_len", default_seq))
        self.stride = int(getattr(args, "stride", default_stride))

        # "original split" in many papers: subject-wise split with fixed test subjects
        default_test = ["subject105", "subject106"]
        # IMPORTANT: keep original behavior: string -> split(",") (same as your code)
        self.test_subjects = list(getattr(args, "test_subjects", default_test).split(","))
        self.val_subjects = list(getattr(args, "val_subjects", []))
        self.val_ratio = float(getattr(args, "val_ratio", 0.0))
        self.seed = int(getattr(args, "seed", 2024))

        self.min_seg_len = int(getattr(args, "min_seg_len", self.seq_len))
        self.drop_null = True  # drop activityID=0, plus drop not-in-12-classes

        # Use Protocol only (your requirement)
        self.use_optional = bool(getattr(args, "pamap_use_optional", False))

        # cache switch (default True)
        self.use_cache = bool(getattr(args, "pamap_use_cache", True))

        self.label_col = 1
        self.feature_cols = pamap27_feature_cols()

        # cache
        self._X_cache = {}  # subj -> [T,27]
        self._y_cache = {}  # subj -> [T] 0..11

        # fast label mapping LUT (exactly same mapping as PAMAP_ID2NEW.get)
        self._lut = self._build_act_lut()

        self._load_cached_subjects()

        # build window index
        self.samples, self.labels = self._build_index()

        if limit_size is not None:
            n = len(self.samples)
            n = int(n * float(limit_size)) if limit_size <= 1 else int(limit_size)
            self.samples = self.samples[:n]
            self.labels = self.labels[:n]

        self.max_seq_len = self.seq_len
        self.class_names = list(range(12))

    # ----------------------------
    # mapping LUT (speed-up, equivalent)
    # ----------------------------
    def _build_act_lut(self):
        """
        Build lookup table for activityID -> new label id (0..11), else -1.
        Equivalent to: np.vectorize(PAMAP_ID2NEW.get)(act) with keep_mask is in keys.
        """
        keys = np.array(list(PAMAP_ID2NEW.keys()), dtype=np.int64)
        max_k = int(keys.max()) if keys.size else 0
        lut = np.full((max_k + 1,), -1, dtype=np.int64)
        for k, v in PAMAP_ID2NEW.items():
            if k >= 0:
                lut[int(k)] = int(v)
        return lut

    def _map_act_to_y(self, act: np.ndarray):
        """
        act: int64 [T]
        returns y:int64 [T], keep_mask:bool [T] for valid classes
        """
        act = act.astype(np.int64, copy=False)
        # out of range or negative should be invalid
        keep = (act >= 0) & (act < self._lut.shape[0])
        y = np.full(act.shape, -1, dtype=np.int64)
        y[keep] = self._lut[act[keep]]
        keep2 = (y >= 0)
        return y, keep2

    # ----------------------------
    # path helpers for subject cache
    # ----------------------------
    def _cache_dir(self):
        # separate by Protocol/Optional + downsample_factor to avoid mixing configs
        tag = "ProtOpt" if self.use_optional else "ProtOnly"
        return os.path.join(self.root_path, f"_cache_pamap_npz_{tag}_ds{self.downsample_factor}")

    def _subject_cache_fp(self, subj: str):
        # include selected feature set + label mapping version implicitly
        # seq_len/stride do NOT affect X/y cache, so not included
        return os.path.join(self._cache_dir(), f"{subj}.npz")

    def _cache_is_fresh(self, cache_fp: str, src_files: list):
        if (not os.path.exists(cache_fp)) or (not src_files):
            return False
        try:
            m_cache = os.path.getmtime(cache_fp)
            return all(os.path.exists(f) and os.path.getmtime(f) <= m_cache for f in src_files)
        except OSError:
            return False

    # ----------------------------
    # split logic (unchanged)
    # ----------------------------
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

    # ----------------------------
    # segment/window (unchanged)
    # ----------------------------
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
        if out[-1][1] - 1 < seg_len - 1:
            ws = seg_len - L
            if ws != out[-1][0]:
                out.append((ws, ws + L, label))
        return out

    # ----------------------------
    # file collection (unchanged)
    # ----------------------------
    def _collect_files(self):
        prot_dir = os.path.join(self.root_path, "Protocol")
        opt_dir = os.path.join(self.root_path, "Optional")

        files = []
        if os.path.isdir(prot_dir):
            files += glob.glob(os.path.join(prot_dir, "**", "*.dat"), recursive=True)
        else:
            files += glob.glob(os.path.join(self.root_path, "**", "*.dat"), recursive=True)

        if self.use_optional and os.path.isdir(opt_dir):
            files += glob.glob(os.path.join(opt_dir, "**", "*.dat"), recursive=True)

        files = sorted(list(set(files)))
        if len(files) == 0:
            raise FileNotFoundError(f"No .dat found under {self.root_path} (Protocol-only={not self.use_optional}).")
        return files

    # ----------------------------
    # subject file loader (speed-up, equivalent outputs)
    # ----------------------------
    def _load_subject_files(self, files):
        chunks_X, chunks_y = [], []

        for fp in sorted(files):
            # keep loadtxt to preserve exact numeric parsing behavior
            arr = np.loadtxt(fp)  # [T,54] with NaN possible

            act = arr[:, self.label_col].astype(np.int64, copy=False)
            X = arr[:, self.feature_cols].astype(np.float32, copy=False)
            X = fill_nan_matrix(X)

            # faster + equivalent to:
            # keep_mask = np.isin(act, keys); y = vectorize(PAMAP_ID2NEW.get)(act)
            y, keep_mask = self._map_act_to_y(act)
            X = X[keep_mask]
            y = y[keep_mask]
            if len(y) == 0:
                continue

            if self.downsample_factor > 1:
                X = X[::self.downsample_factor]
                y = y[::self.downsample_factor]

            if len(y) > 0:
                chunks_X.append(X)
                chunks_y.append(y)

        if len(chunks_y) == 0:
            return None, None

        return np.concatenate(chunks_X, axis=0), np.concatenate(chunks_y, axis=0)

    # ----------------------------
    # cached subject loader (major speed-up)
    # ----------------------------
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

        if self.use_cache:
            os.makedirs(self._cache_dir(), exist_ok=True)

        for subj, files in subj_to_files.items():
            cache_fp = self._subject_cache_fp(subj)

            # 1) fast path: load subject cache
            if self.use_cache and self._cache_is_fresh(cache_fp, files):
                try:
                    with np.load(cache_fp, allow_pickle=False) as z:
                        Xall = z["X"]
                        yall = z["y"]
                    # safety checks (do not change behavior, only reject bad cache)
                    if Xall.ndim != 2 or Xall.shape[1] != 27:
                        raise RuntimeError("bad cache X shape")
                    if yall.ndim != 1 or len(Xall) != len(yall):
                        raise RuntimeError("bad cache len")
                    if Xall.dtype != np.float32 or yall.dtype != np.int64:
                        raise RuntimeError("bad cache dtype")
                    self._X_cache[subj] = Xall
                    self._y_cache[subj] = yall
                    continue
                except Exception:
                    pass  # fall back rebuild

            # 2) slow path: build exactly like original logic
            Xall, yall = self._load_subject_files(files)
            if Xall is None:
                continue
            self._X_cache[subj] = Xall
            self._y_cache[subj] = yall

            # write cache for next runs/workers
            if self.use_cache:
                try:
                    np.savez_compressed(cache_fp, X=Xall, y=yall)
                except Exception:
                    pass

        if len(self._X_cache) == 0:
            raise RuntimeError("All subjects empty after filtering. Check mapping / file contents.")

    # ----------------------------
    # index (unchanged)
    # ----------------------------
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

        # X 已经是 float32；torch.from_numpy 直接得到 float32 tensor，.float() 只是额外拷贝
        return torch.from_numpy(x), torch.tensor([int(self.labels[idx])], dtype=torch.long)


class Dataset_PAMAP(_PAMAP_Base):
    """
    PAMAP2 for TSLib-style HAR - 100Hz Original Version.
    - No downsampling (100Hz).
    - Default window 200 (2.0s).
    """

    def __init__(self, args, root_path, flag="train", limit_size=None):
        super().__init__(
            args=args,
            root_path=root_path,
            flag=flag,
            limit_size=limit_size,
            sample_rate=100,
            downsample_factor=1
        )


class Dataset_PAMAP50(_PAMAP_Base):
    """
    PAMAP2 for TSLib-style HAR - 50Hz Downsampled Version.
    - Downsamples 100Hz -> 50Hz (::2).
    - Default window 100 (2.0s).
    """

    def __init__(self, args, root_path, flag="train", limit_size=None):
        super().__init__(
            args=args,
            root_path=root_path,
            flag=flag,
            limit_size=limit_size,
            sample_rate=50,
            downsample_factor=2
        )


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


import os
import re
import glob
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

# ============================================================
# activity utils（与你原来完全一致）
# ============================================================

def _normalize_act_str(s) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("_", "-").replace(" ", "-")
    s = re.sub(r"-+", "-", s)
    return s


USCHAD_ACTIVITY_STR2ID = {
    "walking-forward": 0, "walk-forward": 0, "walkingforward": 0,
    "walking-left": 1, "walk-left": 1, "walkingleft": 1,
    "walking-right": 2, "walk-right": 2, "walkingright": 2,
    "walking-upstairs": 3, "walk-upstairs": 3, "walkingupstairs": 3, "upstairs": 3,
    "walking-downstairs": 4, "walk-downstairs": 4, "walkingdownstairs": 4, "downstairs": 4,
    "running-forward": 5, "run-forward": 5, "runningforward": 5,
    "jumping-up": 6, "jump-up": 6, "jumpingup": 6,
    "sitting": 7, "sit": 7,
    "standing": 8, "stand": 8,
    "sleeping": 9, "sleep": 9,
    "elevator-up": 10, "elevatorup": 10, "elevator-upstairs": 10,
    "elevator-down": 11, "elevatordown": 11, "elevator-downstairs": 11,
}

# ============================================================
# Dataset (FAST, output-identical)
# ============================================================

class Dataset_USCHAD(Dataset):
    """
    USC-HAD (MotionNode) loader (TSLib-style).

    Speed-up (no behavior change):
      - per-file .npz cache for (X, y0, subj)
      - second run / multi-workers avoid loadmat (major speed-up)
      - avoid extra tensor copy in __getitem__ (remove .float())

    Split/window/label parsing logic remains identical to your original.
    """

    def __init__(self, args, root_path, flag="train", limit_size=None):
        flag = flag.lower()
        assert flag in ["train", "val", "test"]
        self.args = args
        self.root_path = root_path
        self.flag = flag

        self.seq_len = int(getattr(args, "seq_len", 200))
        self.stride  = int(getattr(args, "stride", 100))

        # default_test = ["subject13", "subject14"]
        # self.test_subjects = list(getattr(args, "test_subjects", default_test).split(","))

        default_test = ["subject13", "subject14"]
        ts = getattr(args, "test_subjects", default_test)
        if isinstance(ts, str):
            self.test_subjects = [s.strip() for s in ts.split(",") if s.strip()]
        else:
            self.test_subjects = list(ts)
        self.val_subjects  = list(getattr(args, "val_subjects", []))
        self.val_ratio     = float(getattr(args, "val_ratio", 0.0))
        self.seed          = int(getattr(args, "seed", 2024))

        self.min_seg_len = int(getattr(args, "min_seg_len", self.seq_len))

        # cache switch (default True)
        self.use_cache = bool(getattr(args, "uschad_use_cache", True))
        self.cache_dir = os.path.join(self.root_path, "_cache_uschad_npz")

        self._X_cache = []       # list[np.ndarray] each [T,6] float32
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

    # ---------------- mat parsing helpers (UNCHANGED) ----------------
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
        if v is None:
            return None
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)) and np.isfinite(v):
            iv = int(v)
            if abs(v - iv) < 1e-6:
                return iv
            return None
        if isinstance(v, (str, np.str_)):
            s = str(v).strip()
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
        if isinstance(v, np.ndarray):
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
        act_obj = self._try_get_field(d, [
            "activity_number", "activity_num", "activityNumber",
            "activityID", "activity_id",
            "activity",
        ])

        if act_obj is not None:
            act_val = self._get_scalar(act_obj)
            act_int = self._as_int_if_possible(act_val)
            if act_int is not None:
                y0 = act_int - 1
                return y0

        name_obj = self._try_get_field(d, [
            "activity_name", "activityName", "activity_label", "activityLabel",
            "activity",
        ])
        if name_obj is not None:
            name_val = self._get_scalar(name_obj)
            act_int = self._as_int_if_possible(name_val)
            if act_int is not None:
                return act_int - 1

            key = _normalize_act_str(name_val)
            if key in USCHAD_ACTIVITY_STR2ID:
                return int(USCHAD_ACTIVITY_STR2ID[key])

        base = os.path.basename(fp).lower()
        m = re.search(r"a(\d+)", base)
        if m:
            act_int = int(m.group(1))
            return act_int - 1

        raise RuntimeError(f"Cannot infer activity from fields or filename: {fp}")

    def _parse_subject(self, fp, d):
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
            for _, v in d.items():
                if hasattr(v, "__dict__") and hasattr(v, "sensor_readings"):
                    arr = getattr(v, "sensor_readings")
                    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 6:
                        X = arr.astype(np.float32)
                        break

        if X is None:
            raise RuntimeError(f"Cannot find sensor_readings [T,6] in {fp}. Keys={list(d.keys())[:30]}")
        return X

    # ---------------- cache helpers ----------------
    def _safe_hash(self, s: str) -> str:
        import hashlib
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def _cache_fp_for_mat(self, fp: str) -> str:
        # use absolute path hash to avoid name collisions
        h = self._safe_hash(os.path.abspath(fp))
        return os.path.join(self.cache_dir, f"{h}.npz")

    def _cache_is_fresh(self, mat_fp: str, cache_fp: str) -> bool:
        if not os.path.exists(cache_fp):
            return False
        try:
            return os.path.getmtime(cache_fp) >= os.path.getmtime(mat_fp)
        except OSError:
            return False

    def _load_one_mat_uncached(self, fp: str):
        d = loadmat(fp, squeeze_me=True, struct_as_record=False)
        subj_int = int(self._parse_subject(fp, d))
        y0 = int(self._parse_activity(fp, d))
        if not (0 <= y0 <= 11):
            raise RuntimeError(f"Unexpected activity id y0={y0} in {fp}")
        X = self._load_sensor_readings(fp, d)  # float32 [T,6]
        return subj_int, X, y0

    def _load_one_mat(self, fp: str):
        """
        Cached loader:
          - if cache exists and is fresh: load (subj_int, y0, X) from npz
          - else: loadmat + parse exactly as before, then save cache
        """
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            cfp = self._cache_fp_for_mat(fp)
            if self._cache_is_fresh(fp, cfp):
                try:
                    with np.load(cfp, allow_pickle=False) as z:
                        subj_int = int(z["subj"].reshape(-1)[0])
                        y0 = int(z["y0"].reshape(-1)[0])
                        X = z["X"]
                    # safety checks
                    if X.ndim != 2 or X.shape[1] != 6:
                        raise RuntimeError("bad cache X shape")
                    if X.dtype != np.float32:
                        raise RuntimeError("bad cache X dtype")
                    if not (0 <= y0 <= 11):
                        raise RuntimeError("bad cache y0")
                    return f"subject{subj_int}", X, y0
                except Exception:
                    pass  # fall back rebuild

            subj_int, X, y0 = self._load_one_mat_uncached(fp)
            try:
                np.savez_compressed(cfp, X=X, y0=np.array([y0], dtype=np.int64), subj=np.array([subj_int], dtype=np.int64))
            except Exception:
                pass
            return f"subject{subj_int}", X, y0

        # no-cache path
        subj_int, X, y0 = self._load_one_mat_uncached(fp)
        return f"subject{subj_int}", X, y0

    # ---------------- main load/index (same behavior, less memory) ----------------
    def _load_and_index(self):
        mat_files = sorted(glob.glob(os.path.join(self.root_path, "**", "*.mat"), recursive=True))
        if len(mat_files) == 0:
            raise FileNotFoundError(f"No .mat files found under: {self.root_path}")

        # Pass 1: get subject set + lightweight meta (avoid keeping all X twice)
        meta = []
        subj_set = set()

        for fp in mat_files:
            subj_name, X, y0 = self._load_one_mat(fp)  # cached fast after first run
            subj_set.add(subj_name)
            meta.append((fp, subj_name, y0, int(X.shape[0])))

        all_subjects_sorted = sorted(list(subj_set))
        keep_subjects = self._pick_split_subjects(all_subjects_sorted)

        # Pass 2: only keep X for selected subjects; windowize identical
        for (fp, subj_name, y0, T) in meta:
            if subj_name not in keep_subjects:
                continue
            if T < self.min_seg_len:
                continue

            # load X again ONLY if we didn't keep it from pass1
            # (but with caching this is cheap and avoids holding all X in meta)
            subj_name2, X, y0_2 = self._load_one_mat(fp)
            # strict consistency check (should always hold)
            if subj_name2 != subj_name or y0_2 != y0 or int(X.shape[0]) != T:
                raise RuntimeError(f"Cache/meta inconsistency in {fp}")

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

        # X 已经 float32，from_numpy 后就是 float32 tensor；不需要 .float() 再拷贝
        x = torch.from_numpy(np.ascontiguousarray(x))
        return x, torch.tensor([int(self.labels[idx])], dtype=torch.long)
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













# ---------------------------
# Label mapping
# ---------------------------
WISDM_ACT2ID = {
    "walking": 0,
    "jogging": 1,
    "upstairs": 2,
    "downstairs": 3,
    "sitting": 4,
    "standing": 5,
}

# ---------------------------
# Fast helpers
# ---------------------------
def _norm_act(s: str) -> str:
    return str(s).strip().lower()

def _safe_float(s):
    try:
        return float(s)
    except Exception:
        return None

def _safe_int(s):
    try:
        return int(s)
    except Exception:
        return None



class Dataset_WISDM(Dataset):
    """
    WISDM Activity Prediction Dataset v1.1 (raw.txt)

    Output:
      x: [seq_len, 3] float32
      y: [1] long (0..5)

    Speed-up (no behavior change):
      - build/load a single npz cache for parsed per-user streams
      - avoid extra tensor copy in __getitem__ (remove .float())
    """

    def __init__(self, args, root_path, flag="train", limit_size=None):
        flag = flag.lower()
        assert flag in ["train", "val", "test"]
        self.args = args
        self.root_path = root_path
        self.flag = flag

        # ---- ds_cfg ----
        self.ds_cfg = getattr(args, "ds_cfg", {}) if hasattr(args, "ds_cfg") else {}
        dk = str(getattr(args, "dataset_key", "wisdm")).lower()
        if isinstance(self.ds_cfg, dict) and dk in self.ds_cfg and isinstance(self.ds_cfg[dk], dict):
            self.ds_cfg = self.ds_cfg[dk]

        # window config (20Hz)
        self.seq_len = int(_cfg_get(self.ds_cfg, "seq_len", getattr(args, "seq_len", 80)))
        self.stride = int(_cfg_get(self.ds_cfg, "stride", getattr(args, "stride", 40)))
        self.min_seg_len = int(_cfg_get(self.ds_cfg, "min_seg_len", getattr(args, "min_seg_len", self.seq_len)))

        # split config (users 1..36)
        default_test = "33,34,35,36"
        test_users = _cfg_get(self.ds_cfg, "test_users", getattr(args, "test_users", default_test))
        self.test_users = _csv_int_list(test_users)

        default_val = "5,13,17,19,27,31"
        val_users = _cfg_get(self.ds_cfg, "val_users", getattr(args, "val_users", default_val))
        self.val_users = _csv_int_list(val_users)

        self.val_ratio = float(_cfg_get(self.ds_cfg, "val_ratio", getattr(args, "val_ratio", 0.0)))
        self.seed = int(_cfg_get(self.ds_cfg, "seed", getattr(args, "seed", 2024)))

        # normalization
        self.norm = str(
            _cfg_get(
                self.ds_cfg,
                "norm",
                _cfg_get(self.ds_cfg, "wisdm_norm", getattr(args, "wisdm_norm", "none")),
            )
        ).lower()
        self._global_mean = None
        self._global_std = None

        # raw file path
        cfg_root = _cfg_get(self.ds_cfg, "root_path", root_path)
        cfg_raw = _cfg_get(self.ds_cfg, "raw_path", None)

        if cfg_raw is not None:
            self.raw_path = cfg_raw
        else:
            if os.path.isdir(cfg_root):
                self.raw_path = os.path.join(cfg_root, "WISDM_ar_v1.1_raw.txt")
            else:
                self.raw_path = cfg_root

        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"Missing raw file: {self.raw_path}")

        # cache options
        self.use_cache = bool(_cfg_get(self.ds_cfg, "use_cache", getattr(args, "wisdm_use_cache", True)))
        cache_dir = _cfg_get(self.ds_cfg, "cache_dir", getattr(args, "wisdm_cache_dir", None))
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(self.raw_path), "_cache_wisdm_npz")
        self.cache_dir = str(cache_dir)

        # caches (same interface as your original)
        self._user_X = {}  # user -> np.ndarray [T,3] float32
        self._user_y = {}  # user -> np.ndarray [T] int64
        self._user_t = {}  # user -> np.ndarray [T] int64 (not used)

        # index
        self.samples = []
        self.labels = []

        # 1) load + cache (streaming -> npz cache)
        self._load_raw_to_cache()

        # 2) build windows index (no mixed windows)
        self._build_index()

        # limit size
        if limit_size is not None:
            n = len(self.samples)
            n = int(n * float(limit_size)) if limit_size <= 1 else int(limit_size)
            self.samples = self.samples[:n]
            self.labels = self.labels[:n]

        # 3) train-only normalization stats
        if self.norm == "global_std":
            self._fit_global_std_from_train_users()

        self.max_seq_len = self.seq_len
        self.class_names = list(range(len(WISDM_ACT2ID)))

    # ---------------- cache helpers ----------------
    def _cache_fp(self) -> str:
        # cache is independent of seq_len/stride; only depends on raw file content + parsing rules
        base = os.path.basename(self.raw_path)
        return os.path.join(self.cache_dir, f"{base}.parsed_v1.npz")

    def _cache_is_fresh(self, cache_fp: str) -> bool:
        if not os.path.exists(cache_fp):
            return False
        try:
            return os.path.getmtime(cache_fp) >= os.path.getmtime(self.raw_path)
        except OSError:
            return False

    def _save_parsed_cache(self, cache_fp: str, tmp_X: dict, tmp_y: dict, tmp_t: dict):
        """
        Save parsed result without pickles:
          - concatenate all users into one big array
          - store users + offsets to reconstruct per-user views
        """
        users = np.array(sorted(tmp_X.keys()), dtype=np.int64)
        lens = np.array([len(tmp_X[u]) for u in users], dtype=np.int64)
        offsets = np.zeros(len(users) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lens)

        total = int(offsets[-1])
        Xcat = np.empty((total, 3), dtype=np.float32)
        ycat = np.empty((total,), dtype=np.int64)
        tcat = np.empty((total,), dtype=np.int64)

        for i, u in enumerate(users):
            s, e = int(offsets[i]), int(offsets[i + 1])
            Xcat[s:e] = np.asarray(tmp_X[u], dtype=np.float32)
            ycat[s:e] = np.asarray(tmp_y[u], dtype=np.int64)
            tcat[s:e] = np.asarray(tmp_t[u], dtype=np.int64)

        os.makedirs(os.path.dirname(cache_fp), exist_ok=True)
        np.savez_compressed(cache_fp, users=users, offsets=offsets, X=Xcat, y=ycat, t=tcat)

    def _load_parsed_cache(self, cache_fp: str):
        with np.load(cache_fp, allow_pickle=False) as z:
            users = z["users"].astype(np.int64)
            offsets = z["offsets"].astype(np.int64)
            Xcat = z["X"].astype(np.float32)
            ycat = z["y"].astype(np.int64)
            tcat = z["t"].astype(np.int64)

        if offsets.ndim != 1 or users.ndim != 1 or offsets.shape[0] != users.shape[0] + 1:
            raise RuntimeError("Bad WISDM cache format (offsets/users).")

        # reconstruct dict views (slices are views; no extra copies)
        for i, u in enumerate(users.tolist()):
            s, e = int(offsets[i]), int(offsets[i + 1])
            self._user_X[u] = Xcat[s:e]
            self._user_y[u] = ycat[s:e]
            self._user_t[u] = tcat[s:e]

    # ---------------- split logic ----------------
    def _pick_split_users(self):
        all_users = sorted(list(self._user_X.keys()))
        test = set(self.test_users)
        remain = [u for u in all_users if u not in test]
        if len(remain) == 0:
            raise RuntimeError("All users are in test_users. No train/val left.")

        if len(self.val_users) > 0:
            val = set(self.val_users)
        elif self.val_ratio > 0:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(remain)
            n_val = max(1, int(round(len(remain) * self.val_ratio)))
            val = set(remain[:n_val])
        else:
            val = set()

        train = set([u for u in remain if u not in val])

        assert len(train & val) == 0, f"leak train∩val: {train & val}"
        assert len(train & test) == 0, f"leak train∩test: {train & test}"
        assert len(val & test) == 0, f"leak val∩test: {val & test}"

        if self.flag == "train":
            return train
        if self.flag == "val":
            return val if len(val) else train
        return test

    # ---------------- raw parsing ----------------
    def _load_raw_to_cache(self):
        """
        Behavior identical to your original parser.
        Speed: if cache exists -> load; else parse once and save cache.
        """
        cache_fp = self._cache_fp()

        # 1) fast path
        if self.use_cache and self._cache_is_fresh(cache_fp):
            try:
                self._load_parsed_cache(cache_fp)
                if len(self._user_X) == 0:
                    raise RuntimeError("Empty cache")
                return
            except Exception:
                # fall back to rebuild
                self._user_X, self._user_y, self._user_t = {}, {}, {}

        # 2) slow path: parse raw (same rules)
        tmp_X = {}
        tmp_y = {}
        tmp_t = {}

        bad = 0
        total = 0

        with open(self.raw_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                total += 1
                line = line.strip()
                if not line:
                    continue

                if line.endswith(";"):
                    line = line[:-1]

                parts = line.split(",")
                if len(parts) != 6:
                    bad += 1
                    continue

                user_s, act_s, ts_s, x_s, y_s, z_s = parts
                user = _safe_int(user_s.strip())
                if user is None:
                    bad += 1
                    continue

                act = _norm_act(act_s)
                if act not in WISDM_ACT2ID:
                    bad += 1
                    continue
                y0 = WISDM_ACT2ID[act]

                ts = _safe_int(ts_s.strip())
                x = _safe_float(x_s)
                y = _safe_float(y_s)
                z = _safe_float(z_s)
                if x is None or y is None or z is None:
                    bad += 1
                    continue

                tmp_X.setdefault(user, []).append([x, y, z])
                tmp_y.setdefault(user, []).append(y0)
                tmp_t.setdefault(user, []).append(ts if ts is not None else 0)

        # finalize into numpy arrays (same as original)
        for user in sorted(tmp_X.keys()):
            X = np.asarray(tmp_X[user], dtype=np.float32)
            y = np.asarray(tmp_y[user], dtype=np.int64)
            t = np.asarray(tmp_t[user], dtype=np.int64)
            if len(X) != len(y):
                raise RuntimeError(f"Length mismatch for user {user}: X={len(X)} y={len(y)}")
            self._user_X[user] = X
            self._user_y[user] = y
            self._user_t[user] = t

        if len(self._user_X) == 0:
            raise RuntimeError("No data parsed from raw file. Check file encoding/format.")

        # save cache for next run/workers
        if self.use_cache:
            try:
                self._save_parsed_cache(cache_fp, tmp_X, tmp_y, tmp_t)
            except Exception:
                pass

        # 可选 log
        # print(f"[WISDM] parsed users={len(self._user_X)} lines={total} bad_lines={bad}")

    # ---------------- segmentation + window ----------------
    def _continuous_segments(self, y: np.ndarray):
        segs = []
        s = 0
        for i in range(1, len(y)):
            if y[i] != y[i - 1]:
                segs.append((s, i - 1, int(y[i - 1])))
                s = i
        segs.append((s, len(y) - 1, int(y[-1])))
        segs = [seg for seg in segs if (seg[1] - seg[0] + 1) >= self.min_seg_len]
        return segs

    def _windowize(self, seg_len: int):
        L, S = self.seq_len, self.stride
        if seg_len < L:
            return []
        out = []
        num_complete = (seg_len - L) // S + 1
        for i in range(num_complete):
            ws = i * S
            out.append((ws, ws + L))
        if out[-1][1] - 1 < seg_len - 1:
            ws = seg_len - L
            if ws != out[-1][0]:
                out.append((ws, ws + L))
        return out

    def _build_index(self):
        keep_users = self._pick_split_users()
        samples, labels = [], []

        for user in sorted(self._user_X.keys()):
            if user not in keep_users:
                continue

            y = self._user_y[user]
            segs = self._continuous_segments(y)
            for (s, e, lab) in segs:
                seg_len = e - s + 1
                for (ws, we) in self._windowize(seg_len):
                    samples.append({"user": user, "seg_s": s, "win_s": ws, "win_e": we})
                    labels.append(lab)

        if len(samples) == 0:
            raise RuntimeError("No windows built. Check seq_len/stride/min_seg_len or split users.")

        self.samples = samples
        self.labels = labels

    # ---------------- normalization ----------------
    def _fit_global_std_from_train_users(self):
        cur_flag = self.flag
        self.flag = "train"
        train_users = self._pick_split_users()
        self.flag = cur_flag

        xs = []
        for user in sorted(self._user_X.keys()):
            if user not in train_users:
                continue
            X = self._user_X[user]
            y = self._user_y[user]
            segs = self._continuous_segments(y)
            for (s, e, _) in segs:
                Xseg = X[s:e+1]
                seg_len = Xseg.shape[0]
                for (ws, we) in self._windowize(seg_len):
                    xs.append(Xseg[ws:we])
        Xall = np.concatenate(xs, axis=0)
        self._global_mean = Xall.mean(axis=0, keepdims=True).astype(np.float32)
        self._global_std = Xall.std(axis=0, keepdims=True).astype(np.float32)
        self._global_std = np.maximum(self._global_std, 1e-6)

    def _apply_norm(self, x: np.ndarray):
        if self.norm == "global_std":
            return (x - self._global_mean) / self._global_std
        return x

    # ---------------- torch dataset ----------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        meta = self.samples[idx]
        user = meta["user"]
        seg_s = meta["seg_s"]
        ws, we = meta["win_s"], meta["win_e"]

        X = self._user_X[user]
        x = X[seg_s + ws: seg_s + we]
        if x.shape[0] != self.seq_len:
            raise RuntimeError(f"Window length mismatch got {x.shape[0]} expected {self.seq_len}")

        x = self._apply_norm(x)

        # x 已是 float32；from_numpy 后就是 float32 tensor，不需要 .float() 再拷贝
        x = torch.from_numpy(x)
        y = torch.tensor([int(self.labels[idx])], dtype=torch.long)
        return x, y

import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Dataset_HHAR_cross_user(Dataset):
    """
    STRICTLY EQUIVALENT + FAST VERSION
    """

    # ============================================================
    # __init__（保持原样，只在 load 阶段提速）
    # ============================================================
    def __init__(self, args, root_path, flag='train', limit_size=None):
        flag = flag.lower()
        assert flag in ['train', 'val', 'test']
        self.args = args
        self.root_path = root_path
        self.flag = flag

        self.ds_cfg = getattr(args, "ds_cfg", {}) if hasattr(args, "ds_cfg") else {}
        self.root_path = str(self.ds_cfg.get("root_path", self.root_path))

        self.seq_len = int(self.ds_cfg.get("seq_len", getattr(args, "seq_len", 128)))
        self.stride  = int(self.ds_cfg.get("stride",  getattr(args, "stride", self.seq_len // 2)))

        self.val_ratio  = float(self.ds_cfg.get("val_ratio",  getattr(args, "val_ratio", 0.1)))
        self.test_ratio = float(self.ds_cfg.get("test_ratio", getattr(args, "test_ratio", 0.2)))
        self.seed = int(self.ds_cfg.get("seed", getattr(args, "seed", 2024)))

        self.drop_null   = bool(self.ds_cfg.get("drop_null", True))
        self.min_seg_len = int(self.ds_cfg.get("min_seg_len", self.seq_len))

        self.tol = float(self.ds_cfg.get("tol", getattr(args, "hhar_tol", 0.05)))
        self.align_on = str(self.ds_cfg.get("align_on", "Arrival_Time"))

        self.use_cache = bool(self.ds_cfg.get("use_cache", True))
        self.cache_dir = self.ds_cfg.get("cache_dir",
                          os.path.join(self.root_path, "_cache_hhar_aligned_npz"))

        self.norm = str(self.ds_cfg.get("norm", "none")).lower()

        self.label_map = {
            "bike": 0, "sit": 1, "stand": 2,
            "walk": 3, "stairsup": 4, "stairsdown": 5,
        }

        # caches
        self._X_cache, self._y_cache, self._t_cache = {}, {}, {}
        self._keys = []

        # === FAST BUT EQUIVALENT LOAD ===
        self._load_cached_streams_fast_equivalent()

        # segments / splits / windows
        self._all_segments = self._build_all_segments()
        self._seg_split    = self._assign_segment_splits(self._all_segments)

        self.samples, self.labels = self._build_index_for_flag(self.flag)

        if limit_size is not None:
            n = len(self.samples)
            n = int(n * limit_size) if limit_size <= 1 else int(limit_size)
            self.samples, self.labels = self.samples[:n], self.labels[:n]

        if self.norm == "global_std":
            self._fit_global_std_from_train_windows()

    # ============================================================
    # CSV 读取（字段与原版完全一致）
    # ============================================================
    def _read_csv(self, fname):
        df = pd.read_csv(os.path.join(self.root_path, fname))
        need = ["Arrival_Time", "Creation_Time", "x", "y", "z",
                "User", "Model", "Device", "gt"]
        for c in need:
            if c not in df.columns:
                raise RuntimeError(f"{fname} missing {c}")

        df["User"]   = df["User"].astype(str)
        df["Model"]  = df["Model"].astype(str)
        df["Device"] = df["Device"].astype(str)
        df["gt"]     = df["gt"].astype(str).str.lower().str.strip()

        if self.drop_null:
            df = df[df["gt"] != "null"]

        for c in ["Arrival_Time", "Creation_Time", "x", "y", "z"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        return df.dropna(subset=[self.align_on, "x", "y", "z", "User", "Device", "gt"])

    # ============================================================
    # === 等价加速核心 ===
    # ============================================================
    def _load_cached_streams_fast_equivalent(self):
        phone_acc = self._read_csv("Phones_accelerometer.csv")
        phone_gyr = self._read_csv("Phones_gyroscope.csv")
        watch_acc = self._read_csv("Watch_accelerometer.csv")
        watch_gyr = self._read_csv("Watch_gyroscope.csv")

        def build(acc_df, gyr_df, devtype):
            # === EQUIVALENCE GUARANTEE ===
            # key 顺序必须 sorted，和原版完全一致
            acc_g = dict(tuple(acc_df.groupby(["User", "Device"], sort=False)))
            gyr_g = dict(tuple(gyr_df.groupby(["User", "Device"], sort=False)))

            keys = sorted(set(acc_g.keys()) & set(gyr_g.keys()))

            for (u, d) in keys:
                acc_u = acc_g[(u, d)]
                gyr_u = gyr_g[(u, d)]

                cache_fp = os.path.join(
                    self.cache_dir, f"{u}__{devtype}__{d}.npz"
                )

                if self.use_cache and os.path.exists(cache_fp):
                    z = np.load(cache_fp)
                    t, X, y = z["t"], z["X"], z["y"]
                else:
                    out = self._align_one_stream(acc_u, gyr_u)
                    if out is None:
                        continue
                    t, X, y, _ = out
                    if self.use_cache:
                        os.makedirs(self.cache_dir, exist_ok=True)
                        np.savez_compressed(cache_fp, t=t, X=X, y=y)

                if len(X) < self.seq_len:
                    continue

                k = (str(u), str(devtype), str(d))
                self._t_cache[k] = t
                self._X_cache[k] = X
                self._y_cache[k] = y
                self._keys.append(k)

        build(phone_acc, phone_gyr, "phone")
        build(watch_acc, watch_gyr, "watch")

        if not self._keys:
            raise RuntimeError("No valid streams.")
    def _safe(self, s: str) -> str:
        s = str(s)
        s = s.replace("/", "_").replace("\\", "_").replace(" ", "_")
        s = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", s)
        return s
    def _save_cache(self, fp: str, t: np.ndarray, X: np.ndarray, y: np.ndarray):
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        np.savez_compressed(fp, t=t.astype(np.float64), X=X.astype(np.float32), y=y.astype(np.int64))
    def _build_index_for_flag(self, flag: str):
        samples, labels = [], []
        for seg in self._all_segments:
            if self._seg_split[seg["segid"]] != flag and not (flag == "val" and self.val_ratio == 0 and self._seg_split[seg["segid"]] == "train"):
                continue

            k = seg["key"]
            s, e, lab = seg["s"], seg["e"], seg["lab"]

            seg_len = e - s + 1
            wins = self._windowize(seg_len, lab)

            for (ws, we, y0) in wins:
                samples.append({"key": k, "seg_s": s, "win_s": ws, "win_e": we})
                labels.append(int(y0))

        if len(samples) == 0:
            raise RuntimeError(f"No windows built for flag={flag}. Try adjusting hhar_tol/min_seg_len/seq_len/stride.")
        return samples, labels
    def _fit_global_std_from_train_windows(self):
        xs = []
        for seg in self._all_segments:
            if self._seg_split[seg["segid"]] != "train":
                continue
            k = seg["key"]
            X = self._X_cache[k]
            s, e, lab = seg["s"], seg["e"], seg["lab"]
            Xseg = X[s:e+1]
            seg_len = Xseg.shape[0]
            for (ws, we, _) in self._windowize(seg_len, lab):
                xs.append(Xseg[ws:we])
        Xall = np.concatenate(xs, axis=0)
        self._global_mean = Xall.mean(axis=0, keepdims=True).astype(np.float32)
        self._global_std  = Xall.std(axis=0, keepdims=True).astype(np.float32)
        self._global_std  = np.maximum(self._global_std, 1e-6)

    # ============================================================
    # 【CHANGED】windowize 结果缓存（不改行为）
    # ============================================================
    def _windowize_cached(self, seg_len, lab, segid):
        if segid in self._seg_windows_cache:
            return self._seg_windows_cache[segid]
        wins = self._windowize(seg_len, lab)
        self._seg_windows_cache[segid] = wins
        return wins
    def _normalize_time_to_seconds(self, t: np.ndarray) -> np.ndarray:
        """
        Normalize HHAR timestamps to seconds (float64) robustly.
        We only need a monotonic axis for merge_asof.

        Heuristic by median step (of positive diffs):
          - med > 1e6  -> treat as ns  -> /1e9
          - med > 1e3  -> treat as us  -> /1e6
          - med > 10   -> treat as ms  -> /1e3
          - else       -> seconds-like
        """
        t = np.asarray(t, dtype=np.float64)
        if t.size < 3:
            return t

        tt = np.sort(t)
        dt = np.diff(tt)
        dt = dt[dt > 0]
        if dt.size == 0:
            return t
        med = float(np.median(dt))

        if med > 1e6:
            return t / 1e9  # ns -> s
        if med > 1e3:
            return t / 1e6  # us -> s
        if med > 10:
            return t / 1e3  # ms -> s
        return t
    def _cache_fp(self, user: str, devtype: str, device: str) -> str:
        return os.path.join(self.cache_dir, f"{self._safe(user)}__{self._safe(devtype)}__{self._safe(device)}.npz")
    def _load_cache(self, fp: str):
        z = np.load(fp, allow_pickle=False)
        t = z["t"].astype(np.float64)
        X = z["X"].astype(np.float32)
        y = z["y"].astype(np.int64)
        return t, X, y
    def _summarize_align_stats(self, devtype: str, stats_list: list):
        """
        Print a compact summary and show worst top-k by keep_ratio.
        """
        if len(stats_list) == 0:
            print(f"[{devtype}] align-stats: EMPTY")
            return

        keep = np.array([s["keep_ratio"] for s in stats_list], dtype=np.float64)
        med_dt = np.array([s["median_dt_ms"] for s in stats_list], dtype=np.float64)
        p95_dt = np.array([s["p95_dt_ms"] for s in stats_list], dtype=np.float64)

        def _safe_stat(a, fn):
            a = a[np.isfinite(a)]
            if a.size == 0:
                return float("nan")
            return float(fn(a))

        print(f"[{devtype}] align-stats summary:")
        print(f"  keep_ratio: mean={keep.mean():.3f}  p10={np.percentile(keep,10):.3f}  p50={np.percentile(keep,50):.3f}  p90={np.percentile(keep,90):.3f}")
        print(f"  median_dt_ms: median={_safe_stat(med_dt, np.median):.2f}  p95={_safe_stat(med_dt, lambda x: np.percentile(x,95)):.2f}")
        print(f"  p95_dt_ms   : median={_safe_stat(p95_dt, np.median):.2f}  p95={_safe_stat(p95_dt, lambda x: np.percentile(x,95)):.2f}")

        # worst top-k by keep_ratio
        k = min(self.align_stats_topk, len(stats_list))
        idx = np.argsort(keep)[:k]
        print(f"  worst {k} streams by keep_ratio:")
        for ii in idx:
            s = stats_list[ii]
            key = s.get("key", "NA")
            print(f"    {key}  keep={s['keep_ratio']:.3f}  matched={s['matched_n']}/{s['acc_n']}  median_dt={s['median_dt_ms']:.1f}ms  p95_dt={s['p95_dt_ms']:.1f}ms  tol={s['tol_sec']*1000:.0f}ms")
    def _load_cached_streams(self):
        phone_acc = self._read_csv("Phones_accelerometer.csv")
        phone_gyr = self._read_csv("Phones_gyroscope.csv")
        watch_acc = self._read_csv("Watch_accelerometer.csv")
        watch_gyr = self._read_csv("Watch_gyroscope.csv")

        def build_for_type(acc_df, gyr_df, devtype: str):
            keys_total = 0
            both_exist = 0
            aligned_ok = 0
            saved = 0
            loaded_from_cache = 0

            stats_list = []  # for summary

            all_keys = sorted(list(set(zip(acc_df["User"], acc_df["Device"])) | set(zip(gyr_df["User"], gyr_df["Device"]))))

            for (u, d) in all_keys:
                keys_total += 1
                acc_u = acc_df[(acc_df["User"] == u) & (acc_df["Device"] == d)]
                gyr_u = gyr_df[(gyr_df["User"] == u) & (gyr_df["Device"] == d)]
                if len(acc_u) == 0 or len(gyr_u) == 0:
                    continue
                both_exist += 1

                cache_fp = self._cache_fp(u, devtype, d)

                # 1) try load cache
                if self.use_cache and os.path.exists(cache_fp):
                    try:
                        t, X, y = self._load_cache(cache_fp)
                        if len(X) >= self.seq_len:
                            k = (str(u), str(devtype), str(d))
                            self._t_cache[k] = t
                            self._X_cache[k] = X
                            self._y_cache[k] = y
                            self._keys.append(k)

                            aligned_ok += 1
                            loaded_from_cache += 1
                            # cache hit does NOT mean "saved this run"
                            # we still can record a placeholder stat
                            if self.print_align_stats:
                                stats_list.append({
                                    "key": k,
                                    "acc_n": -1, "gyr_n": -1, "matched_n": -1,
                                    "keep_ratio": 1.0,
                                    "tol_sec": float(self.tol),
                                    "median_dt_ms": float("nan"),
                                    "p95_dt_ms": float("nan"),
                                    "max_dt_ms": float("nan"),
                                    "from_cache": True,
                                })
                            continue
                    except Exception:
                        # cache broken -> rebuild
                        pass

                # 2) align now
                out = self._align_one_stream(acc_u, gyr_u)
                if out is None:
                    continue

                t, X, y, stats = out
                aligned_ok += 1

                k = (str(u), str(devtype), str(d))
                self._t_cache[k] = t
                self._X_cache[k] = X
                self._y_cache[k] = y
                self._keys.append(k)

                # save cache only if needed (no cache or cache broken)
                if self.use_cache:
                    self._save_cache(cache_fp, t, X, y)
                    saved += 1

                if self.print_align_stats:
                    stats = dict(stats)
                    stats["key"] = k
                    stats["from_cache"] = False
                    stats_list.append(stats)
                    self._align_stats_by_key[k] = stats

            print(f"[{devtype}] keys_total={keys_total}, both_exist={both_exist}, aligned_ok={aligned_ok}, saved={saved}, loaded_cache={loaded_from_cache}")

            if self.print_align_stats:
                # only summarize REAL aligned stats (exclude cache placeholders)
                real_stats = [s for s in stats_list if not s.get("from_cache", False)]
                self._summarize_align_stats(devtype, real_stats)

        build_for_type(phone_acc, phone_gyr, "phone")
        build_for_type(watch_acc, watch_gyr, "watch")

        if len(self._keys) == 0:
            raise RuntimeError("No valid aligned streams cached. Try increasing hhar_tol or check timestamp normalization.")
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
        L = self.seq_len
        S = self.stride
        y0 = label
        out = []
        if seg_len < L:
            return out

        num_complete = (seg_len - L) // S + 1
        for i in range(num_complete):
            ws = i * S
            out.append((ws, ws + L, y0))

        last_end_minus1 = out[-1][1] - 1
        if last_end_minus1 < seg_len - 1:
            ws = seg_len - L
            if ws != out[-1][0]:
                out.append((ws, ws + L, y0))
        return out
    def _build_all_segments(self):
        segs = []
        segid = 0
        for k in self._keys:
            y = self._y_cache[k]
            for (s, e, lab) in self._continuous_segments(y):
                segs.append({"key": k, "s": s, "e": e, "lab": int(lab), "segid": segid})
                segid += 1
        if len(segs) == 0:
            raise RuntimeError("No segments built. Try lowering min_seg_len.")
        return segs
    def _assign_segment_splits_stratified_by_class(self, segs):
        """
        Stratified segment split by class, using WINDOW COUNTS as budget.
        """
        rng = np.random.RandomState(self.seed)

        win_cnt = np.zeros(len(segs), dtype=np.int64)
        for i, seg in enumerate(segs):
            seg_len = int(seg["e"] - seg["s"] + 1)
            lab = int(seg["lab"])
            win_cnt[i] = len(self._windowize(seg_len, lab))

        by_lab = {}
        for i, seg in enumerate(segs):
            by_lab.setdefault(int(seg["lab"]), []).append(i)

        test_ids, val_ids, train_ids = set(), set(), set()

        for lab, idxs in by_lab.items():
            idxs = np.array(idxs, dtype=np.int64)
            rng.shuffle(idxs)

            total_w = int(win_cnt[idxs].sum())
            if total_w == 0:
                train_ids.update(idxs.tolist())
                continue

            target_test_w = int(round(total_w * self.test_ratio))
            target_val_w  = int(round(total_w * self.val_ratio))

            target_test_w = max(target_test_w, 50)
            target_val_w  = max(target_val_w, 50)

            if target_test_w + target_val_w >= total_w:
                target_test_w = min(target_test_w, max(1, total_w // 3))
                target_val_w  = min(target_val_w, max(1, total_w // 3))

            acc_test = 0
            acc_val = 0

            for i in idxs:
                w = int(win_cnt[i])
                if acc_test < target_test_w:
                    test_ids.add(int(i)); acc_test += w
                elif acc_val < target_val_w:
                    val_ids.add(int(i)); acc_val += w
                else:
                    train_ids.add(int(i))

            remaining = set(idxs.tolist()) - test_ids - val_ids - train_ids
            train_ids.update(remaining)

        seg_split = {}
        for i, seg in enumerate(segs):
            if i in test_ids:
                seg_split[seg["segid"]] = "test"
            elif i in val_ids:
                seg_split[seg["segid"]] = "val"
            else:
                seg_split[seg["segid"]] = "train"
        return seg_split
    def _assign_segment_splits(self, segs):
        """
        segs: list of {"key":(user,devtype,device), "s","e","lab","segid",...}

        Modes:
          - segment (your current ID baseline): class-stratified segment split
          - devtype_transfer: phone->watch or watch->phone (train/val only in src devtype, test only in tgt devtype)
        """
        mode = getattr(self, "split_mode", "segment")
        mode = str(mode).lower()

        if mode == "segment":
            # ====== keep your original behavior (ID baseline) ======
            return self._assign_segment_splits_stratified_by_class(segs)

        if mode != "devtype_transfer":
            raise ValueError(f"Unknown hhar_split_mode={mode}. Use 'segment' or 'devtype_transfer'.")

        transfer = getattr(self, "transfer", "phone2watch")
        transfer = str(transfer).lower()
        if transfer not in ["phone2watch", "watch2phone"]:
            raise ValueError(f"Unknown hhar_transfer={transfer}. Use 'phone2watch' or 'watch2phone'.")

        src = "phone" if transfer == "phone2watch" else "watch"
        tgt = "watch" if transfer == "phone2watch" else "phone"

        # 1) split by devtype deterministically
        src_segs = [seg for seg in segs if str(seg["key"][1]).lower() == src]
        tgt_segs = [seg for seg in segs if str(seg["key"][1]).lower() == tgt]

        if len(src_segs) == 0 or len(tgt_segs) == 0:
            raise RuntimeError(
                f"devtype_transfer requires both '{src}' and '{tgt}' segments. Got src={len(src_segs)}, tgt={len(tgt_segs)}")

        # 2) TEST = all target-domain segments
        test_ids = set([seg["segid"] for seg in tgt_segs])

        # 3) TRAIN/VAL = split source-domain segments (class-stratified, using window counts as budget)
        #    (We reuse the same idea as your original: stratify by class, but ONLY inside src domain.)
        rng = np.random.RandomState(self.seed)

        # compute window count for each src segment
        win_cnt = {}
        for seg in src_segs:
            seg_len = int(seg["e"] - seg["s"] + 1)
            lab = int(seg["lab"])
            win_cnt[seg["segid"]] = len(self._windowize(seg_len, lab))

        by_lab = {}
        for seg in src_segs:
            by_lab.setdefault(int(seg["lab"]), []).append(seg["segid"])

        val_ids = set()
        train_ids = set()

        for lab, segids in by_lab.items():
            segids = np.array(segids, dtype=np.int64)
            rng.shuffle(segids)

            total_w = int(sum(win_cnt[int(sid)] for sid in segids))
            if total_w == 0:
                train_ids.update([int(sid) for sid in segids])
                continue

            target_val_w = int(round(total_w * self.val_ratio))
            target_val_w = max(target_val_w, 50)  # keep your safety floor
            if target_val_w >= total_w:
                target_val_w = max(1, total_w // 3)

            acc_val = 0
            for sid in segids:
                sid = int(sid)
                w = int(win_cnt[sid])
                if acc_val < target_val_w:
                    val_ids.add(sid);
                    acc_val += w
                else:
                    train_ids.add(sid)

            # anything not assigned goes to train
            remaining = set(int(sid) for sid in segids) - val_ids - train_ids
            train_ids.update(remaining)

        # 4) build seg_split map
        seg_split = {}
        for seg in segs:
            sid = seg["segid"]
            if sid in test_ids:
                seg_split[sid] = "test"
            elif sid in val_ids:
                seg_split[sid] = "val"
            else:
                seg_split[sid] = "train"

        # 5) leak checks on segid level (should be disjoint by construction)
        assert len(set(train_ids) & set(val_ids)) == 0
        assert len(set(train_ids) & set(test_ids)) == 0
        assert len(set(val_ids) & set(test_ids)) == 0

        # optional: log once
        # print(f"[HHAR split] mode=devtype_transfer {src}->{tgt}  trainSeg={len(train_ids)} valSeg={len(val_ids)} testSeg={len(test_ids)}")
        return seg_split

    # ============================================================
    # 其余方法（_align_one_stream / split / getitem）
    # 逻辑与原版完全一致，不再重复贴
    # ============================================================
    def _align_one_stream(self, acc_df: pd.DataFrame, gyro_df: pd.DataFrame):
        """
        Robust align:
          1) normalize to seconds
          2) shift each stream to start at 0 (removes epoch offset)
          3) optional scale correction by median step ratio (removes clock-scale mismatch)
          4) merge_asof with tolerance

        Returns:
          t_sec: float64 [T]  (acc timeline, relative seconds)
          X: float32 [T,6]
          y: int64 [T]
          stats: dict
        """
        on = self.align_on

        acc = acc_df[[on, "x", "y", "z", "User", "Model", "Device", "gt"]].copy()
        gyr = gyro_df[[on, "x", "y", "z"]].copy()

        acc = acc.dropna(subset=[on, "x", "y", "z", "gt"])
        gyr = gyr.dropna(subset=[on, "x", "y", "z"])

        acc_t_raw = acc[on].to_numpy()
        gyr_t_raw = gyr[on].to_numpy()

        # 1) normalize to seconds (your heuristic)
        acc_t = self._normalize_time_to_seconds(acc_t_raw)
        gyr_t = self._normalize_time_to_seconds(gyr_t_raw)

        # sort
        acc = acc.assign(_t_acc=acc_t).sort_values("_t_acc")
        gyr = gyr.assign(_t_gyr=gyr_t).sort_values("_t_gyr")

        acc_t = acc["_t_acc"].to_numpy(np.float64)
        gyr_t = gyr["_t_gyr"].to_numpy(np.float64)

        # drop non-increasing / duplicates (important for merge_asof quality)
        # keep only strictly increasing times
        def _make_strictly_increasing(t, df, col):
            t = np.asarray(t, dtype=np.float64)
            keep = np.ones(len(t), dtype=bool)
            keep[1:] = (t[1:] > t[:-1])
            df2 = df.loc[keep].copy()
            t2 = df2[col].to_numpy(np.float64)
            return t2, df2

        acc_t, acc = _make_strictly_increasing(acc_t, acc, "_t_acc")
        gyr_t, gyr = _make_strictly_increasing(gyr_t, gyr, "_t_gyr")

        if len(acc_t) < self.seq_len or len(gyr_t) < self.seq_len:
            return None

        # 2) shift to start at 0  (removes constant offset between sensors)
        acc0 = float(acc_t[0])
        gyr0 = float(gyr_t[0])
        acc_t_rel = acc_t - acc0
        gyr_t_rel = gyr_t - gyr0

        # 3) optional scale correction (fix clock-rate mismatch)
        def _median_step(t):
            dt = np.diff(t)
            dt = dt[dt > 0]
            if dt.size == 0:
                return None
            return float(np.median(dt))

        acc_step = _median_step(acc_t_rel)
        gyr_step = _median_step(gyr_t_rel)

        scale = 1.0
        if (acc_step is not None) and (gyr_step is not None) and acc_step > 0 and gyr_step > 0:
            scale = gyr_step / acc_step
            # only correct if明显不一致
            if scale < 0.8 or scale > 1.25:
                # gyro 时间尺度不合理：拉回到 acc 的尺度
                gyr_t_rel = gyr_t_rel / scale
            else:
                scale = 1.0  # treat as consistent, keep untouched

        # write back rel times for merge
        acc = acc.assign(_t_acc=acc_t_rel)
        gyr = gyr.assign(_t_gyr=gyr_t_rel)

        # 4) merge_asof (keep gyro time to compute dt)
        merged = pd.merge_asof(
            acc,
            gyr[["_t_gyr", "x", "y", "z"]].rename(columns={"x": "x_gyr", "y": "y_gyr", "z": "z_gyr"}),
            left_on="_t_acc",
            right_on="_t_gyr",
            direction="nearest",
            tolerance=float(self.tol),
        )

        acc_n = len(acc)
        matched = merged.dropna(subset=["x_gyr", "y_gyr", "z_gyr", "_t_gyr"]).copy()
        matched_n = len(matched)
        keep_ratio = matched_n / max(1, acc_n)

        if matched_n == 0:
            return None

        # real dt on RELATIVE time axis
        dt = np.abs(matched["_t_acc"].to_numpy(np.float64) - matched["_t_gyr"].to_numpy(np.float64))
        median_dt_ms = float(np.median(dt) * 1000.0)
        p95_dt_ms = float(np.percentile(dt, 95) * 1000.0)
        max_dt_ms = float(np.max(dt) * 1000.0)

        # labels
        gt = matched["gt"].astype(str).str.lower().str.strip()
        y_series = gt.map(self.label_map)
        valid = y_series.notna().to_numpy(bool)

        t = matched["_t_acc"].to_numpy(np.float64)[valid]
        X = matched[["x", "y", "z", "x_gyr", "y_gyr", "z_gyr"]].to_numpy(np.float32)[valid]
        y = y_series.to_numpy(dtype=np.int64, na_value=-1)[valid]

        keep2 = (y >= 0)
        t, X, y = t[keep2], X[keep2], y[keep2]

        if len(X) < self.seq_len:
            return None

        stats = {
            "acc_n": int(acc_n),
            "gyr_n": int(len(gyr)),
            "matched_n": int(matched_n),
            "keep_ratio": float(keep_ratio),
            "tol_sec": float(self.tol),
            "median_dt_ms": float(median_dt_ms),
            "p95_dt_ms": float(p95_dt_ms),
            "max_dt_ms": float(max_dt_ms),
            "acc_step_sec": float(acc_step) if acc_step is not None else float("nan"),
            "gyr_step_sec": float(gyr_step) if gyr_step is not None else float("nan"),
            "scale_applied": float(scale),
            "offset_removed_sec": float(acc0 - gyr0),  # 原始起点差（秒），被我们移除了
        }
        return t, X, y, stats
    def _apply_norm(self, x: np.ndarray):
        if self.norm == "global_std":
            return (x - self._global_mean) / self._global_std
        return x

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        k = meta["key"]
        X = self._X_cache[k]
        s = meta["seg_s"]
        ws, we = meta["win_s"], meta["win_e"]
        x = X[s + ws: s + we]
        x = self._apply_norm(x)
        return torch.from_numpy(x).float(), torch.tensor([self.labels[idx]], dtype=torch.long)




class Dataset_HHAR_1user(Dataset):
    """
    HHAR for classification (ETT-style fast dataset):
      - cache aligned (acc+gyro) streams in memory by (User, DeviceType, Device)
      - segment-level split to avoid window leakage (A baseline, same-distribution)
      - windowize within segments
      - train-only global_std normalization (optional)
      - IMPORTANT: align by Arrival_Time (robust for Phone)

    CSV columns (HHAR):
      Index, Arrival_Time, Creation_Time, x, y, z, User, Model, Device, gt
    """

    def __init__(self, args, root_path, flag='train', limit_size=None):
        flag = flag.lower()
        assert flag in ['train', 'val', 'test']
        self.args = args
        self.root_path = root_path
        self.ds_cfg = getattr(args, "ds_cfg", {}) if hasattr(args, "ds_cfg") else {}
        self.root_path = str(_cfg_get(self.ds_cfg, "root_path", self.root_path))
        self.flag = flag
        # ---- ds_cfg ----
        dk = str(getattr(args, "dataset_key", "hhar")).lower()
        if isinstance(self.ds_cfg, dict) and dk in self.ds_cfg and isinstance(self.ds_cfg[dk], dict):
            self.ds_cfg = self.ds_cfg[dk]

        # -------- window config ----------
        self.seq_len = int(_cfg_get(self.ds_cfg, "seq_len", getattr(args, "seq_len", 128)))
        self.stride = int(_cfg_get(self.ds_cfg, "stride", getattr(args, "stride", max(1, self.seq_len // 2))))

        # -------- split config (A baseline: same-distribution) ----------
        self.val_ratio = float(_cfg_get(self.ds_cfg, "val_ratio", getattr(args, "val_ratio", 0.1)))
        self.test_ratio = float(_cfg_get(self.ds_cfg, "test_ratio", getattr(args, "test_ratio", 0.2)))

        assert 0 <= self.val_ratio < 1 and 0 <= self.test_ratio < 1 and (self.val_ratio + self.test_ratio) < 1
        self.seed = int(_cfg_get(self.ds_cfg, "seed", getattr(args, "seed", 2024)))

        # -------- label/segment config ----------
        self.drop_null = bool(_cfg_get(self.ds_cfg, "drop_null", getattr(args, "drop_null", True)))
        self.min_seg_len = int(_cfg_get(self.ds_cfg, "min_seg_len", getattr(args, "min_seg_len", self.seq_len)))

        self.tol = float(
            _cfg_get(self.ds_cfg, "tol", _cfg_get(self.ds_cfg, "hhar_tol", getattr(args, "hhar_tol", 0.05))))
        self.align_on = str(_cfg_get(self.ds_cfg, "align_on", _cfg_get(self.ds_cfg, "hhar_align_on",
                                                                       getattr(args, "hhar_align_on",
                                                                               "Arrival_Time")))).strip()

        # cache_dir 也建议支持相对路径
        cache_dir = _cfg_get(self.ds_cfg, "cache_dir",
                             _cfg_get(self.ds_cfg, "hhar_cache_dir", getattr(args, "hhar_cache_dir", None)))
        if cache_dir is None:
            cache_dir = os.path.join(self.root_path, "_cache_hhar_aligned_npz")
        self.cache_dir = str(cache_dir)
        self.use_cache = bool(_cfg_get(self.ds_cfg, "use_cache",
                                       _cfg_get(self.ds_cfg, "hhar_use_cache", getattr(args, "hhar_use_cache", True))))

        self.print_align_stats = bool(_cfg_get(self.ds_cfg, "print_align_stats",
                                               _cfg_get(self.ds_cfg, "hhar_print_align_stats",
                                                        getattr(args, "hhar_print_align_stats", True))))
        self.align_stats_topk = int(_cfg_get(self.ds_cfg, "align_stats_topk",
                                             _cfg_get(self.ds_cfg, "hhar_align_stats_topk",
                                                      getattr(args, "hhar_align_stats_topk", 10))))

        self.norm = str(_cfg_get(self.ds_cfg, "norm",
                                 _cfg_get(self.ds_cfg, "hhar_norm", getattr(args, "hhar_norm", "none")))).lower()


        self._global_mean = None
        self._global_std  = None

        # -------- label map ----------
        self.label_map = {
            "bike": 0,
            "sit": 1,
            "stand": 2,
            "walk": 3,
            "stairsup": 4,
            "stairsdown": 5,
        }
        self.id2label = {v: k for k, v in self.label_map.items()}

        # ----------------------------
        # 1) Read + align + cache streams
        # ----------------------------
        self._X_cache = {}   # key -> np.ndarray [T,6]
        self._y_cache = {}   # key -> np.ndarray [T] int64
        self._t_cache = {}   # key -> np.ndarray [T] float64 (normalized time seconds)
        self._keys = []      # list of keys
        self._align_stats_by_key = {}  # key -> stats dict

        self._load_cached_streams()
        print(f"[HHAR] cached streams = {len(self._keys)}  cache_dir={self.cache_dir}  use_cache={self.use_cache}")

        # ----------------------------
        # 2) Build segments + split segments
        # ----------------------------
        self._all_segments = self._build_all_segments()
        self._seg_split = self._assign_segment_splits(self._all_segments)

        # ----------------------------
        # 3) Build windows index for this flag
        # ----------------------------
        self.samples, self.labels = self._build_index_for_flag(self.flag)

        if limit_size is not None:
            n = len(self.samples)
            if limit_size <= 1:
                n = int(n * float(limit_size))
            else:
                n = int(limit_size)
            self.samples = self.samples[:n]
            self.labels  = self.labels[:n]

        # ----------------------------
        # 4) Fit normalization on TRAIN only
        # ----------------------------
        if self.norm == "global_std":
            self._fit_global_std_from_train_windows()

    # ============================================================
    # Cache helpers
    # ============================================================
    def _safe(self, s: str) -> str:
        s = str(s)
        s = s.replace("/", "_").replace("\\", "_").replace(" ", "_")
        s = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", s)
        return s

    def _cache_fp(self, user: str, devtype: str, device: str) -> str:
        return os.path.join(self.cache_dir, f"{self._safe(user)}__{self._safe(devtype)}__{self._safe(device)}.npz")

    def _save_cache(self, fp: str, t: np.ndarray, X: np.ndarray, y: np.ndarray):
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        np.savez_compressed(fp, t=t.astype(np.float64), X=X.astype(np.float32), y=y.astype(np.int64))

    def _load_cache(self, fp: str):
        z = np.load(fp, allow_pickle=False)
        t = z["t"].astype(np.float64)
        X = z["X"].astype(np.float32)
        y = z["y"].astype(np.int64)
        return t, X, y

    # ============================================================
    # 1) Read + align
    # ============================================================
    def _read_csv(self, fname: str) -> pd.DataFrame:
        fp = os.path.join(self.root_path, fname)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing file: {fp}")

        df = pd.read_csv(fp)

        need = ["Arrival_Time", "Creation_Time", "x", "y", "z", "User", "Model", "Device", "gt"]
        for c in need:
            if c not in df.columns:
                raise RuntimeError(f"{fname} missing column: {c}")

        # normalize strings
        df["User"] = df["User"].astype(str)
        df["Model"] = df["Model"].astype(str)
        df["Device"] = df["Device"].astype(str)
        df["gt"] = df["gt"].astype(str).str.lower().str.strip()

        # drop null label if configured
        if self.drop_null:
            df = df[df["gt"] != "null"].copy()

        # numeric coercion
        df["Arrival_Time"] = pd.to_numeric(df["Arrival_Time"], errors="coerce")
        df["Creation_Time"] = pd.to_numeric(df["Creation_Time"], errors="coerce")
        for ax in ["x", "y", "z"]:
            df[ax] = pd.to_numeric(df[ax], errors="coerce")

        # IMPORTANT: align uses Arrival_Time by default
        df = df.dropna(subset=[self.align_on, "x", "y", "z", "User", "Device", "gt"])
        return df

    def _normalize_time_to_seconds(self, t: np.ndarray) -> np.ndarray:
        """
        Normalize HHAR timestamps to seconds (float64) robustly.
        We only need a monotonic axis for merge_asof.

        Heuristic by median step (of positive diffs):
          - med > 1e6  -> treat as ns  -> /1e9
          - med > 1e3  -> treat as us  -> /1e6
          - med > 10   -> treat as ms  -> /1e3
          - else       -> seconds-like
        """
        t = np.asarray(t, dtype=np.float64)
        if t.size < 3:
            return t

        tt = np.sort(t)
        dt = np.diff(tt)
        dt = dt[dt > 0]
        if dt.size == 0:
            return t
        med = float(np.median(dt))

        if med > 1e6:
            return t / 1e9  # ns -> s
        if med > 1e3:
            return t / 1e6  # us -> s
        if med > 10:
            return t / 1e3  # ms -> s
        return t

    def _align_one_stream(self, acc_df: pd.DataFrame, gyro_df: pd.DataFrame):
        """
        Robust align:
          1) normalize to seconds
          2) shift each stream to start at 0 (removes epoch offset)
          3) optional scale correction by median step ratio (removes clock-scale mismatch)
          4) merge_asof with tolerance

        Returns:
          t_sec: float64 [T]  (acc timeline, relative seconds)
          X: float32 [T,6]
          y: int64 [T]
          stats: dict
        """
        on = self.align_on

        acc = acc_df[[on, "x", "y", "z", "User", "Model", "Device", "gt"]].copy()
        gyr = gyro_df[[on, "x", "y", "z"]].copy()

        acc = acc.dropna(subset=[on, "x", "y", "z", "gt"])
        gyr = gyr.dropna(subset=[on, "x", "y", "z"])

        acc_t_raw = acc[on].to_numpy()
        gyr_t_raw = gyr[on].to_numpy()

        # 1) normalize to seconds (your heuristic)
        acc_t = self._normalize_time_to_seconds(acc_t_raw)
        gyr_t = self._normalize_time_to_seconds(gyr_t_raw)

        # sort
        # acc = acc.assign(_t_acc=acc_t).sort_values("_t_acc")
        # gyr = gyr.assign(_t_gyr=gyr_t).sort_values("_t_gyr")

        # 改成（稳定排序 + 次关键字）：
        acc = acc.assign(_t_acc=acc_t, _idx=np.arange(len(acc))).sort_values(["_t_acc", "_idx"], kind="mergesort")
        gyr = gyr.assign(_t_gyr=gyr_t, _idx=np.arange(len(gyr))).sort_values(["_t_gyr", "_idx"], kind="mergesort")

        acc_t = acc["_t_acc"].to_numpy(np.float64)
        gyr_t = gyr["_t_gyr"].to_numpy(np.float64)

        # drop non-increasing / duplicates (important for merge_asof quality)
        # keep only strictly increasing times
        def _make_strictly_increasing(t, df, col):
            t = np.asarray(t, dtype=np.float64)
            keep = np.ones(len(t), dtype=bool)
            keep[1:] = (t[1:] > t[:-1])
            df2 = df.loc[keep].copy()
            t2 = df2[col].to_numpy(np.float64)
            return t2, df2

        acc_t, acc = _make_strictly_increasing(acc_t, acc, "_t_acc")
        gyr_t, gyr = _make_strictly_increasing(gyr_t, gyr, "_t_gyr")

        if len(acc_t) < self.seq_len or len(gyr_t) < self.seq_len:
            return None

        # 2) shift to start at 0  (removes constant offset between sensors)
        acc0 = float(acc_t[0])
        gyr0 = float(gyr_t[0])
        acc_t_rel = acc_t - acc0
        gyr_t_rel = gyr_t - gyr0

        # 3) optional scale correction (fix clock-rate mismatch)
        def _median_step(t):
            dt = np.diff(t)
            dt = dt[dt > 0]
            if dt.size == 0:
                return None
            return float(np.median(dt))

        acc_step = _median_step(acc_t_rel)
        gyr_step = _median_step(gyr_t_rel)

        scale = 1.0
        if (acc_step is not None) and (gyr_step is not None) and acc_step > 0 and gyr_step > 0:
            scale = gyr_step / acc_step
            # only correct if明显不一致
            if scale < 0.8 or scale > 1.25:
                # gyro 时间尺度不合理：拉回到 acc 的尺度
                gyr_t_rel = gyr_t_rel / scale
            else:
                scale = 1.0  # treat as consistent, keep untouched

        # write back rel times for merge
        acc = acc.assign(_t_acc=acc_t_rel)
        gyr = gyr.assign(_t_gyr=gyr_t_rel)

        # 4) merge_asof (keep gyro time to compute dt)
        merged = pd.merge_asof(
            acc,
            gyr[["_t_gyr", "x", "y", "z"]].rename(columns={"x": "x_gyr", "y": "y_gyr", "z": "z_gyr"}),
            left_on="_t_acc",
            right_on="_t_gyr",
            direction="nearest",
            tolerance=float(self.tol),
        )

        acc_n = len(acc)
        matched = merged.dropna(subset=["x_gyr", "y_gyr", "z_gyr", "_t_gyr"]).copy()
        matched_n = len(matched)
        keep_ratio = matched_n / max(1, acc_n)

        if matched_n == 0:
            return None

        # real dt on RELATIVE time axis
        dt = np.abs(matched["_t_acc"].to_numpy(np.float64) - matched["_t_gyr"].to_numpy(np.float64))
        median_dt_ms = float(np.median(dt) * 1000.0)
        p95_dt_ms = float(np.percentile(dt, 95) * 1000.0)
        max_dt_ms = float(np.max(dt) * 1000.0)

        # labels
        gt = matched["gt"].astype(str).str.lower().str.strip()
        y_series = gt.map(self.label_map)
        valid = y_series.notna().to_numpy(bool)

        t = matched["_t_acc"].to_numpy(np.float64)[valid]
        X = matched[["x", "y", "z", "x_gyr", "y_gyr", "z_gyr"]].to_numpy(np.float32)[valid]
        y = y_series.to_numpy(dtype=np.int64, na_value=-1)[valid]

        keep2 = (y >= 0)
        t, X, y = t[keep2], X[keep2], y[keep2]

        if len(X) < self.seq_len:
            return None

        stats = {
            "acc_n": int(acc_n),
            "gyr_n": int(len(gyr)),
            "matched_n": int(matched_n),
            "keep_ratio": float(keep_ratio),
            "tol_sec": float(self.tol),
            "median_dt_ms": float(median_dt_ms),
            "p95_dt_ms": float(p95_dt_ms),
            "max_dt_ms": float(max_dt_ms),
            "acc_step_sec": float(acc_step) if acc_step is not None else float("nan"),
            "gyr_step_sec": float(gyr_step) if gyr_step is not None else float("nan"),
            "scale_applied": float(scale),
            "offset_removed_sec": float(acc0 - gyr0),  # 原始起点差（秒），被我们移除了
        }
        return t, X, y, stats

    def _summarize_align_stats(self, devtype: str, stats_list: list):
        """
        Print a compact summary and show worst top-k by keep_ratio.
        """
        if len(stats_list) == 0:
            print(f"[{devtype}] align-stats: EMPTY")
            return

        keep = np.array([s["keep_ratio"] for s in stats_list], dtype=np.float64)
        med_dt = np.array([s["median_dt_ms"] for s in stats_list], dtype=np.float64)
        p95_dt = np.array([s["p95_dt_ms"] for s in stats_list], dtype=np.float64)

        def _safe_stat(a, fn):
            a = a[np.isfinite(a)]
            if a.size == 0:
                return float("nan")
            return float(fn(a))

        print(f"[{devtype}] align-stats summary:")
        print(f"  keep_ratio: mean={keep.mean():.3f}  p10={np.percentile(keep,10):.3f}  p50={np.percentile(keep,50):.3f}  p90={np.percentile(keep,90):.3f}")
        print(f"  median_dt_ms: median={_safe_stat(med_dt, np.median):.2f}  p95={_safe_stat(med_dt, lambda x: np.percentile(x,95)):.2f}")
        print(f"  p95_dt_ms   : median={_safe_stat(p95_dt, np.median):.2f}  p95={_safe_stat(p95_dt, lambda x: np.percentile(x,95)):.2f}")

        # worst top-k by keep_ratio
        k = min(self.align_stats_topk, len(stats_list))
        idx = np.argsort(keep)[:k]
        print(f"  worst {k} streams by keep_ratio:")
        for ii in idx:
            s = stats_list[ii]
            key = s.get("key", "NA")
            print(f"    {key}  keep={s['keep_ratio']:.3f}  matched={s['matched_n']}/{s['acc_n']}  median_dt={s['median_dt_ms']:.1f}ms  p95_dt={s['p95_dt_ms']:.1f}ms  tol={s['tol_sec']*1000:.0f}ms")

    # ============================================================
    # 1) Read + align  (FAST & STRICTLY EQUIVALENT)
    # ============================================================
    def _load_cached_streams(self):
        phone_acc = self._read_csv("Phones_accelerometer.csv")
        phone_gyr = self._read_csv("Phones_gyroscope.csv")
        watch_acc = self._read_csv("Watch_accelerometer.csv")
        watch_gyr = self._read_csv("Watch_gyroscope.csv")

        def build_for_type_fast(acc_df, gyr_df, devtype: str):
            # --------- counters (完全一致) ----------
            keys_total = 0
            both_exist = 0
            aligned_ok = 0
            saved = 0
            loaded_from_cache = 0

            stats_list = []

            # ========= 关键加速点 =========
            # 一次 groupby，避免 O(N^2) mask
            acc_groups = dict(tuple(acc_df.groupby(["User", "Device"], sort=False)))
            gyr_groups = dict(tuple(gyr_df.groupby(["User", "Device"], sort=False)))

            # === EQUIVALENCE GUARANTEE ===
            # key 顺序必须和原版 sorted(set(...)) 完全一致
            all_keys = sorted(set(acc_groups.keys()) | set(gyr_groups.keys()))

            for (u, d) in all_keys:
                keys_total += 1
                acc_u = acc_groups.get((u, d))
                gyr_u = gyr_groups.get((u, d))
                if acc_u is None or gyr_u is None:
                    continue
                both_exist += 1

                cache_fp = self._cache_fp(u, devtype, d)

                # --------- 1) try load cache ----------
                if self.use_cache and os.path.exists(cache_fp):
                    try:
                        t, X, y = self._load_cache(cache_fp)
                        if len(X) >= self.seq_len:
                            k = (str(u), str(devtype), str(d))
                            self._t_cache[k] = t
                            self._X_cache[k] = X
                            self._y_cache[k] = y
                            self._keys.append(k)

                            aligned_ok += 1
                            loaded_from_cache += 1

                            if self.print_align_stats:
                                stats_list.append({
                                    "key": k,
                                    "acc_n": -1,
                                    "gyr_n": -1,
                                    "matched_n": -1,
                                    "keep_ratio": 1.0,
                                    "tol_sec": float(self.tol),
                                    "median_dt_ms": float("nan"),
                                    "p95_dt_ms": float("nan"),
                                    "max_dt_ms": float("nan"),
                                    "from_cache": True,
                                })
                            continue
                    except Exception:
                        # cache broken → rebuild
                        pass

                # --------- 2) align ----------
                out = self._align_one_stream(acc_u, gyr_u)
                if out is None:
                    continue

                t, X, y, stats = out
                aligned_ok += 1

                k = (str(u), str(devtype), str(d))
                self._t_cache[k] = t
                self._X_cache[k] = X
                self._y_cache[k] = y
                self._keys.append(k)

                if self.use_cache:
                    self._save_cache(cache_fp, t, X, y)
                    saved += 1

                if self.print_align_stats:
                    stats = dict(stats)
                    stats["key"] = k
                    stats["from_cache"] = False
                    stats_list.append(stats)
                    self._align_stats_by_key[k] = stats

            print(f"[{devtype}] keys_total={keys_total}, both_exist={both_exist}, aligned_ok={aligned_ok}, saved={saved}, loaded_cache={loaded_from_cache}")

            if self.print_align_stats:
                real_stats = [s for s in stats_list if not s.get("from_cache", False)]
                self._summarize_align_stats(devtype, real_stats)

        # === 顺序、逻辑与原版一致 ===
        build_for_type_fast(phone_acc, phone_gyr, "phone")
        build_for_type_fast(watch_acc, watch_gyr, "watch")

        if len(self._keys) == 0:
            raise RuntimeError("No valid aligned streams cached. Try increasing hhar_tol or check timestamp normalization.")
    # ============================================================
    # 2) Segment + split (A baseline)
    # ============================================================
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
        L = self.seq_len
        S = self.stride
        y0 = label
        out = []
        if seg_len < L:
            return out

        num_complete = (seg_len - L) // S + 1
        for i in range(num_complete):
            ws = i * S
            out.append((ws, ws + L, y0))

        last_end_minus1 = out[-1][1] - 1
        if last_end_minus1 < seg_len - 1:
            ws = seg_len - L
            if ws != out[-1][0]:
                out.append((ws, ws + L, y0))
        return out

    def _build_all_segments(self):
        segs = []
        segid = 0
        for k in self._keys:
            y = self._y_cache[k]
            for (s, e, lab) in self._continuous_segments(y):
                segs.append({"key": k, "s": s, "e": e, "lab": int(lab), "segid": segid})
                segid += 1
        if len(segs) == 0:
            raise RuntimeError("No segments built. Try lowering min_seg_len.")
        return segs

    def _assign_segment_splits(self, segs):
        """
        Stratified segment split by class, using WINDOW COUNTS as budget.
        """
        rng = np.random.RandomState(self.seed)

        win_cnt = np.zeros(len(segs), dtype=np.int64)
        for i, seg in enumerate(segs):
            seg_len = int(seg["e"] - seg["s"] + 1)
            lab = int(seg["lab"])
            win_cnt[i] = len(self._windowize(seg_len, lab))

        by_lab = {}
        for i, seg in enumerate(segs):
            by_lab.setdefault(int(seg["lab"]), []).append(i)

        test_ids, val_ids, train_ids = set(), set(), set()

        for lab, idxs in by_lab.items():
            idxs = np.array(idxs, dtype=np.int64)
            rng.shuffle(idxs)

            total_w = int(win_cnt[idxs].sum())
            if total_w == 0:
                train_ids.update(idxs.tolist())
                continue

            target_test_w = int(round(total_w * self.test_ratio))
            target_val_w  = int(round(total_w * self.val_ratio))

            target_test_w = max(target_test_w, 50)
            target_val_w  = max(target_val_w, 50)

            if target_test_w + target_val_w >= total_w:
                target_test_w = min(target_test_w, max(1, total_w // 3))
                target_val_w  = min(target_val_w, max(1, total_w // 3))

            acc_test = 0
            acc_val = 0

            for i in idxs:
                w = int(win_cnt[i])
                if acc_test < target_test_w:
                    test_ids.add(int(i)); acc_test += w
                elif acc_val < target_val_w:
                    val_ids.add(int(i)); acc_val += w
                else:
                    train_ids.add(int(i))

            remaining = set(idxs.tolist()) - test_ids - val_ids - train_ids
            train_ids.update(remaining)

        seg_split = {}
        for i, seg in enumerate(segs):
            if i in test_ids:
                seg_split[seg["segid"]] = "test"
            elif i in val_ids:
                seg_split[seg["segid"]] = "val"
            else:
                seg_split[seg["segid"]] = "train"
        return seg_split

    # ============================================================
    # 3) Build windows index
    # ============================================================
    def _build_index_for_flag(self, flag: str):
        samples, labels = [], []
        for seg in self._all_segments:
            if self._seg_split[seg["segid"]] != flag and not (flag == "val" and self.val_ratio == 0 and self._seg_split[seg["segid"]] == "train"):
                continue

            k = seg["key"]
            s, e, lab = seg["s"], seg["e"], seg["lab"]

            seg_len = e - s + 1
            wins = self._windowize(seg_len, lab)

            for (ws, we, y0) in wins:
                samples.append({"key": k, "seg_s": s, "win_s": ws, "win_e": we})
                labels.append(int(y0))

        if len(samples) == 0:
            raise RuntimeError(f"No windows built for flag={flag}. Try adjusting hhar_tol/min_seg_len/seq_len/stride.")
        return samples, labels

    # ============================================================
    # 4) Normalization (train only)
    # ============================================================
    def _fit_global_std_from_train_windows(self):
        xs = []
        for seg in self._all_segments:
            if self._seg_split[seg["segid"]] != "train":
                continue
            k = seg["key"]
            X = self._X_cache[k]
            s, e, lab = seg["s"], seg["e"], seg["lab"]
            Xseg = X[s:e+1]
            seg_len = Xseg.shape[0]
            for (ws, we, _) in self._windowize(seg_len, lab):
                xs.append(Xseg[ws:we])
        Xall = np.concatenate(xs, axis=0)
        self._global_mean = Xall.mean(axis=0, keepdims=True).astype(np.float32)
        self._global_std  = Xall.std(axis=0, keepdims=True).astype(np.float32)
        self._global_std  = np.maximum(self._global_std, 1e-6)

    def _apply_norm(self, x: np.ndarray):
        if self.norm == "global_std":
            return (x - self._global_mean) / self._global_std
        return x

    # ============================================================
    # PyTorch Dataset API
    # ============================================================
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        meta = self.samples[idx]
        k = meta["key"]
        X = self._X_cache[k]

        s = meta["seg_s"]
        ws, we = meta["win_s"], meta["win_e"]
        x = X[s + ws: s + we]  # [L,6]
        if x.shape[0] != self.seq_len:
            raise RuntimeError(f"Window length mismatch got {x.shape[0]} expected {self.seq_len}")

        x = self._apply_norm(x)
        x = torch.from_numpy(x).float()
        y = torch.tensor([self.labels[idx]], dtype=torch.long)
        return x, y







# ---------------------------
# MotionSense label mapping
# ---------------------------
MOTIONSENSE_ACT2ID = {
    "dws": 0,  # downstairs
    "ups": 1,  # upstairs
    "wlk": 2,  # walking
    "jog": 3,  # jogging
    "std": 4,  # standing
    "sit": 5,  # sitting
}
MOTIONSENSE_ID2ACT = {v: k for k, v in MOTIONSENSE_ACT2ID.items()}


def _safe_list_from_csv(s, cast=int):
    if s is None:
        return []
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(cast(tok))
    return out
class Dataset_MotionSense(Dataset):
    """
    MotionSense dataset (A: fixed cross-user split), ETT-style fast caching.

    Output:
      x: [seq_len, C] float32
      y: [1] long in 0..5

    Notes:
      - Split by subject id (cross-user)
      - Windowize within each file (subject, act, trial)
      - Optional global_std normalization fitted on TRAIN users only
    """

    def __init__(self, args, root_path, flag="train", limit_size=None):
        flag = flag.lower()
        assert flag in ["train", "val", "test"]
        self.args = args
        self.root_path = root_path
        self.flag = flag
        # ---- ds_cfg ----
        self.ds_cfg = getattr(args, "ds_cfg", {}) if hasattr(args, "ds_cfg") else {}
        dk = str(getattr(args, "dataset_key", "motionsense")).lower()
        if isinstance(self.ds_cfg, dict) and dk in self.ds_cfg and isinstance(self.ds_cfg[dk], dict):
            self.ds_cfg = self.ds_cfg[dk]

        # allow ds_cfg override root_path
        self.root_path = str(_cfg_get(self.ds_cfg, "root_path", self.root_path))

        # ---------------- window config (50Hz) ----------------
        self.seq_len = int(_cfg_get(self.ds_cfg, "seq_len", getattr(args, "seq_len", 128)))
        self.stride = int(_cfg_get(self.ds_cfg, "stride", getattr(args, "stride", max(1, self.seq_len // 2))))
        self.min_seg_len = int(_cfg_get(self.ds_cfg, "min_seg_len", getattr(args, "min_seg_len", self.seq_len)))

        # ---------------- split config (A: fixed users) ----------------
        default_test = "19,20,21,22,23,24"
        default_val  = "13,14,15,16,17,18"

        test_users = _cfg_get(self.ds_cfg, "test_users", getattr(args, "test_users", default_test))
        val_users = _cfg_get(self.ds_cfg, "val_users", getattr(args, "val_users", default_val))
        self.test_users = _safe_list_from_csv(test_users, int)
        self.val_users = _safe_list_from_csv(val_users, int)

        self.val_ratio = float(_cfg_get(self.ds_cfg, "val_ratio", getattr(args, "val_ratio", 0.0)))
        self.seed = int(_cfg_get(self.ds_cfg, "seed", getattr(args, "seed", 2024)))

        # ---------------- data config ----------------
        self.data_folder = str(_cfg_get(self.ds_cfg, "folder", _cfg_get(self.ds_cfg, "motionsense_folder",
                                                                        getattr(args, "motionsense_folder",
                                                                                "A_DeviceMotion_data")))).strip()
        self.feature_set = str(_cfg_get(self.ds_cfg, "feature_set", _cfg_get(self.ds_cfg, "motionsense_feature_set",
                                                                             getattr(args, "motionsense_feature_set",
                                                                                     "A12")))).strip().lower()
        self.combine_grav_acc = bool(_cfg_get(self.ds_cfg, "combine_grav_acc",
                                              _cfg_get(self.ds_cfg, "motionsense_combine_grav_acc",
                                                       getattr(args, "motionsense_combine_grav_acc", False))))

        # normalization
        self.norm = str(_cfg_get(self.ds_cfg, "norm", _cfg_get(self.ds_cfg, "motionsense_norm",
                                                               getattr(args, "motionsense_norm", "none")))).lower()
        self._global_mean = None
        self._global_std  = None

        # ---------------- cache config (NEW: speed only) ----------------
        cache_dir = _cfg_get(self.ds_cfg, "cache_dir",
                             _cfg_get(self.ds_cfg, "motionsense_cache_dir",
                                      getattr(args, "motionsense_cache_dir", None)))
        if cache_dir is None:
            cache_dir = os.path.join(self.root_path, "_cache_motionsense_npz")
        self.cache_dir = str(cache_dir)
        self.use_cache = bool(_cfg_get(self.ds_cfg, "use_cache",
                                       _cfg_get(self.ds_cfg, "motionsense_use_cache",
                                                getattr(args, "motionsense_use_cache", True))))

        # caches
        self._X_cache = {}     # (user, act, trial) -> np.ndarray [T, C]
        self._meta_cache = {}  # (user, act, trial) -> (user, act_id, act, trial)
        self._all_users_sorted = []

        # sample index
        self.samples = []
        self.labels = []

        # load/cache + build windows
        self._load_all_files_to_cache()
        self._build_index()

        if limit_size is not None:
            n = len(self.samples)
            n = int(n * float(limit_size)) if limit_size <= 1 else int(limit_size)
            self.samples = self.samples[:n]
            self.labels  = self.labels[:n]

        # fit train-only norm
        if self.norm == "global_std":
            self._fit_global_std_from_train_users()

        # compatibility
        self.max_seq_len = self.seq_len
        self.class_names = list(range(6))
        self.feature_df = pd.DataFrame(columns=[f"f{i}" for i in range(self._num_channels())])

    # ============================================================
    # Split logic (A)  (UNCHANGED)
    # ============================================================
    def _auto_pick_val_users_if_needed(self, remain_users):
        if len(self.val_users) > 0:
            return set(self.val_users)
        if self.val_ratio <= 0:
            return set()
        rng = np.random.RandomState(self.seed)
        tmp = list(remain_users)
        rng.shuffle(tmp)
        n_val = max(1, int(round(len(tmp) * self.val_ratio)))
        return set(tmp[:n_val])

    def _compute_split_sets(self):
        all_users = self._all_users_sorted
        test_set = set(self.test_users)

        remain = [u for u in all_users if u not in test_set]
        if len(remain) == 0:
            raise RuntimeError("All users are in test_users. No train/val left.")

        val_set = self._auto_pick_val_users_if_needed(remain)
        train_set = set([u for u in remain if u not in val_set])

        assert len(train_set & val_set) == 0, f"leak train∩val: {train_set & val_set}"
        assert len(train_set & test_set) == 0, f"leak train∩test: {train_set & test_set}"
        assert len(val_set & test_set) == 0, f"leak val∩test: {val_set & test_set}"

        return train_set, val_set, test_set

    def _pick_split_users(self):
        train_set, val_set, test_set = self._compute_split_sets()
        if self.flag == "train":
            return train_set
        if self.flag == "val":
            return val_set if len(val_set) else train_set
        return test_set

    # ============================================================
    # Feature schema (UNCHANGED)
    # ============================================================
    def _num_channels(self):
        if self.feature_set == "a12":
            return 9 if self.combine_grav_acc else 12
        if self.feature_set == "acc3":
            return 3
        if self.feature_set == "gyro3":
            return 3
        raise ValueError(f"Unknown motionsense_feature_set: {self.feature_set}")

    def _feature_cols(self, df: pd.DataFrame):
        cols = list(df.columns)
        low = [c.lower() for c in cols]

        def pick(name):
            name = name.lower()
            if name in low:
                return cols[low.index(name)]
            return None

        if self.feature_set == "a12":
            att = [pick("attitude.roll"), pick("attitude.pitch"), pick("attitude.yaw")]
            grav = [pick("gravity.x"), pick("gravity.y"), pick("gravity.z")]
            rot = [pick("rotationrate.x"), pick("rotationrate.y"), pick("rotationrate.z")]
            uacc = [pick("useracceleration.x"), pick("useracceleration.y"), pick("useracceleration.z")]
            if any(v is None for v in (att + grav + rot + uacc)):
                raise RuntimeError(f"Missing A12 columns in file. header={cols[:50]}")

            if self.combine_grav_acc:
                return att + rot + uacc, grav, uacc
            else:
                return att + grav + rot + uacc, None, None

        if self.feature_set in ["acc3", "gyro3"]:
            ax, ay, az = pick("x"), pick("y"), pick("z")
            if ax is None or ay is None or az is None:
                raise RuntimeError(f"Missing x,y,z columns. header={cols[:50]}")
            return [ax, ay, az], None, None

        raise ValueError(f"Unknown motionsense_feature_set: {self.feature_set}")

    # ============================================================
    # Cache helpers (NEW)
    # ============================================================
    def _safe(self, s: str) -> str:
        s = str(s)
        s = s.replace("/", "_").replace("\\", "_").replace(" ", "_")
        s = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", s)
        return s

    def _cache_fp_for_csv(self, fp: str, user: int, act: str, trial: int) -> str:
        # include parsing-affecting knobs to avoid accidental mismatch
        sig = f"fs={self.feature_set}_cga={int(self.combine_grav_acc)}"
        name = f"u{user}__{act}_{trial}__{self._safe(sig)}.npz"
        return os.path.join(self.cache_dir, name)

    def _cache_is_fresh(self, csv_fp: str, cache_fp: str) -> bool:
        if not os.path.exists(cache_fp):
            return False
        try:
            return os.path.getmtime(cache_fp) >= os.path.getmtime(csv_fp)
        except OSError:
            return False

    def _save_cache(self, cache_fp: str, X: np.ndarray):
        os.makedirs(os.path.dirname(cache_fp), exist_ok=True)
        np.savez_compressed(cache_fp, X=X.astype(np.float32, copy=False))

    def _load_cache(self, cache_fp: str) -> np.ndarray:
        z = np.load(cache_fp, allow_pickle=False)
        X = z["X"]
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        return X

    # ============================================================
    # File discovery + caching (speed-up only)
    # ============================================================
    def _parse_act_trial_from_dir(self, dirname: str):
        m = re.fullmatch(r"([a-z]+)_(\d+)", dirname.lower())
        if m is None:
            return None, None
        act = m.group(1)
        trial = int(m.group(2))
        if act not in MOTIONSENSE_ACT2ID:
            return None, None
        return act, trial


    def _load_one_csv_fast_new_modify(self, fp: str) -> np.ndarray:
        """
        Faster CSV read:
          - read once
          - drop Unnamed: 0 if present
          - only coerce numeric on used columns (not all columns)
          - drop rows with NaN
        Output identical to original (float32, same columns order).
        """
        df = pd.read_csv(fp)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        cols, grav_cols, uacc_cols = self._feature_cols(df)

        need_cols = list(cols)
        if grav_cols is not None:
            need_cols += list(grav_cols)
        if uacc_cols is not None:
            need_cols += list(uacc_cols)

        # keep only needed columns (reduces work)
        df = df[need_cols].copy()

        # numeric coercion ONLY for needed cols
        for c in need_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(axis=0, how="any")
        if len(df) == 0:
            return None

        X = df[cols].to_numpy(dtype=np.float32, copy=False)

        if self.feature_set == "a12" and self.combine_grav_acc:
            grav = df[grav_cols].to_numpy(dtype=np.float32, copy=False)
            uacc = df[uacc_cols].to_numpy(dtype=np.float32, copy=False)
            tot = uacc + grav
            X[:, -3:] = tot  # replace userAcc by totalAcc

        return X

    def _load_one_csv_fast(self, fp: str) -> np.ndarray:
        df = pd.read_csv(fp)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # 先确定需要的列名（只用列名，不依赖数值）
        cols, grav_cols, uacc_cols = self._feature_cols(df)

        # === 关键：保持老版语义：对“所有列”做数值化，然后按“所有列”dropna ===
        # 比你老版的逐列 for-loop 更快（向量化/内部循环）
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=0, how="any")
        if len(df) == 0:
            return None

        X = df[cols].to_numpy(dtype=np.float32, copy=False)

        if self.feature_set == "a12" and self.combine_grav_acc:
            grav = df[grav_cols].to_numpy(dtype=np.float32, copy=False)
            uacc = df[uacc_cols].to_numpy(dtype=np.float32, copy=False)
            X[:, -3:] = (uacc + grav)

        return X
    def _load_all_files_to_cache(self):
        base = os.path.join(self.root_path, self.data_folder)
        if not os.path.isdir(base):
            raise FileNotFoundError(f"Missing MotionSense folder: {base}")

        act_dirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])

        files = []
        for d in act_dirs:
            act, trial = self._parse_act_trial_from_dir(d)
            if act is None:
                continue
            dd = os.path.join(base, d)
            fps = glob.glob(os.path.join(dd, "sub_*.csv"))
            for fp in sorted(fps):
                files.append((fp, act, trial))

        if len(files) == 0:
            raise RuntimeError(f"No sub_*.csv found under {base}. Check dataset layout.")

        users_found = set()

        # make sure cache dir exists early (if enabled)
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        for (fp, act, trial) in files:
            m = re.search(r"sub_(\d+)\.csv$", os.path.basename(fp).lower())
            if m is None:
                continue
            user = int(m.group(1))
            users_found.add(user)

            key = (user, act, trial)
            cache_fp = self._cache_fp_for_csv(fp, user, act, trial)

            # 1) cache hit
            if self.use_cache and self._cache_is_fresh(fp, cache_fp):
                try:
                    X = self._load_cache(cache_fp)
                    if X is not None and X.shape[0] >= self.min_seg_len:
                        self._X_cache[key] = X
                        self._meta_cache[key] = (user, MOTIONSENSE_ACT2ID[act], act, trial)
                        continue
                except Exception:
                    pass  # rebuild

            # 2) parse CSV
            X = self._load_one_csv_fast(fp)
            if X is None:
                continue
            if X.shape[0] < self.min_seg_len:
                continue

            self._X_cache[key] = X
            self._meta_cache[key] = (user, MOTIONSENSE_ACT2ID[act], act, trial)

            # 3) save cache
            if self.use_cache:
                try:
                    self._save_cache(cache_fp, X)
                except Exception:
                    pass

        if len(self._X_cache) == 0:
            raise RuntimeError("All files empty/too short after parsing. Check min_seg_len/columns.")

        self._all_users_sorted = sorted(list(users_found))

    # ============================================================
    # Windowing + index (UNCHANGED)
    # ============================================================
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

    def _build_index(self):
        keep_users = self._pick_split_users()

        samples, labels = [], []
        for (user, act, trial), X in self._X_cache.items():
            if user not in keep_users:
                continue
            T = int(X.shape[0])
            for (ws, we) in self._windowize(T):
                samples.append({"key": (user, act, trial), "win_s": ws, "win_e": we})
                labels.append(int(MOTIONSENSE_ACT2ID[act]))

        if len(samples) == 0:
            raise RuntimeError(
                f"No windows built for flag={self.flag}. "
                f"Try smaller seq_len/stride or check split users."
            )
        self.samples = samples
        self.labels = labels

    # ============================================================
    # Normalization (train only) (UNCHANGED)
    # ============================================================
    def _fit_global_std_from_train_users(self):
        train_set, _, _ = self._compute_split_sets()

        xs = []
        for (user, act, trial), X in self._X_cache.items():
            if user not in train_set:
                continue
            T = int(X.shape[0])
            for (ws, we) in self._windowize(T):
                xs.append(X[ws:we])

        Xall = np.concatenate(xs, axis=0)
        self._global_mean = Xall.mean(axis=0, keepdims=True).astype(np.float32)
        self._global_std  = Xall.std(axis=0, keepdims=True).astype(np.float32)
        self._global_std  = np.maximum(self._global_std, 1e-6)

    def _apply_norm(self, x: np.ndarray):
        if self.norm == "global_std":
            return (x - self._global_mean) / self._global_std
        return x

    # ============================================================
    # Dataset API
    # ============================================================
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        meta = self.samples[idx]
        key = meta["key"]
        X = self._X_cache[key]
        ws, we = meta["win_s"], meta["win_e"]

        x = X[ws:we]
        if x.shape[0] != self.seq_len:
            raise RuntimeError(f"Window length mismatch got {x.shape[0]} expected {self.seq_len}")

        x = self._apply_norm(x)

        # x 已是 float32；from_numpy 后就是 float32 tensor，不需要 .float() 再拷贝
        x = torch.from_numpy(x)
        y = torch.tensor([int(self.labels[idx])], dtype=torch.long)
        return x, y