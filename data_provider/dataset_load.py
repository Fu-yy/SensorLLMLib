import os
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset
import os
import re
import glob
import numpy as np
import pandas as pd
import torch

# 你工程里已有 Dataset_CAPTURE24，就从你自己的位置 import
# 按你工程结构改这一行（常见是 data_provider.data_loader）

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


@dataclass
class Args:
    # CAPTURE-24
    seq_len: int = 500
    stride: int = 250

    capture24_train_n: int = 100
    capture24_test_n: int = 51

    capture24_keep_ratio: float = 0.05  # <- 你指定的
    seed: int = 2024

    downsample_factor: int = 2
    capture24_label_col: str = "label:WillettsSpecific2018"


def make_capture24(root_path: str, flag: str) -> Dataset_CAPTURE24:
    a = Args()
    return Dataset_CAPTURE24(a, root_path, flag=flag)


def export_split_to_npz(ds: Dataset_CAPTURE24, out_path: str, chunk_size: int = 4096, compress: bool = True):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    N = len(ds)
    L = int(ds.seq_len)
    C = 3

    X_all = np.empty((N, L, C), dtype=np.float32)
    y_all = np.empty((N,), dtype=np.int64)

    pid_all = np.empty((N,), dtype=object)
    ws_all = np.empty((N,), dtype=np.int64)
    we_all = np.empty((N,), dtype=np.int64)

    for s0 in range(0, N, chunk_size):
        s1 = min(N, s0 + chunk_size)
        for i in range(s0, s1):
            x_t, y_t = ds[i]  # torch tensors
            X_all[i] = x_t.numpy().astype(np.float32, copy=False)
            y_all[i] = int(y_t.item())

            pid, ws, we = ds.samples[i]  # (pid, ws, we)
            pid_all[i] = pid
            ws_all[i] = int(ws)
            we_all[i] = int(we)

        print(f"[export] {os.path.basename(out_path)} {s1}/{N}", flush=True)

    save_fn = np.savez_compressed if compress else np.savez
    save_fn(
        out_path,
        X=X_all,
        y=y_all,
        pid=pid_all,
        ws=ws_all,
        we=we_all,
        # 记录导出配置，方便你之后核对
        seq_len=np.array([L], dtype=np.int64),
        stride=np.array([int(ds.stride)], dtype=np.int64),
        downsample_factor=np.array([int(ds.downsample_factor)], dtype=np.int64),
        keep_ratio=np.array([float(getattr(ds, "keep_ratio", -1.0))], dtype=np.float32),
        label_scheme_col=np.array([str(ds.label_scheme_col)], dtype=object),
        train_n=np.array([int(ds.train_n)], dtype=np.int64),
        test_n=np.array([int(ds.test_n)], dtype=np.int64),
        seed=np.array([int(getattr(ds, "seed", 0))], dtype=np.int64),
    )

    print(f"[OK] saved: {out_path}")
    print(f"  X: {X_all.shape} {X_all.dtype}")
    print(f"  y: {y_all.shape} {y_all.dtype}")
    print(f"  keep_ratio={getattr(ds, 'keep_ratio', None)}  label_col={ds.label_scheme_col}")


def main():
    root_capture24 = r"D:\fuy\MyCode\SensorLLMLib\datasets\capture24\capture24"
    out_dir = r"D:\fuy\MyCode\SensorLLMLib\datasets\capture24\capture24\npz_out_keep005"
    compress = True  # 想更快加载就设 False（更占空间）

    # train
    ds_tr = make_capture24(root_capture24, "train")
    export_split_to_npz(
        ds_tr,
        os.path.join(out_dir, "capture24_train_keep005.npz"),
        chunk_size=4096,
        compress=compress,
    )

    # test
    ds_te = make_capture24(root_capture24, "test")
    export_split_to_npz(
        ds_te,
        os.path.join(out_dir, "capture24_test_keep005.npz"),
        chunk_size=4096,
        compress=compress,
    )

    print("\n✅ DONE")


if __name__ == "__main__":
    main()
