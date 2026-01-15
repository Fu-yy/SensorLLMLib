import os

# from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
#     MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_MHealth
from data_provider.data_loader import Dataset_MHealth, Dataset_CAPTURE24, \
    Dataset_USCHAD, Dataset_UCIHAR_Official, Dataset_PAMAP50, Dataset_PAMAP, Dataset_WISDM, Dataset_HHAR_1user, \
    Dataset_HHAR_cross_user, Dataset_MotionSense
from data_provider.mhealth import collate_fn_mhealth
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    # 'ETTh1': Dataset_ETT_hour,
    # 'ETTh2': Dataset_ETT_hour,
    # 'ETTm1': Dataset_ETT_minute,
    # 'ETTm2': Dataset_ETT_minute,
    # 'custom': Dataset_Custom,
    # 'm4': Dataset_M4,
    # 'PSM': PSMSegLoader,
    # 'MSL': MSLSegLoader,
    # 'SMAP': SMAPSegLoader,
    # 'SMD': SMDSegLoader,
    # 'SWAT': SWATSegLoader,
    # 'UEA': UEAloader,
    'uschad' :Dataset_USCHAD,
    'ucihar':Dataset_UCIHAR_Official,
    'pamap50':Dataset_PAMAP50,
    'pamap':Dataset_PAMAP,
    'capture24':Dataset_CAPTURE24,
    'mhealth': Dataset_MHealth,
    'USCHAD': Dataset_USCHAD,
    'UCIHAR': Dataset_UCIHAR_Official,
    'PAMAP50': Dataset_PAMAP50,
    'PAMAP': Dataset_PAMAP,
    'CAPTURE24': Dataset_CAPTURE24,
    'MHealth': Dataset_MHealth,
    'WISDM': Dataset_WISDM,
    'HHAR_1user': Dataset_HHAR_1user,
    'HHAR_cross_user': Dataset_HHAR_cross_user,
    'MotionSense': Dataset_MotionSense,
}

# def _npz_exists(npz_path: str) -> bool:
#     return npz_path is not None and os.path.exists(npz_path)
#
# def build_dataset(args, flag):
#     """
#     Unified dataset builder:
#     - if npz exists: use Dataset_FromNPZ (FAST)
#     - else: use raw dataset (SLOW) and optionally build npz
#     """
#
#     dataset_name = args.data.lower()
#
#     # -------------------------------
#     # CAPTURE-24
#     # -------------------------------
#     if dataset_name == "capture24":
#         npz_path = os.path.join(
#             args.npz_out,
#             f"capture24_{flag}.npz"
#         )
#
#         if _npz_exists(npz_path):
#             print(f"[DataFactory] load CAPTURE-24 from NPZ: {npz_path}")
#             return Dataset_FromNPZ(npz_path)
#
#         print("[DataFactory] load CAPTURE-24 from RAW csv.gz (slow)")
#         ds = Dataset_CAPTURE24(args, args.capture24_root, flag=flag)
#
#         # 可选：自动缓存
#         # if getattr(args, "auto_build_npz", True):
#         #     export_dataset_to_npz(ds, npz_path)
#
#         return ds
#
#     # -------------------------------
#     # USC-HAD
#     # -------------------------------
#     if dataset_name == "uschad":
#         npz_path = os.path.join(
#             args.npz_out,
#             f"uschad_{flag}.npz"
#         )
#
#         if _npz_exists(npz_path):
#             print(f"[DataFactory] load USC-HAD from NPZ: {npz_path}")
#             return Dataset_FromNPZ(npz_path)
#
#         print("[DataFactory] load USC-HAD from RAW mat (slow)")
#         ds = Dataset_USCHAD(args, args.uschad_root, flag=flag)
#
#         if getattr(args, "auto_build_npz", True):
#             export_dataset_to_npz(ds, npz_path)
#
#         return ds
#
#     # -------------------------------
#     # PAMAP2
#     # -------------------------------
#     if dataset_name == "pamap2":
#         variant = args.pamap_variant
#         npz_path = os.path.join(
#             args.npz_out,
#             f"pamap2_{flag}_{variant}.npz"
#         )
#
#         if _npz_exists(npz_path):
#             print(f"[DataFactory] load PAMAP2 from NPZ: {npz_path}")
#             return Dataset_FromNPZ(npz_path)
#
#         print("[DataFactory] load PAMAP2 from RAW dat (slow)")
#         ds = Dataset_PAMAP2(args, args.pamap_root, flag=flag)
#
#         if getattr(args, "auto_build_npz", True):
#             export_dataset_to_npz(ds, npz_path)
#
#         return ds
#
#     raise ValueError(f"Unknown dataset: {args.data}")
#
HHAR_DATASETS=['MHealth','USCHAD','UCIHAR','PAMAP50','PAMAP','CAPTURE24','WISDM','HHAR_1user','HHAR_cross_user','MotionSense']

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        if args.data in HHAR_DATASETS:
            drop_last = False
            data_set = Data(
                args=args,
                root_path=args.root_path,
                flag=flag,
            )
            # data_set = build_dataset(args, flag)

            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=collate_fn_mhealth
            )
            return data_set, data_loader
        elif args.data == 'UEA':
            drop_last = False
            data_set = Data(
                args=args,
                root_path=args.root_path,
                flag=flag,
            )

            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
            )
            return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
