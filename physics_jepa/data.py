import os
import torch
from torch.utils.data import Dataset, Sampler, DataLoader, DistributedSampler
from tqdm import tqdm
from the_well.data import WellDataset
from the_well.data.datasets import WellMetadata
from einops import rearrange
import h5py
from pathlib import Path
from typing import Sized, Iterator, TypeVar, Optional, Tuple, Dict, List
import numpy as np
import torch.distributed as dist
import random
import weakref
from collections import OrderedDict





class WellDatasetForJEPA(Dataset):
    """
    Auto-discovers HDF5 shards and yields (context, target) windows from full trajectories.

    Assumptions:
      - Each HDF5 file contains one or more objects (top-level groups), each with a
        dataset called `data_key` containing the full trajectory.
      - Trajectory array shape is (H, W, C, T) (time is the last axis).
      - Optional `phys_key` dataset exists per object (any shape).

    Returns:
      dict with 'context': (C,T,H,W), 'target': (C,T,H,W), and 'physical_params' (tensor or empty tensor).
    """

    def __init__(
        self,
        data_dir: str | Path,
        num_frames: int,
        split: str,
        resolution: Optional[Tuple[int, int]] = None,   # (H_out, W_out)
        stride: int = None, # temporal overlap of training examples, default is num_frames
        subset_config_path: Optional[str | Path] = None, # path to config file containing subset_indices
        noise_std: float = 0.0, # standard deviation of Gaussian noise to add
        # HDF5 handle/cache tuning:
        max_open_files: int = 6,
        rdcc_nbytes: int = 512 * 1024**2,
        rdcc_nslots: int = 1_000_003,
        rdcc_w0: float = 0.75,
    ):
        if split == "val":
            split = "valid"

        self.data_dir = Path(data_dir) / "data" / split
        self.dataset_name = Path(data_dir).stem
        self.split = split
        self.num_frames = int(num_frames)
        assert self.num_frames > 0
        self.stride = stride
        self.resolution = resolution
        self.noise_std = float(noise_std)
        if self.noise_std > 0:
            print(f"Adding Gaussian noise with std {self.noise_std}", flush=True)

        # Load subset indices if provided
        self.subset_indices = None
        if subset_config_path is not None:
            subset_config_path = Path(subset_config_path)
            if subset_config_path.exists():
                import json
                with open(subset_config_path, 'r') as f:
                    config = json.load(f)
                    self.subset_indices = config.get('subset_indices', None)
                if self.subset_indices is not None:
                    print(f"Loaded {len(self.subset_indices)} subset indices from {subset_config_path}", flush=True)
            else:
                print(f"Warning: subset_config_path {subset_config_path} does not exist, using full dataset", flush=True)

        # Per-worker LRU of open files
        self._open: OrderedDict[int, tuple[h5py.File, dict]] | None = None
        self._max_open_files = int(max_open_files)
        self._rdcc = (int(rdcc_nbytes), int(rdcc_nslots), float(rdcc_w0))

        # Build flat index of (file_id, obj_id, t0) with stride=1 and non-overlapping (ctx,tgt)
        self.index, self.physical_params_idx = self._build_index()
        print(f"Found {len(self.index)} examples", flush=True)
        print(f"Physical params: {self.physical_params_idx}", flush=True)
        self._build_global_field_schema(Path(self.data_dir) / self.index[0][0])

        if len(self.index) == 0:
            raise ValueError("No valid windows found. "
                             "Check num_frames and that trajectories have at least 2*num_frames frames.")

    # ---- Discovery & indexing ----

    def _build_index(self) -> Tuple[List[tuple[int, int, int]], Dict[str, List[np.ndarray]]]:
        """
        Valid start t0 satisfy: t0 + 2*num_frames <= T.
        We step by 1 to allow maximal coverage; ctx=[t0, t0+F), tgt=[t0+F, t0+2F).
        """
        
        idx: List[tuple[int, int, int]] = []
        physical_params_idx: Dict[str, List[np.ndarray]] = {}
        
        F = self.num_frames
        paths = sorted(list(self.data_dir.rglob("*.h5")) + list(self.data_dir.rglob("*.hdf5")))
        
        for path in paths:
            with h5py.File(path, 'r') as f:
                example_scalar_field = f['t0_fields'][list(f['t0_fields'].keys())[0]]
                T = int(example_scalar_field.shape[1]) # expected shape: num_objs t h w
                max_t0 = T - 2 * F
                if max_t0 < 0:
                    continue
                stride = self.stride if self.stride is not None else F
                for obj_id in range(example_scalar_field.shape[0]):
                    for t0 in range(0, max_t0 + 1, stride):  # stride=1
                        idx.append((path.name, obj_id, t0))
                physical_params_idx[path.name] = [f['scalars'][key][()] for key in f['scalars'].keys() if key != "L"] # ignore L for active matter

        return idx, physical_params_idx
    
    def _build_global_field_schema(self, sample_path):
        field_paths, d_sizes, comp_shapes = [], [], []
        order = ["t0_fields", "t1_fields", "t2_fields"]
        with h5py.File(sample_path, "r") as f:
            for group in order:
                if group in f:
                    for name, ds in f[group].items():
                        if isinstance(ds, h5py.Dataset):
                            if not isinstance(ds, h5py.Dataset):
                                continue
                            comp = tuple(ds.shape[4:])      # () for scalars, (2,) for vectors, (2,2) for tensors...
                            d_sizes.append(int(np.prod(comp) or 1))
                            comp_shapes.append(comp)
                            field_paths.append(f"{group}/{name}")
            # basic sanity checks
            if not field_paths:
                raise RuntimeError(f"No fields found in {sample_path}")
            # take shape/dtype from the first field
            _, _, H, W = f[field_paths[0]].shape # t0_fields has shape (N, T, H, W)
            if self.dataset_name == "shear_flow":
                H = W = 256 # cut shear flow x axis in half to make square
            if self.dataset_name == "rayleigh_benard":
                H = W = 128 # take middle 128x128 square
            dtype = f[field_paths[0]].dtype

        d_sizes = np.asarray(d_sizes, dtype=np.int64)
        chan_offsets = np.concatenate(([0], np.cumsum(d_sizes)))
        self._field_paths = tuple(field_paths)
        self._d_sizes = d_sizes
        self._comp_shapes = comp_shapes
        self._chan_offsets = chan_offsets
        self._C_total = int(chan_offsets[-1])
        self._spatial_shape = (H, W)
        self._dtype = dtype

    # ---- Reading data ----

    def _get_ds_handle(self, f, state, path):
        ds_cache = state.setdefault("ds_cache", {})
        if path in ds_cache:
            return ds_cache[path]
        ds = f[path]  # fast path lookup; avoid tree walks
        try:
            ds.id.set_chunk_cache(self._rdcc[1], self._rdcc[0], self._rdcc[2])
        except Exception:
            pass
        ds_cache[path] = ds
        return ds

    # ---- Dataset API ----

    def __len__(self) -> int:
        if self.subset_indices is not None:
            return len(self.subset_indices)
        return len(self.index)

    def __getitem__(self, i):
        # Use subset index if available, otherwise use direct index
        if self.subset_indices is not None:
            actual_index = self.subset_indices[i]
        else:
            actual_index = i
        file_id, local_obj_idx, t0 = self.index[actual_index]
        F = self.num_frames

        f, state = self._open_file(file_id)  # per-worker LRU open
        H, W = self._spatial_shape
        C = self._C_total

        # Preallocate final outputs once per sample
        ctx = np.empty((F, H, W, C), dtype=self._dtype, order="C")
        tgt = np.empty((F, H, W, C), dtype=self._dtype, order="C")

       # selections: time-contiguous 2F slice
        if self.dataset_name == "shear_flow":
            h_slice = slice(None)
            w_slice = slice(0, W)
        elif self.dataset_name == "rayleigh_benard":
            h_slice = slice(192, 192+W)
            w_slice = slice(None)
        else:
            h_slice = slice(None)
            w_slice = slice(None)

        sel_2f_prefix = (local_obj_idx, slice(t0, t0 + 2*F), h_slice, w_slice)

        # per-worker cache of temporary buffers keyed by component shape
        tmp_cache = state.setdefault("twobuf_cache", {})  # e.g., {(): arr(2F,H,W), (2,): arr(2F,H,W,2), (2,2): arr(2F,H,W,2,2)}

        c0 = 0
        for path, dsize, comp_shape in zip(self._field_paths, self._d_sizes, self._comp_shapes):
            c1 = c0 + dsize
            ds = self._get_ds_handle(f, state, path)

             # ensure a reusable temp buffer of shape (2F, H, W, *comp_shape)
            need_shape = (2*F, H, W) + comp_shape
            buf = tmp_cache.get(comp_shape)
            if buf is None or buf.shape != need_shape or buf.dtype != self._dtype:
                buf = np.empty(need_shape, dtype=self._dtype, order="C")
                tmp_cache[comp_shape] = buf

            # build full source sel including component dims
            sel = sel_2f_prefix + (slice(None),) * len(comp_shape)
            ds.read_direct(buf, source_sel=sel)  # one I/O per field

            # flatten component axes to channels view and split into ctx/tgt
            view = buf.reshape(2*F, H, W, dsize)   # no copy; C-order
            c1 = c0 + dsize
            ctx[..., c0:c1] = view[:F]
            tgt[..., c0:c1] = view[F:]
            c0 = c1

        # -> torch (C, T, H, W)
        ctx_t = torch.from_numpy(ctx).permute(3, 0, 1, 2).contiguous()
        tgt_t = torch.from_numpy(tgt).permute(3, 0, 1, 2).contiguous()

        # Optional resize
        if self.resolution is not None and tuple(ctx_t.shape[-2:]) != tuple(self.resolution):
            ctx_t = torch.nn.functional.interpolate(ctx_t, size=self.resolution, mode='bilinear', align_corners=False)
            tgt_t = torch.nn.functional.interpolate(tgt_t, size=self.resolution, mode='bilinear', align_corners=False)

        # Add Gaussian noise if specified
        if self.noise_std > 0:
            noise_ctx = torch.randn_like(ctx_t) * self.noise_std
            noise_tgt = torch.randn_like(tgt_t) * self.noise_std
            ctx_t = ctx_t + noise_ctx
            tgt_t = tgt_t + noise_tgt

        return {"context": ctx_t, "target": tgt_t, "physical_params": torch.tensor(self.physical_params_idx[file_id])}

    # ---- Worker-local file LRU ----

    def _ensure_worker_state(self):
        if self._open is None:
            self._open = OrderedDict()  # file_id -> (h5file, {"_dummy": True})
            weakref.finalize(self, self._close_all)

    def _close_all(self):
        if self._open:
            for (f, _) in self._open.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._open.clear()

    def _open_file(self, file_id: int) -> tuple[h5py.File, dict]:
        self._ensure_worker_state()
        if file_id in self._open:
            f, st = self._open.pop(file_id)
            self._open[file_id] = (f, st)  # move to MRU
            return f, st

        # Evict LRU if needed
        while len(self._open) >= self._max_open_files:
            _, (old_f, _) = self._open.popitem(last=False)
            try:
                old_f.close()
            except Exception:
                pass

        path = self.data_dir / file_id
        f = h5py.File(
            path, mode='r', libver='latest', swmr=True,
            rdcc_nbytes=self._rdcc[0], rdcc_nslots=self._rdcc[1], rdcc_w0=self._rdcc[2]
        )
        st = {}
        self._open[file_id] = (f, st)
        return f, st

    def __getstate__(self):
        # Drop open handles when DataLoader forks
        st = self.__dict__.copy()
        st["_open"] = None
        return st

class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels):
        # Check if inputs are HDF5 datasets or numpy arrays
        if hasattr(embeddings, 'shape') and hasattr(embeddings, '__getitem__') and not isinstance(embeddings, np.ndarray):
            # HDF5 dataset - use memory mapping
            self.embeddings = embeddings
            self.labels = labels
            self._is_hdf5 = True
        else:
            # Numpy array - convert to tensor as before
            self.embeddings = torch.from_numpy(embeddings)
            if labels.dtype != np.object_ and labels[0].dtype != np.str_:
                self.labels = torch.from_numpy(labels)
            else:
                self.labels = labels
            self._is_hdf5 = False
        
    def __len__(self):
        return len(self.embeddings)
            
    def __getitem__(self, idx):
        if self._is_hdf5:
            # For HDF5 datasets, load data on-demand and convert to tensor
            embedding = torch.from_numpy(self.embeddings[idx])
            label = self.labels[idx]
        else:
            # For numpy arrays already converted to tensors
            embedding = self.embeddings[idx]
            label = self.labels[idx]
            
        return {'embeddings': embedding, 'label': label}

class DISCOLatentDataset(Dataset):
    def __init__(self, path, split="train"):
        self.path = Path(path) / split
        self.files = sorted(self.path.glob("batch_*.pt"))

        # load batch sizes
        self.batch_size = self._infer_batch_size()

    def _infer_batch_size(self):
        f = self.files[0]
        data = torch.load(f, weights_only=False)
        self.batch_size = data["labels"].shape[0]
        return self.batch_size

    def __len__(self):
        return len(self.files) * self.batch_size

    def __getitem__(self, idx):
        # find which batch this idx belongs to
        file_idx = idx // self.batch_size
        local_idx = idx % self.batch_size
        data = torch.load(self.files[file_idx], weights_only=False)
        return data["theta_latent"][local_idx].unsqueeze(0), data["labels"][local_idx]

class WellDatasetForMPP(Dataset):
    """
    Auto-discovers HDF5 shards and yields (context, target) windows from full trajectories.

    Assumptions:
      - Each HDF5 file contains one or more objects (top-level groups), each with a
        dataset called `data_key` containing the full trajectory.
      - Trajectory array shape is (H, W, C, T) (time is the last axis).
      - Optional `phys_key` dataset exists per object (any shape).

    Returns:
      dict with 'context': (C,T,H,W), 'target': (C,T,H,W), and 'physical_params' (tensor or empty tensor).
    """

    def __init__(
        self,
        data_dir: str | Path,
        num_frames: int,
        split: str,
        resolution: Optional[Tuple[int, int]] = None,   # (H_out, W_out)
        stride: int = None, # temporal overlap of training examples, default is num_frames
        # HDF5 handle/cache tuning:
        max_open_files: int = 6,
        rdcc_nbytes: int = 512 * 1024**2,
        rdcc_nslots: int = 1_000_003,
        rdcc_w0: float = 0.75,
    ):
        if split == "val":
            split = "valid"

        self.data_dir = Path(data_dir) / "data" / split
        self.dataset_name = Path(data_dir).stem
        self.split = split
        self.num_frames = int(num_frames)
        assert self.num_frames > 0
        self.stride = stride
        self.resolution = resolution
        self.state_labels = {
            'gray_scott_reaction_diffusion': [4, 5], # activator, inhibitor
            'shear_flow': [0, 1, 2, 3], # vx, vy, density, pressure
            'active_matter': range(11),
            'rayleigh_benard': range(4), # vx, vy, buoyancy, pressure
        }
        self.bcs = {
            'gray_scott_reaction_diffusion': [1, 1], # periodic
            'shear_flow': [1, 1], # periodic
            'active_matter': [1, 1], # periodic
            'rayleigh_benard': [1, 1], # periodic in one direction TODO how to represent this
        }

        # Per-worker LRU of open files
        self._open: OrderedDict[int, tuple[h5py.File, dict]] | None = None
        self._max_open_files = int(max_open_files)
        self._rdcc = (int(rdcc_nbytes), int(rdcc_nslots), float(rdcc_w0))

        # Build flat index of (file_id, obj_id, t0) with stride=1 and non-overlapping (ctx,tgt)
        self.index, self.physical_params_idx = self._build_index()
        print(f"Found {len(self.index)} examples", flush=True)
        print(f"Physical params: {self.physical_params_idx}", flush=True)
        self._build_global_field_schema(Path(self.data_dir) / self.index[0][0])

        if len(self.index) == 0:
            raise ValueError("No valid windows found. "
                             "Check num_frames and that trajectories have at least 2*num_frames frames.")

    # ---- Discovery & indexing ----

    def _build_index(self) -> Tuple[List[tuple[int, int, int]], Dict[str, List[np.ndarray]]]:
        """
        Valid start t0 satisfy: t0 + 2*num_frames <= T.
        We step by 1 to allow maximal coverage; ctx=[t0, t0+F), tgt=[t0+F, t0+2F).
        """
        
        idx: List[tuple[int, int, int]] = []
        physical_params_idx: Dict[str, List[np.ndarray]] = {}
        
        F = self.num_frames
        paths = sorted(list(self.data_dir.rglob("*.h5")) + list(self.data_dir.rglob("*.hdf5")))
        
        for path in paths:
            with h5py.File(path, 'r') as f:
                example_scalar_field = f['t0_fields'][list(f['t0_fields'].keys())[0]]
                T = int(example_scalar_field.shape[1]) # expected shape: num_objs t h w
                max_t0 = T - 2 * F
                if max_t0 < 0:
                    continue
                stride = self.stride if self.stride is not None else F
                for obj_id in range(example_scalar_field.shape[0]):
                    for t0 in range(0, max_t0 + 1, stride):  # stride=1
                        idx.append((path.name, obj_id, t0))
                physical_params_idx[path.name] = [f['scalars'][key][()] for key in f['scalars'].keys() if key != "L"] # ignore L for active matter

        return idx, physical_params_idx
    
    def _build_global_field_schema(self, sample_path):
        field_paths, d_sizes, comp_shapes = [], [], []
        order = ["t0_fields", "t1_fields", "t2_fields"]
        with h5py.File(sample_path, "r") as f:
            for group in order:
                if group in f:
                    for name, ds in f[group].items():
                        if isinstance(ds, h5py.Dataset):
                            if not isinstance(ds, h5py.Dataset):
                                continue
                            comp = tuple(ds.shape[4:])      # () for scalars, (2,) for vectors, (2,2) for tensors...
                            d_sizes.append(int(np.prod(comp) or 1))
                            comp_shapes.append(comp)
                            field_paths.append(f"{group}/{name}")
            # basic sanity checks
            if not field_paths:
                raise RuntimeError(f"No fields found in {sample_path}")
            # take shape/dtype from the first field
            _, _, H, W = f[field_paths[0]].shape # t0_fields has shape (N, T, H, W)
            if self.dataset_name == "shear_flow":
                H = W = 256 # cut shear flow x axis in half to make square
            if self.dataset_name == "rayleigh_benard":
                H = W = 128 # take middle 128x128 square
            dtype = f[field_paths[0]].dtype

        d_sizes = np.asarray(d_sizes, dtype=np.int64)
        chan_offsets = np.concatenate(([0], np.cumsum(d_sizes)))
        self._field_paths = tuple(field_paths)
        self._d_sizes = d_sizes
        self._comp_shapes = comp_shapes
        self._chan_offsets = chan_offsets
        self._C_total = int(chan_offsets[-1])
        self._spatial_shape = (H, W)
        self._dtype = dtype

    # ---- Reading data ----

    def _get_ds_handle(self, f, state, path):
        ds_cache = state.setdefault("ds_cache", {})
        if path in ds_cache:
            return ds_cache[path]
        ds = f[path]  # fast path lookup; avoid tree walks
        try:
            ds.id.set_chunk_cache(self._rdcc[1], self._rdcc[0], self._rdcc[2])
        except Exception:
            pass
        ds_cache[path] = ds
        return ds

    # ---- Dataset API ----

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i):
        file_id, local_obj_idx, t0 = self.index[i]
        F = self.num_frames

        f, state = self._open_file(file_id)  # per-worker LRU open
        H, W = self._spatial_shape
        C = self._C_total

        # Preallocate final outputs once per sample
        ctx = np.empty((F, H, W, C), dtype=self._dtype, order="C")
        tgt = np.empty((F, H, W, C), dtype=self._dtype, order="C")

       # selections: time-contiguous 2F slice
        if self.dataset_name == "shear_flow":
            h_slice = slice(None)
            w_slice = slice(0, W)
        elif self.dataset_name == "rayleigh_benard":
            h_slice = slice(192, 192+W)
            w_slice = slice(None)
        else:
            h_slice = slice(None)
            w_slice = slice(None)

        sel_2f_prefix = (local_obj_idx, slice(t0, t0 + 2*F), h_slice, w_slice)

        # per-worker cache of temporary buffers keyed by component shape
        tmp_cache = state.setdefault("twobuf_cache", {})  # e.g., {(): arr(2F,H,W), (2,): arr(2F,H,W,2), (2,2): arr(2F,H,W,2,2)}

        c0 = 0
        for path, dsize, comp_shape in zip(self._field_paths, self._d_sizes, self._comp_shapes):
            c1 = c0 + dsize
            ds = self._get_ds_handle(f, state, path)

             # ensure a reusable temp buffer of shape (2F, H, W, *comp_shape)
            need_shape = (2*F, H, W) + comp_shape
            buf = tmp_cache.get(comp_shape)
            if buf is None or buf.shape != need_shape or buf.dtype != self._dtype:
                buf = np.empty(need_shape, dtype=self._dtype, order="C")
                tmp_cache[comp_shape] = buf

            # build full source sel including component dims
            sel = sel_2f_prefix + (slice(None),) * len(comp_shape)
            ds.read_direct(buf, source_sel=sel)  # one I/O per field

            # flatten component axes to channels view and split into ctx/tgt
            view = buf.reshape(2*F, H, W, dsize)   # no copy; C-order
            c1 = c0 + dsize
            ctx[..., c0:c1] = view[:F]
            tgt[..., c0:c1] = view[F:]
            c0 = c1

        # -> torch (C, T, H, W)
        ctx_t = torch.from_numpy(ctx).permute(3, 0, 1, 2).contiguous()
        tgt_t = torch.from_numpy(tgt).permute(3, 0, 1, 2).contiguous()

        # Optional resize
        if self.resolution is not None and tuple(ctx_t.shape[-2:]) != tuple(self.resolution):
            ctx_t = torch.nn.functional.interpolate(ctx_t, size=self.resolution, mode='bilinear', align_corners=False)
            tgt_t = torch.nn.functional.interpolate(tgt_t, size=self.resolution, mode='bilinear', align_corners=False)
        
        if self.dataset_name == "rayleigh_benard" or self.dataset_name == "shear_flow":
            channel_order = [2, 3, 0, 1] # reorder to vx, vy, density/buoyancy, pressure
            ctx = ctx[channel_order, ...]
            tgt = tgt[channel_order, ...]

        # return {"context": ctx_t, "target": tgt_t, "physical_params": torch.tensor(self.physical_params_idx[file_id])}
        return ctx_t, torch.tensor(self.physical_params_idx[file_id]), torch.tensor(self.state_labels[self.dataset_name]), torch.tensor(self.bcs[self.dataset_name])

    # ---- Worker-local file LRU ----

    def _ensure_worker_state(self):
        if self._open is None:
            self._open = OrderedDict()  # file_id -> (h5file, {"_dummy": True})
            weakref.finalize(self, self._close_all)

    def _close_all(self):
        if self._open:
            for (f, _) in self._open.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._open.clear()

    def _open_file(self, file_id: int) -> tuple[h5py.File, dict]:
        self._ensure_worker_state()
        if file_id in self._open:
            f, st = self._open.pop(file_id)
            self._open[file_id] = (f, st)  # move to MRU
            return f, st

        # Evict LRU if needed
        while len(self._open) >= self._max_open_files:
            _, (old_f, _) = self._open.popitem(last=False)
            try:
                old_f.close()
            except Exception:
                pass

        path = self.data_dir / file_id
        f = h5py.File(
            path, mode='r', libver='latest', swmr=True,
            rdcc_nbytes=self._rdcc[0], rdcc_nslots=self._rdcc[1], rdcc_w0=self._rdcc[2]
        )
        st = {}
        self._open[file_id] = (f, st)
        return f, st

    def __getstate__(self):
        # Drop open handles when DataLoader forks
        st = self.__dict__.copy()
        st["_open"] = None
        return st


def get_dataset(
    dataset_name, 
    num_frames, 
    split="train", 
    task=None,
    include_labels=False, 
    num_examples=None, 
    predict_n_steps=False,
    n_steps=1,
    fields=None,
    resolution=None,
    balance_classes=False,
    offset=None,
    subset_config_path=None,
    noise_std=0.0,
):

    resolution = (resolution, resolution) if isinstance(resolution, int) else resolution
    well_data_dir = os.environ.get("THE_WELL_DATA_DIR")
    if well_data_dir is None:
        raise ValueError("THE_WELL_DATA_DIR environment variable is not set. "
                         "Set it to the path of The Well datasets directory.")
    return WellDatasetForJEPA(
        data_dir=Path(well_data_dir) / dataset_name,
        num_frames=num_frames,
        split=split,
        resolution=resolution,
        stride=offset,
        subset_config_path=subset_config_path,
        noise_std=noise_std,
    )

def get_dataset_metadata(dataset_name):
    well_data_dir = os.environ.get("THE_WELL_DATA_DIR")
    if well_data_dir is None:
        raise ValueError("THE_WELL_DATA_DIR environment variable is not set. "
                         "Set it to the path of The Well datasets directory.")
    dataset = WellDataset(
        well_base_path=well_data_dir,
        well_dataset_name=dataset_name,
        well_split_name="train",
        n_steps_input=1,
        n_steps_output=1,
        use_normalization=True,
    )
    if dataset_name == "active_matter":
        dataset.metadata.constant_scalar_names = ["zeta", "alpha"] # don't predict L, it's always the same
    return dataset.metadata

def get_train_dataloader_from_cfg(cfg, stage="train", rank=None, world_size=None):
    return get_train_dataloader(
        cfg.dataset.name,
        cfg.dataset.num_frames,
        cfg.dataset.get("num_examples", None),
        cfg[stage].batch_size,
        include_labels=(stage == "ft" or cfg[stage].include_labels),
        predict_n_steps=cfg[stage].get("predict_n_steps", False),
        n_steps=cfg[stage].get("n_steps", 1),
        rank=rank,
        world_size=world_size,
        task=cfg[stage].get("task", None), # no task for pretraining
        fields=cfg[stage].get("fields", None),
        balance_classes=cfg[stage].get("balance_classes", False),
        resolution=cfg.dataset.get("resolution", None),
        offset=cfg.dataset.get("offset", None),
        subset_config_path=cfg.dataset.get("subset_config_path", None),
        noise_std=cfg[stage].get("noise_std", 0.0),
    )

def get_val_dataloader_from_cfg(cfg, stage="train", rank=None, world_size=None):
    return get_val_dataloader(
        cfg.dataset.name,
        cfg.dataset.num_frames,
        cfg.dataset.get("num_examples", None),
        cfg[stage].batch_size,
        include_labels=(stage == "ft" or cfg[stage].include_labels),
        predict_n_steps=cfg[stage].get("predict_n_steps", False),
        n_steps=cfg[stage].get("n_steps", 1),
        rank=rank,
        world_size=world_size,
        task=cfg[stage].get("task", None),
        fields=cfg[stage].get("fields", None),
        balance_classes=False,
        resolution=cfg.dataset.get("resolution", None),
        offset=cfg.dataset.get("offset", None),
        noise_std=cfg[stage].get("noise_std", 0.0),
    )

def get_train_dataloader(
        dataset_name,
        num_frames,
        num_examples,
        batch_size, 
        task=None,
        rank=0,
        world_size=1, 
        seed=42, 
        shuffle=True,
        num_workers=4, 
        persistent_workers=True, 
        pin_memory=True, 
        prefetch_factor=2,
        include_labels=False,
        predict_n_steps=False,
        n_steps=1,
        fields=None,
        balance_classes=False,
        resolution=None,
        offset=None,
        subset_config_path=None,
        noise_std=0.0,
    ):
    dataset = get_dataset(dataset_name,
                          num_frames, 
                          split="train", 
                          include_labels=include_labels, 
                          num_examples=num_examples, 
                          predict_n_steps=predict_n_steps, 
                          task=task, 
                          n_steps=n_steps,
                          fields=fields,
                          balance_classes=balance_classes,
                          resolution=resolution,
                          offset=offset,
                          subset_config_path=subset_config_path,
                          noise_std=noise_std,
                        )
    if rank == 0:
        print(f"total number of block pairs in train dataset: {len(dataset)}")
    if world_size == 1:
        sampler = None
    else:
        sampler = DistributedSampler(
            dataset=dataset,
            drop_last=True,
            shuffle=shuffle,
            rank=rank,
            num_replicas=world_size,
            seed=seed,
        )
    
    def worker_init_fn(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        worker_seed = worker_seed + rank if rank is not None else worker_seed

        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=True,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    return loader

def get_val_dataloader(
        dataset_name,
        num_frames,
        num_examples,
        batch_size, 
        task=None,
        rank=0, 
        world_size=1, 
        seed=42, 
        shuffle=False,
        persistent_workers=True, 
        pin_memory=True, 
        prefetch_factor=2,
        include_labels=False,
        predict_n_steps=False,
        n_steps=1,
        fields=None,
        balance_classes=False,
        resolution=None,
        offset=None,
        noise_std=0.0,
    ):
    dataset = get_dataset(dataset_name, 
                          num_frames, 
                          split="val", 
                          include_labels=include_labels, 
                          num_examples=num_examples, 
                          predict_n_steps=predict_n_steps, 
                          task=task,
                          n_steps=n_steps,
                          fields=fields,
                          balance_classes=balance_classes,
                          resolution=resolution,
                          offset=offset,
                          noise_std=noise_std,
                        )
    if world_size == 1:
        sampler = None
    else:
        sampler = DistributedSampler(
            dataset=dataset,
            drop_last=True,
            shuffle=shuffle,
            rank=rank,
            num_replicas=world_size,
            seed=seed,
        )
    
    def worker_init_fn(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        worker_seed = worker_seed + rank if world_size > 1 else worker_seed

        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    val_loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    return val_loader