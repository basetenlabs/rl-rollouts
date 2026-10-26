#!/usr/bin/env python3
"""Publish checkpoints for live weight updates (see README.md).

  full    upload a complete Hugging Face checkpoint as a new anchor
  delta   encode --new against --parent as arc_v2 shards and publish them
  verify  apply a delta locally and check every Adler-32 checksum
  from-miles-delta  republish a Miles disk-delta version (xor, adler32) as arc_v2
  startup write STARTUP_SNAPSHOT, naming the snapshot a new deployment loads first

Snapshots are built and checked locally first (delta shards are staged under
$TMPDIR), uploaded with create-only puts so a published snapshot is never
overwritten, and snapshot_info.json is written last. Publishing prints the
hot-load request that loads the snapshot. --out-dir DIR writes the same layout
to a local directory instead of a bucket. Files over 64 MiB use multipart uploads,
with up to 128 parts of all files in flight (about 8 GiB of RAM).
--bucket takes s3://BUCKET/PREFIX (Amazon S3) or gs://BUCKET/PREFIX (Google Cloud
Storage through its S3-compatible XML API, with HMAC keys as the AWS credentials).

Requires: pip install boto3 numpy safetensors zstandard
"""
import argparse
import itertools
import json
import mmap
import os
import shutil
import struct
import tempfile
import threading
import zlib
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from urllib.parse import urlparse

import numpy as np
import zstandard
from safetensors import safe_open

INDEX = "model.safetensors.index.json"
PART = 64 << 20  # multipart part size: files up to 625 GiB (10,000 parts)
UPLOADS = 16  # files uploaded concurrently; their parts share the PARTS pool
PARTS = 128  # parts in flight across all files, each buffering one PART
GCS_ENDPOINT = "https://storage.googleapis.com"


def fail(message):
    raise SystemExit(f"error: {message}")


class Checkpoint:
    """Tensor headers of a safetensors directory and zero-copy views of their bytes."""

    def __init__(self, hf_dir):
        self.dir, self.index, self.maps = hf_dir, {}, {}
        for shard in sorted(f for f in os.listdir(hf_dir) if f.endswith(".safetensors")):
            with open(os.path.join(hf_dir, shard), "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(size))
                self.maps[shard] = np.frombuffer(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ), np.uint8)
            header.pop("__metadata__", None)
            for name, entry in header.items():
                if name in self.index:
                    fail(f"{hf_dir}: tensor {name} is in both {self.index[name][0]} and {shard}")
                self.index[name] = (shard, entry, 8 + size)
        if not self.index:
            fail(f"{hf_dir} has no *.safetensors files")

    def view(self, name):
        """Raw bytes of one tensor, so every dtype (bf16, fp8, ...) is handled alike."""
        shard, entry, start = self.index[name]
        lo, hi = entry["data_offsets"]
        buf = self.maps[shard][start + lo:start + hi]
        return buf if buf.size == hi - lo else fail(f"{self.dir}/{shard} is truncated")


def same_layout(parent, new):
    if parent.keys() != new.keys():
        fail(f"tensor names differ, e.g. {sorted(parent.keys() ^ new.keys())[:5]}")
    for name in sorted(parent):
        a, b = parent[name][1], new[name][1]
        if (a["dtype"], a["shape"]) != (b["dtype"], b["shape"]):
            fail(f"{name}: parent {a['dtype']}{a['shape']} != new {b['dtype']}{b['shape']}")


def adler32(buf):
    return str(zlib.adler32(buf) & 0xFFFFFFFF)


def write_shard(path, entries):
    """Write ``[(name, (frame, checksum))]`` as a safetensors file of uint8 frames."""
    header, offset = {"__metadata__": {name: checksum for name, (_, checksum) in entries}}, 0
    for name, (frame, _) in entries:
        header[name] = {"dtype": "U8", "shape": [len(frame)], "data_offsets": [offset, offset + len(frame)]}
        offset += len(frame)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)  # safetensors aligns data to 8 bytes
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        f.writelines(frame for _, (frame, _) in entries)


def encode_delta(parent_dir, new_dir, out_dir, shard_of, workers=None):
    """Write arc_v2 shards; ``shard_of`` maps each tensor to its anchor shard file.

    The service decodes a delta shard against the anchor shard of the same file
    name, so output follows the anchor's layout, not --parent's or --new's.
    Tensors are encoded in parallel; each worker holds one tensor-sized XOR
    buffer, and a shard's compressed frames stay in memory until it is written.
    """
    parent, new = Checkpoint(parent_dir), Checkpoint(new_dir)
    same_layout(parent.index, new.index)
    if shard_of.keys() != parent.index.keys():
        fail(f"anchor index and --parent tensors differ, e.g. {sorted(shard_of.keys() ^ parent.index.keys())[:5]}")
    local = threading.local()  # zstd contexts are not thread-safe

    def encode(name):
        cur = new.view(name)
        if getattr(local, "buf", None) is None or local.buf.size < cur.size:  # reuse: fresh pages are slow
            local.buf = np.empty(cur.size, np.uint8)
        diff = np.bitwise_xor(parent.view(name), cur, out=local.buf[:cur.size])
        if not diff.any():
            return None  # unchanged tensors are omitted
        local.cctx = getattr(local, "cctx", None) or zstandard.ZstdCompressor(level=3)
        return local.cctx.compress(diff), adler32(cur)

    names = sorted(parent.index, key=lambda n: (shard_of[n], parent.index[n][0], parent.index[n][1]["data_offsets"]))
    written = []
    with ThreadPoolExecutor(workers or os.cpu_count()) as pool:
        for shard, group in itertools.groupby(zip(names, pool.map(encode, names)), lambda r: shard_of[r[0]]):
            entries = [(name, result) for name, result in group if result]
            if entries:
                write_shard(os.path.join(out_dir, shard), entries)
                written.append(shard)
    if not written:
        fail("--new is identical to --parent; nothing to publish")
    return written


def verify_delta(parent_dir, delta_dir, new_dir=None, shard_of=None, workers=None):
    parent, delta, new = Checkpoint(parent_dir), Checkpoint(delta_dir), new_dir and Checkpoint(new_dir)
    if new:
        same_layout(parent.index, new.index)
    checksums, local = {}, threading.local()
    for shard in delta.maps:
        with safe_open(os.path.join(delta_dir, shard), framework="np") as f:  # also validates the container
            checksums.update(f.metadata() or {})

    def check(name):
        shard = delta.index[name][0]
        if name not in parent.index or (shard_of and shard_of[name] != shard):
            fail(f"{shard}: {name} is not a tensor of the parent's {shard}")
        old = parent.view(name)
        local.dctx = getattr(local, "dctx", None) or zstandard.ZstdDecompressor()
        diff = np.frombuffer(local.dctx.decompress(delta.view(name), max_output_size=old.size), np.uint8)
        if diff.size != old.size:
            fail(f"{name}: delta is {diff.size} bytes, parent tensor is {old.size}")
        cur = np.bitwise_xor(old, diff)
        if adler32(cur) != checksums.get(name):
            fail(f"{name}: Adler-32 {adler32(cur)} != recorded {checksums.get(name)}")
        if new and not np.array_equal(cur, new.view(name)):
            fail(f"{name}: decoded tensor differs from --new")

    def unchanged(name):
        if not np.array_equal(parent.view(name), new.view(name)):
            fail(f"{name} differs in --new but is missing from the delta")

    with ThreadPoolExecutor(workers or os.cpu_count()) as pool:
        list(pool.map(check, sorted(delta.index)))
        list(pool.map(unchanged, sorted(parent.index.keys() - delta.index.keys() if new else ())))
    print(f"verified {len(delta.index)} changed tensor(s) in {len(delta.maps)} shard(s)")


def convert_miles_delta(version_dir, out_dir, shard_of):
    """Regroup a Miles disk-delta version into arc_v2 shards named after the anchor's shards.

    With ``xor`` encoding each Miles entry is already zstd(previous XOR new) of the whole
    tensor, so frames are copied unchanged; only the container and the checksum spelling
    (hex to decimal Adler-32) change. The parent bytes are not available here, so the
    delta is checked by the service on apply rather than decoded locally.
    """
    with open(os.path.join(version_dir, INDEX)) as f:
        meta = json.load(f).get("metadata", {})
    found = tuple(meta.get(k) for k in ("delta_encoding", "compression_format", "checksum_format"))
    if found != ("xor", "zstd", "adler32"):
        fail(f"{version_dir} is {found}; run Miles with --update-weight-delta-encoding xor "
             "--update-weight-delta-checksum adler32")
    src, checksums, by_shard = Checkpoint(version_dir), {}, {}
    for shard in src.maps:
        with safe_open(os.path.join(version_dir, shard), framework="np") as f:
            checksums.update(f.metadata() or {})
    for name in sorted(src.index):
        if name not in shard_of or src.index[name][1]["dtype"] != "U8":
            fail(f"{name}: not an anchor tensor, or not a uint8 delta entry")
        by_shard.setdefault(shard_of[name], []).append((name, (src.view(name), str(int(checksums[name], 16)))))
    for shard, entries in by_shard.items():
        write_shard(os.path.join(out_dir, shard), entries)
    return sorted(by_shard)


def gcs_create_only(request, **_):
    """Google Cloud Storage spells S3's create-only ``If-None-Match: *`` as generation 0."""
    if request.headers.get("If-None-Match") == "*":
        del request.headers["If-None-Match"]
        request.headers["x-goog-if-generation-match"] = "0"


class Store:
    """s3:// or gs://BUCKET/PREFIX, or a local directory with the same layout (``url=None``)."""

    def __init__(self, url, out_dir, endpoint_url):
        self.root = out_dir
        if url:
            import boto3
            from botocore.config import Config
            p = urlparse(url)
            if p.scheme not in ("s3", "gs") or not p.netloc:
                fail("--bucket must look like s3://BUCKET/PREFIX or gs://BUCKET/PREFIX")
            options = {}
            if p.scheme == "gs":  # GCS uses its own CRC32C/MD5, not S3's flexible checksums
                endpoint_url = endpoint_url or GCS_ENDPOINT
                options = {"s3": {"addressing_style": "path"},
                           "request_checksum_calculation": "when_required",
                           "response_checksum_validation": "when_required"}
            self.s3 = boto3.client("s3", endpoint_url=endpoint_url,
                                   region_name="us-east-1" if p.scheme == "gs" else None,
                                   config=Config(max_pool_connections=UPLOADS + PARTS, **options))
            self.parts = ThreadPoolExecutor(PARTS)
            if p.scheme == "gs":
                self.s3.meta.events.register("before-sign.s3", gcs_create_only)
            self.bucket = p.netloc
            self.prefix = p.path.strip("/") + "/" if p.path.strip("/") else ""

    def get(self, rel):
        if self.root:
            path = os.path.join(self.root, rel)
            return open(path, "rb").read() if os.path.exists(path) else None
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=self.prefix + rel)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def put(self, rel, path, create_only=True):
        if self.root:
            dst = os.path.join(self.root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if create_only and os.path.exists(dst):
                fail(f"{dst} already exists")
            shutil.copyfile(path, dst + ".tmp")
            return os.replace(dst + ".tmp", dst)
        key = {"Bucket": self.bucket, "Key": self.prefix + rel}
        cond = {"IfNoneMatch": "*"} if create_only else {}  # create-only: never overwrite
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size <= PART:
                return self.s3.put_object(Body=f, **key, **cond)
            upload = self.s3.create_multipart_upload(**key)["UploadId"]

            def part(fd, n):
                body = os.pread(fd, PART, (n - 1) * PART)
                if len(body) != min(PART, size - (n - 1) * PART):
                    raise OSError(f"{path}: short read of part {n}")
                return {"PartNumber": n, "ETag": self.s3.upload_part(
                    Body=body, PartNumber=n, UploadId=upload, **key)["ETag"]}

            futures = []
            try:  # the condition is checked when the multipart upload completes
                for n in range(1, -(-size // PART) + 1):
                    futures.append(self.parts.submit(part, f.fileno(), n))
                done, _ = wait(futures, return_when=FIRST_EXCEPTION)
                for future in done:  # raise the first failure before waiting on earlier parts
                    future.result()
                parts = [future.result() for future in futures]
                self.s3.complete_multipart_upload(UploadId=upload, MultipartUpload={"Parts": parts}, **key, **cond)
            except BaseException:
                for future in futures:
                    future.cancel()
                wait(futures)  # no part may land after the abort or read a closed file
                self.s3.abort_multipart_upload(UploadId=upload, **key)
                raise


def hot_load_request(identity, info):
    """The hot-load request body that loads a published snapshot (README.md, "Update")."""
    body = {"identity": identity}
    if info["snapshot_type"] == "incremental":
        body["incremental_snapshot_metadata"] = info["incremental_snapshot_metadata"]
    return body


def publish(store, args, work, files, info):
    """Upload snapshot files, then snapshot_info.json."""
    with open(os.path.join(work, "snapshot_info.json"), "w") as f:
        json.dump(info, f)
    verb = "would upload" if args.dry_run else "upload"

    def put(item):
        print(f"{verb} {args.identity}/{item[0]}", flush=True)
        if not args.dry_run:
            store.put(f"{args.identity}/{item[0]}", item[1])

    with ThreadPoolExecutor(UPLOADS) as pool:
        list(pool.map(put, files))
    put(("snapshot_info.json", os.path.join(work, "snapshot_info.json")))
    print(f"hot-load request: {json.dumps(hot_load_request(args.identity, info))}")


def write_startup(store, identity):
    """Name a published snapshot in STARTUP_SNAPSHOT, replacing any earlier one."""
    info = json.loads(store.get(f"{identity}/snapshot_info.json") or "null") or fail(
        f"{identity} has no snapshot_info.json; publish it first")
    with tempfile.TemporaryDirectory() as work:
        path = os.path.join(work, "STARTUP_SNAPSHOT")
        with open(path, "w") as f:
            json.dump(hot_load_request(identity, info), f)
        store.put("STARTUP_SNAPSHOT", path, create_only=False)
    print(f"STARTUP_SNAPSHOT names {identity}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    full = sub.add_parser("full", help="publish a complete checkpoint as a new anchor")
    full.add_argument("--dir", required=True, metavar="HF_DIR", help=f"complete checkpoint with {INDEX}")
    delta = sub.add_parser("delta", help="publish an arc_v2 delta against a published parent")
    delta.add_argument("--parent", required=True, metavar="HF_DIR", help="exact weights --previous published")
    delta.add_argument("--new", required=True, metavar="HF_DIR", help="same tensor names, shapes and dtypes")
    miles = sub.add_parser("from-miles-delta", help="republish one Miles disk-delta version without re-encoding")
    miles.add_argument("--version-dir", required=True, metavar="DIR", help="Miles weight_vNNNNNN directory")
    startup = sub.add_parser("startup", help="name the published snapshot a new deployment loads first")
    startup.add_argument("--identity", required=True, metavar="ID", help="published snapshot directory name")
    for p in (delta, miles):
        p.add_argument("--previous", required=True, metavar="ID", help="parent snapshot identity")
        p.add_argument("--base", required=True, metavar="ID", help="full anchor this chain started from")
    for p in (full, delta, miles, startup):
        where = p.add_mutually_exclusive_group(required=True)
        where.add_argument("--bucket", metavar="URL", help="s3:// or gs://BUCKET/PREFIX holding the snapshots")
        where.add_argument("--out-dir", metavar="DIR", help="local directory used in place of the bucket")
        p.add_argument("--endpoint-url", metavar="URL", help="endpoint of an S3-compatible store")
    for p in (full, delta, miles):
        p.add_argument("--identity", required=True, metavar="ID", help="new, unique snapshot directory name")
        p.add_argument("--policy-version", required=True, metavar="N", type=int, help="integer response label")
        p.add_argument("--dry-run", action="store_true", help="build and verify, print the plan, write nothing")
    check = sub.add_parser("verify", help="apply a delta locally and check it")
    check.add_argument("--parent", required=True, metavar="HF_DIR", help="checkpoint the delta encodes against")
    check.add_argument("--delta", required=True, metavar="DIR", help="directory of arc_v2 delta shards")
    check.add_argument("--new", metavar="HF_DIR", help="also require the result to equal this checkpoint")
    for p in (delta, check):
        p.add_argument("--workers", metavar="N", type=int, default=os.cpu_count(),
                       help="tensors processed in parallel, each holding about one tensor copy (default: CPUs)")
    args = parser.parse_args()
    if args.command == "verify":
        return verify_delta(args.parent, args.delta, args.new, workers=args.workers)

    if args.identity in ("", ".", "..") or "/" in args.identity:
        fail("--identity must be a single path component")
    store = Store(args.bucket, args.out_dir, args.endpoint_url)
    if args.command == "startup":
        return write_startup(store, args.identity)
    if store.get(f"{args.identity}/snapshot_info.json") is not None:
        fail(f"snapshot {args.identity} already exists; published snapshots are immutable")
    info = {"identity": args.identity, "snapshot_type": "full", "policy_version": args.policy_version}
    with tempfile.TemporaryDirectory() as work:
        if args.command == "full":
            files = sorted((os.path.relpath(os.path.join(d, f), args.dir).replace(os.sep, "/"),
                            os.path.join(d, f)) for d, _, names in os.walk(args.dir) for f in names)
            files = [(rel, path) for rel, path in files if rel != "snapshot_info.json"
                     and not any(part.startswith(".") for part in rel.split("/"))]
            mapped = json.loads(Store(None, args.dir, None).get(INDEX) or "{}").get("weight_map", {})
            if mapped.keys() != Checkpoint(args.dir).index.keys():
                fail(f"{args.dir} needs a {INDEX} listing exactly the tensors in its shards")
            return publish(store, args, work, files, info)

        parent_info = json.loads(store.get(f"{args.previous}/snapshot_info.json") or "null") or fail(
            f"parent {args.previous} has no snapshot_info.json; publish it first")
        base = (args.previous if parent_info["snapshot_type"] == "full" else
                parent_info["incremental_snapshot_metadata"]["base_full_snapshot_identity"])
        if args.base != base:
            fail(f"--base {args.base} does not match the parent's anchor {base}")
        shard_of = json.loads(store.get(f"{base}/{INDEX}") or "null") or fail(f"anchor {base} has no {INDEX}")
        shard_of = shard_of["weight_map"]
        if args.command == "delta":
            shards = encode_delta(args.parent, args.new, work, shard_of, args.workers)
            verify_delta(args.parent, work, shard_of=shard_of, workers=args.workers)
        else:
            shards = convert_miles_delta(args.version_dir, work, shard_of)
        info["snapshot_type"] = "incremental"
        info["incremental_snapshot_metadata"] = {  # "alder32" is the historical wire spelling
            "previous_snapshot_identity": args.previous, "base_full_snapshot_identity": base,
            "compression_format": "arc_v2", "checksum_format": "alder32"}
        publish(store, args, work, [(s, os.path.join(work, s)) for s in shards], info)


if __name__ == "__main__":
    main()
