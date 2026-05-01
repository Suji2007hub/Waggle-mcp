from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import zipfile
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from waggle.errors import ValidationFailure
from waggle.models import (
    AbhiChunkLoadResult,
    AbhiDiffResult,
    AbhiExportResult,
    AbhiImportResult,
    AbhiInspectResult,
    AbhiMergeResult,
    AbhiQueryResult,
    AbhiValidationResult,
)

ABHI_SPEC_VERSION = "2.0.0"
ABHI_MAJOR_VERSION = 2
ABHI_ENCRYPTION_ALGORITHM = "aes-256-gcm"
ABHI_SIGNATURE_ALGORITHM = "ed25519"
ABHI_CHUNK_NODE_LIMIT = 64
ABHI_TRANSCRIPTS_MEMBER = "transcripts.jsonl"
ABHI_NODES_MEMBER = "nodes.jsonl"
ABHI_EDGES_MEMBER = "edges.jsonl"
ABHI_CONTEXT_WINDOWS_MEMBER = "context_windows.jsonl"
ABHI_MANIFEST_MEMBER = "manifest.json"
ABHI_SIGNATURE_MEMBER = "signatures/content.ed25519"
ABHI_PUBLIC_KEY_MEMBER = "signatures/public_key.pem"
ABHI_DETERMINISTIC_ZIP_TIMESTAMP = (2000, 1, 1, 0, 0, 0)

def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash_with_prefix(value: str) -> str:
    return value if value.startswith("sha256:") else f"sha256:{value}"


def _record_lines(records: list[dict[str, Any]]) -> bytes:
    if not records:
        return b""
    return b"".join(_canonical_json(record) + b"\n" for record in records)


def _stable_record_sort_key(record: dict[str, Any], *fields: str) -> tuple[str, ...]:
    values = [str(record.get(field, "")) for field in fields]
    values.append(_canonical_json(record).decode("utf-8"))
    return tuple(values)


def _sorted_records(records: list[dict[str, Any]], *fields: str) -> list[dict[str, Any]]:
    return sorted((deepcopy(record) for record in records), key=lambda record: _stable_record_sort_key(record, *fields))


def _deterministic_zip_info(member_name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(member_name, date_time=ABHI_DETERMINISTIC_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    info.extra = b""
    return info


def _archive_writestr(archive: zipfile.ZipFile, member_name: str, payload: bytes) -> None:
    archive.writestr(_deterministic_zip_info(member_name), payload)


def _parse_lines(payload: bytes) -> list[dict[str, Any]]:
    if not payload:
        return []
    result: list[dict[str, Any]] = []
    for line in payload.decode("utf-8").splitlines():
        text = line.strip()
        if text:
            result.append(json.loads(text))
    return result


def _scope_match(record: dict[str, Any], *, project: str, agent_id: str, session_id: str) -> bool:
    if project.strip() and str(record.get("project", "")).strip() != project.strip():
        return False
    if agent_id.strip() and str(record.get("agent_id", "")).strip() != agent_id.strip():
        return False
    if session_id.strip() and str(record.get("session_id", "")).strip() != session_id.strip():
        return False
    return True


def _redact_text(text: str, patterns: list[str]) -> str:
    redacted = text
    for pattern in patterns:
        redacted = re.sub(pattern, "[REDACTED]", redacted)
    return redacted


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=600_000)
    return kdf.derive(passphrase.encode("utf-8"))


def _encrypt_bytes(payload: bytes, *, passphrase: str) -> dict[str, Any]:
    salt = os.urandom(16)
    nonce = os.urandom(12)
    key = _derive_key(passphrase, salt)
    ciphertext = AESGCM(key).encrypt(nonce, payload, None)
    return {
        "algorithm": ABHI_ENCRYPTION_ALGORITHM,
        "kdf": "pbkdf2-hmac-sha256",
        "iterations": 600_000,
        "salt_b64": base64.b64encode(salt).decode("ascii"),
        "nonce_b64": base64.b64encode(nonce).decode("ascii"),
        "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
    }


def _decrypt_bytes(payload: dict[str, Any], *, passphrase: str) -> bytes:
    if not passphrase:
        raise ValidationFailure("This .abhi file is encrypted. Provide a passphrase.")
    salt = base64.b64decode(str(payload["salt_b64"]))
    nonce = base64.b64decode(str(payload["nonce_b64"]))
    ciphertext = base64.b64decode(str(payload["ciphertext_b64"]))
    key = _derive_key(passphrase, salt)
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, None)
    except Exception as exc:
        raise ValidationFailure("Could not decrypt .abhi payload. Check the passphrase.") from exc


def _read_member(archive: zipfile.ZipFile, manifest: dict[str, Any], member_name: str, *, passphrase: str) -> bytes:
    metadata = dict(manifest.get("members", {}).get(member_name, {}))
    if member_name not in archive.namelist():
        return b""
    raw = archive.read(member_name)
    if metadata.get("encrypted"):
        payload = json.loads(raw.decode("utf-8"))
        return _decrypt_bytes(payload, passphrase=passphrase)
    return raw


def _write_member(
    archive: zipfile.ZipFile,
    manifest: dict[str, Any],
    member_name: str,
    payload: bytes,
    *,
    passphrase: str,
) -> None:
    member_meta = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "encrypted": bool(passphrase),
    }
    if passphrase:
        encrypted = _encrypt_bytes(payload, passphrase=passphrase)
        member_meta["encryption"] = {key: value for key, value in encrypted.items() if key != "ciphertext_b64"}
        _archive_writestr(archive, member_name, _canonical_json(encrypted))
    else:
        _archive_writestr(archive, member_name, payload)
    manifest.setdefault("members", {})[member_name] = member_meta


def _coerce_embedding_b64(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    return str(value)


def _normalize_transcript(record: dict[str, Any], *, redact_patterns: list[str]) -> dict[str, Any]:
    normalized = deepcopy(record)
    normalized["transcript_text"] = _redact_text(str(record.get("transcript_text", "")), redact_patterns)
    normalized["embedding_b64"] = _coerce_embedding_b64(record.get("embedding_b64") or record.get("embedding"))
    normalized["tags"] = sorted(str(tag) for tag in normalized.get("tags", []) if str(tag).strip())
    normalized.pop("embedding", None)
    return normalized


def _normalize_node(record: dict[str, Any], *, redact_patterns: list[str], include_embeddings: bool) -> dict[str, Any]:
    normalized = deepcopy(record)
    normalized["source_prompt"] = _redact_text(str(record.get("source_prompt", "")), redact_patterns)
    normalized["tags"] = sorted(str(tag) for tag in normalized.get("tags", []) if str(tag).strip())
    normalized["evidence_records"] = _sorted_records(list(normalized.get("evidence_records", [])), "evidence_id", "session_id", "turn_index")
    embedding = normalized.pop("embedding", None)
    if include_embeddings:
        normalized["embedding_b64"] = _coerce_embedding_b64(normalized.get("embedding_b64") or embedding)
    else:
        normalized["embedding_b64"] = ""
    return normalized


def _normalize_edge(record: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(record)
    normalized["shared_entities"] = sorted(str(item) for item in normalized.get("shared_entities", []) if str(item).strip())
    return normalized


def _normalize_window(record: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(record)


def _assert_supported_schema_version(schema_version: str) -> None:
    major_text = str(schema_version).split(".", 1)[0].strip() or "0"
    try:
        major = int(major_text)
    except ValueError as exc:
        raise ValidationFailure(f"Invalid schema version '{schema_version}'.") from exc
    if major != ABHI_MAJOR_VERSION:
        raise ValidationFailure(
            f"Unsupported .abhi schema version '{schema_version}'. "
            f"Readers support major version {ABHI_MAJOR_VERSION} only."
        )


def filter_snapshot_by_scope(
    snapshot: dict[str, Any],
    *,
    project: str = "",
    agent_id: str = "",
    session_id: str = "",
) -> dict[str, Any]:
    if not any((project.strip(), agent_id.strip(), session_id.strip())):
        return deepcopy(snapshot)

    filtered = deepcopy(snapshot)
    filtered["nodes"] = [node for node in snapshot.get("nodes", []) if _scope_match(node, project=project, agent_id=agent_id, session_id=session_id)]
    selected_node_ids = {str(node.get("id", "")).strip() for node in filtered["nodes"]}
    filtered["edges"] = [
        edge
        for edge in snapshot.get("edges", [])
        if str(edge.get("source_id", "")).strip() in selected_node_ids
        and str(edge.get("target_id", "")).strip() in selected_node_ids
    ]
    filtered["transcripts"] = [
        transcript
        for transcript in snapshot.get("transcripts", [])
        if _scope_match(transcript, project=project, agent_id=agent_id, session_id=session_id)
    ]
    selected_window_ids = {
        str(node.get("context_window_id", "")).strip()
        for node in filtered["nodes"]
        if str(node.get("context_window_id", "")).strip()
    }
    filtered["context_windows"] = [
        window for window in snapshot.get("context_windows", []) if str(window.get("id", "")).strip() in selected_window_ids
    ]
    filtered["context_window_edges"] = [
        edge
        for edge in snapshot.get("context_window_edges", [])
        if str(edge.get("source_window_id", "")).strip() in selected_window_ids
        and str(edge.get("target_window_id", "")).strip() in selected_window_ids
    ]
    filtered["repos"] = deepcopy(snapshot.get("repos", []))
    return filtered


def _scope_filter(snapshot: dict[str, Any], *, scope: str, project: str, agent_id: str, session_id: str, since_date: str) -> dict[str, Any]:
    normalized = scope.strip().lower() or "all"
    filtered = deepcopy(snapshot)
    if normalized in {"project", "session"}:
        filtered = filter_snapshot_by_scope(
            filtered,
            project=project if normalized == "project" else project,
            agent_id=agent_id,
            session_id=session_id if normalized == "session" else "",
        )
    elif normalized == "since-date":
        cutoff = since_date.strip()
        if not cutoff:
            raise ValidationFailure("--scope since-date requires --since-date.")
        filtered["nodes"] = [node for node in filtered.get("nodes", []) if str(node.get("updated_at") or node.get("created_at") or "") >= cutoff]
        node_ids = {str(node.get("id", "")).strip() for node in filtered["nodes"]}
        filtered["edges"] = [
            edge
            for edge in filtered.get("edges", [])
            if str(edge.get("source_id", "")).strip() in node_ids and str(edge.get("target_id", "")).strip() in node_ids
        ]
        filtered["transcripts"] = [
            row for row in filtered.get("transcripts", []) if str(row.get("observed_at", "")).strip() >= cutoff
        ]
    elif normalized != "all":
        raise ValidationFailure("scope must be one of: all, project, session, since-date.")
    return filtered


def build_abhi_document(
    snapshot: dict[str, Any],
    *,
    scope: str = "all",
    project: str = "",
    agent_id: str = "",
    session_id: str = "",
    since_date: str = "",
    include_embeddings: bool = True,
    redact_patterns: list[str] | None = None,
    encrypted: bool = False,
) -> dict[str, Any]:
    redact_patterns = redact_patterns or []
    filtered = _scope_filter(
        snapshot,
        scope=scope,
        project=project,
        agent_id=agent_id,
        session_id=session_id,
        since_date=since_date,
    )
    transcripts = _sorted_records(
        [_normalize_transcript(item, redact_patterns=redact_patterns) for item in filtered.get("transcripts", [])],
        "id",
        "turn_pair_id",
        "observed_at",
    )
    nodes = _sorted_records(
        [
        _normalize_node(item, redact_patterns=redact_patterns, include_embeddings=include_embeddings)
        for item in filtered.get("nodes", [])
    ],
        "id",
        "source_turn_pair_id",
        "updated_at",
    )
    edges = _sorted_records([_normalize_edge(item) for item in filtered.get("edges", [])], "id", "source_id", "target_id", "relationship")
    context_windows = _sorted_records([_normalize_window(item) for item in filtered.get("context_windows", [])], "id", "session_id")
    repos = _sorted_records(list(filtered.get("repos", [])), "id", "name")
    context_window_edges = _sorted_records(list(filtered.get("context_window_edges", [])), "id", "source_window_id", "target_window_id", "edge_type")
    manifest = {
        "schema_version": ABHI_SPEC_VERSION,
        "tenant": str(filtered.get("tenant_id", "")),
        "agent_id": agent_id or str(filtered.get("agent_id", "")),
        "project": project or str(filtered.get("project", "")),
        "session_id": session_id or str(filtered.get("session_id", "")),
        "embedding_model_id": str(filtered.get("embedding_model_id") or _infer_embedding_model_id(nodes, transcripts)),
        "embedding_dim": int(filtered.get("embedding_dim") or _infer_embedding_dim(nodes, transcripts)),
        "encryption": {
            "enabled": encrypted,
            "algorithm": ABHI_ENCRYPTION_ALGORITHM if encrypted else "",
        },
        "signatures": {
            "algorithm": ABHI_SIGNATURE_ALGORITHM,
            "present": False,
        },
        "scope": scope,
        "includes_embeddings": include_embeddings,
        "export_context": {},
        "counts": {
            "transcripts": len(transcripts),
            "nodes": len(nodes),
            "edges": len(edges),
            "context_windows": len(context_windows),
        },
        "members": {},
        "ui": deepcopy(filtered.get("ui", {})),
        "repos": repos,
        "context_window_edges": context_window_edges,
    }
    document = {
        "manifest": manifest,
        "transcripts": transcripts,
        "nodes": nodes,
        "edges": edges,
        "context_windows": context_windows,
    }
    manifest["content_hash"] = _hash_with_prefix(compute_abhi_hash(document))
    return _with_compat_views(document)


def _infer_embedding_model_id(nodes: list[dict[str, Any]], transcripts: list[dict[str, Any]]) -> str:
    for record in [*nodes, *transcripts]:
        value = str(record.get("embedding_model_id", "")).strip()
        if value:
            return value
    return ""


def _infer_embedding_dim(nodes: list[dict[str, Any]], transcripts: list[dict[str, Any]]) -> int:
    for record in [*nodes, *transcripts]:
        value = int(record.get("embedding_dim", 0) or 0)
        if value:
            return value
    return 0


def _signature_payload(document: dict[str, Any]) -> bytes:
    content_hash = _hash_with_prefix(str(document.get("manifest", {}).get("content_hash", "")) or compute_abhi_hash(document))
    return content_hash.encode("utf-8")


def _load_or_create_signing_key(signing_key_dir: str | Path) -> tuple[ed25519.Ed25519PrivateKey, Path]:
    key_dir = Path(signing_key_dir).expanduser()
    key_dir.mkdir(parents=True, exist_ok=True)
    private_path = key_dir / "abhi-signing-key.pem"
    if private_path.exists():
        private_key = serialization.load_pem_private_key(private_path.read_bytes(), password=None)
        if not isinstance(private_key, ed25519.Ed25519PrivateKey):
            raise ValidationFailure(f"Unsupported signing key in {private_path}.")
        return private_key, private_path
    private_key = ed25519.Ed25519PrivateKey.generate()
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return private_key, private_path


def write_abhi_document(
    snapshot: dict[str, Any],
    *,
    output_path: str | Path,
    passphrase: str = "",
    scope: str = "all",
    project: str = "",
    agent_id: str = "",
    session_id: str = "",
    since_date: str = "",
    include_embeddings: bool = True,
    redact_patterns: list[str] | None = None,
    sign: bool = False,
    signing_key_dir: str | Path | None = None,
) -> AbhiExportResult:
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = build_abhi_document(
        snapshot,
        scope=scope,
        project=project,
        agent_id=agent_id,
        session_id=session_id,
        since_date=since_date,
        include_embeddings=include_embeddings,
        redact_patterns=redact_patterns,
        encrypted=bool(passphrase),
    )
    manifest = document["manifest"]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _write_member(archive, manifest, ABHI_TRANSCRIPTS_MEMBER, _record_lines(document["transcripts"]), passphrase=passphrase)
        _write_member(archive, manifest, ABHI_NODES_MEMBER, _record_lines(document["nodes"]), passphrase=passphrase)
        _write_member(archive, manifest, ABHI_EDGES_MEMBER, _record_lines(document["edges"]), passphrase=passphrase)
        _write_member(
            archive,
            manifest,
            ABHI_CONTEXT_WINDOWS_MEMBER,
            _record_lines(document["context_windows"]),
            passphrase=passphrase,
        )
        if sign:
            private_key, _ = _load_or_create_signing_key(signing_key_dir or "~/.waggle/keys")
            signature = private_key.sign(_signature_payload(document))
            public_key = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            _archive_writestr(archive, ABHI_SIGNATURE_MEMBER, signature)
            _archive_writestr(archive, ABHI_PUBLIC_KEY_MEMBER, public_key)
            manifest["signatures"] = {
                "algorithm": ABHI_SIGNATURE_ALGORITHM,
                "present": True,
            }
        manifest["content_hash"] = _hash_with_prefix(compute_abhi_hash(document))
        _archive_writestr(archive, ABHI_MANIFEST_MEMBER, _canonical_json(manifest))
    return AbhiExportResult(
        output_path=str(destination),
        tenant_id=str(manifest.get("tenant", "")),
        schema_version=ABHI_MAJOR_VERSION,
        abhi_spec_version=ABHI_SPEC_VERSION,
        node_count=len(document["nodes"]),
        edge_count=len(document["edges"]),
        content_hash=str(manifest.get("content_hash", "")),
        embedding_count=sum(1 for node in document["nodes"] if str(node.get("embedding_b64", "")).strip()),
        encrypted=bool(passphrase),
        encryption_algorithm=ABHI_ENCRYPTION_ALGORITHM if passphrase else "",
        executed_actions=[],
    )


def load_abhi_document(input_path: str | Path, passphrase: str = "") -> dict[str, Any]:
    source = Path(input_path).expanduser()
    with zipfile.ZipFile(source, "r") as archive:
        if ABHI_MANIFEST_MEMBER not in archive.namelist():
            raise ValidationFailure(f"{source} is missing {ABHI_MANIFEST_MEMBER}.")
        manifest = json.loads(archive.read(ABHI_MANIFEST_MEMBER).decode("utf-8"))
        _assert_supported_schema_version(str(manifest.get("schema_version", "")))
        document = {
            "manifest": manifest,
            "transcripts": _parse_lines(_read_member(archive, manifest, ABHI_TRANSCRIPTS_MEMBER, passphrase=passphrase)),
            "nodes": _parse_lines(_read_member(archive, manifest, ABHI_NODES_MEMBER, passphrase=passphrase)),
            "edges": _parse_lines(_read_member(archive, manifest, ABHI_EDGES_MEMBER, passphrase=passphrase)),
            "context_windows": _parse_lines(_read_member(archive, manifest, ABHI_CONTEXT_WINDOWS_MEMBER, passphrase=passphrase)),
        }
        if manifest.get("signatures", {}).get("present"):
            document["signature"] = archive.read(ABHI_SIGNATURE_MEMBER)
            document["public_key_pem"] = archive.read(ABHI_PUBLIC_KEY_MEMBER)
        return _with_compat_views(document)


def inspect_abhi_document(document: dict[str, Any], *, input_path: str | Path) -> AbhiInspectResult:
    manifest = document.get("manifest", {})
    node_types = sorted({str(node.get("node_type", "")).strip() for node in document.get("nodes", []) if str(node.get("node_type", "")).strip()})
    edge_types = sorted({str(edge.get("relationship", "")).strip() for edge in document.get("edges", []) if str(edge.get("relationship", "")).strip()})
    return AbhiInspectResult(
        input_path=str(Path(input_path).expanduser()),
        tenant_id=str(manifest.get("tenant", "")),
        schema_version=ABHI_MAJOR_VERSION,
        abhi_spec_version=str(manifest.get("schema_version", "")) or ABHI_SPEC_VERSION,
        node_count=len(document.get("nodes", [])),
        edge_count=len(document.get("edges", [])),
        node_types=node_types,
        edge_types=edge_types,
        constraint_count=0,
        version_count=1,
        query_count=0,
        event_count=0,
        chunk_count=max(1, (len(document.get("nodes", [])) + ABHI_CHUNK_NODE_LIMIT - 1) // ABHI_CHUNK_NODE_LIMIT),
        load_strategy="chunked" if len(document.get("nodes", [])) > ABHI_CHUNK_NODE_LIMIT else "full",
        preload_chunks=["chunk-0"] if document.get("nodes") else [],
        content_hash=str(manifest.get("content_hash", "")),
        embedding_count=sum(1 for node in document.get("nodes", []) if str(node.get("embedding_b64", "")).strip()),
        encrypted=bool(manifest.get("encryption", {}).get("enabled")),
        encryption_algorithm=str(manifest.get("encryption", {}).get("algorithm", "")),
    )


def _diff_dicts(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _canonical_json(left) != _canonical_json(right)


def diff_abhi_documents(
    document_a: dict[str, Any],
    document_b: dict[str, Any],
    *,
    input_path_a: str | Path,
    input_path_b: str | Path,
) -> AbhiDiffResult:
    nodes_a = {str(node.get("id", "")): node for node in document_a.get("nodes", [])}
    nodes_b = {str(node.get("id", "")): node for node in document_b.get("nodes", [])}
    edges_a = {str(edge.get("id", "")): edge for edge in document_a.get("edges", [])}
    edges_b = {str(edge.get("id", "")): edge for edge in document_b.get("edges", [])}
    nodes_added = sorted(node_id for node_id in nodes_b if node_id and node_id not in nodes_a)
    nodes_removed = sorted(node_id for node_id in nodes_a if node_id and node_id not in nodes_b)
    nodes_updated = sorted(node_id for node_id in nodes_a.keys() & nodes_b.keys() if _diff_dicts(nodes_a[node_id], nodes_b[node_id]))
    edges_added = sorted(edge_id for edge_id in edges_b if edge_id and edge_id not in edges_a)
    edges_removed = sorted(edge_id for edge_id in edges_a if edge_id and edge_id not in edges_b)
    edges_updated = sorted(edge_id for edge_id in edges_a.keys() & edges_b.keys() if _diff_dicts(edges_a[edge_id], edges_b[edge_id]))
    semantic_changes = [f"updated node {node_id}" for node_id in nodes_updated] + [f"updated edge {edge_id}" for edge_id in edges_updated]
    return AbhiDiffResult(
        input_path_a=str(Path(input_path_a).expanduser()),
        input_path_b=str(Path(input_path_b).expanduser()),
        abhi_spec_version_a=str(document_a.get("manifest", {}).get("schema_version", ABHI_SPEC_VERSION)),
        abhi_spec_version_b=str(document_b.get("manifest", {}).get("schema_version", ABHI_SPEC_VERSION)),
        nodes_added=nodes_added,
        nodes_removed=nodes_removed,
        nodes_updated=nodes_updated,
        edges_added=edges_added,
        edges_removed=edges_removed,
        edges_updated=edges_updated,
        semantic_changes=semantic_changes,
    )


def _merge_records(
    base_items: list[dict[str, Any]],
    left_items: list[dict[str, Any]],
    right_items: list[dict[str, Any]],
    *,
    merge_strategy: str,
    conflicts: list[str],
) -> list[dict[str, Any]]:
    base_map = {str(item.get("id", "")): item for item in base_items if str(item.get("id", "")).strip()}
    left_map = {str(item.get("id", "")): item for item in left_items if str(item.get("id", "")).strip()}
    right_map = {str(item.get("id", "")): item for item in right_items if str(item.get("id", "")).strip()}
    merged: dict[str, dict[str, Any]] = {}
    for item_id in sorted(set(base_map) | set(left_map) | set(right_map)):
        left_item = left_map.get(item_id)
        right_item = right_map.get(item_id)
        if left_item is None and right_item is None:
            continue
        if left_item is None:
            merged[item_id] = deepcopy(right_item)
            continue
        if right_item is None:
            merged[item_id] = deepcopy(left_item)
            continue
        if not _diff_dicts(left_item, right_item):
            merged[item_id] = deepcopy(left_item)
            continue
        if merge_strategy == "prefer_left":
            merged[item_id] = deepcopy(left_item)
        elif merge_strategy == "prefer_right":
            merged[item_id] = deepcopy(right_item)
        else:
            left_ts = str(left_item.get("updated_at") or left_item.get("created_at") or "")
            right_ts = str(right_item.get("updated_at") or right_item.get("created_at") or "")
            merged[item_id] = deepcopy(right_item if right_ts >= left_ts else left_item)
        if _diff_dicts(base_map.get(item_id, {}), left_item) and _diff_dicts(base_map.get(item_id, {}), right_item):
            conflicts.append(f"Conflict on {item_id}")
    return list(merged.values())


def merge_abhi_documents(
    base_document: dict[str, Any],
    left_document: dict[str, Any],
    right_document: dict[str, Any],
    *,
    base_input_path: str | Path,
    left_input_path: str | Path,
    right_input_path: str | Path,
    output_path: str | Path,
    merge_strategy: str = "prefer_right",
    passphrase: str = "",
) -> AbhiMergeResult:
    conflicts: list[str] = []
    merged_snapshot = {
        "tenant_id": str(right_document.get("manifest", {}).get("tenant") or left_document.get("manifest", {}).get("tenant", "")),
        "transcripts": _merge_records(
            base_document.get("transcripts", []),
            left_document.get("transcripts", []),
            right_document.get("transcripts", []),
            merge_strategy=merge_strategy,
            conflicts=conflicts,
        ),
        "nodes": _merge_records(
            base_document.get("nodes", []),
            left_document.get("nodes", []),
            right_document.get("nodes", []),
            merge_strategy=merge_strategy,
            conflicts=conflicts,
        ),
        "edges": _merge_records(
            base_document.get("edges", []),
            left_document.get("edges", []),
            right_document.get("edges", []),
            merge_strategy=merge_strategy,
            conflicts=conflicts,
        ),
        "context_windows": _merge_records(
            base_document.get("context_windows", []),
            left_document.get("context_windows", []),
            right_document.get("context_windows", []),
            merge_strategy=merge_strategy,
            conflicts=conflicts,
        ),
        "ui": deepcopy(right_document.get("manifest", {}).get("ui", {}) or left_document.get("manifest", {}).get("ui", {})),
        "repos": deepcopy(right_document.get("manifest", {}).get("repos", []) or left_document.get("manifest", {}).get("repos", [])),
        "context_window_edges": deepcopy(
            right_document.get("manifest", {}).get("context_window_edges", [])
            or left_document.get("manifest", {}).get("context_window_edges", [])
        ),
        "embedding_model_id": str(
            right_document.get("manifest", {}).get("embedding_model_id")
            or left_document.get("manifest", {}).get("embedding_model_id", "")
        ),
        "embedding_dim": int(
            right_document.get("manifest", {}).get("embedding_dim")
            or left_document.get("manifest", {}).get("embedding_dim", 0)
            or 0
        ),
    }
    exported = write_abhi_document(merged_snapshot, output_path=output_path, passphrase=passphrase)
    return AbhiMergeResult(
        base_input_path=str(base_input_path),
        left_input_path=str(left_input_path),
        right_input_path=str(right_input_path),
        output_path=exported.output_path,
        merge_strategy=merge_strategy,
        abhi_spec_version=ABHI_SPEC_VERSION,
        nodes_merged=len(merged_snapshot["nodes"]),
        edges_merged=len(merged_snapshot["edges"]),
        conflicts=conflicts,
        content_hash=exported.content_hash,
        embedding_count=exported.embedding_count,
        encrypted=exported.encrypted,
        encryption_algorithm=exported.encryption_algorithm,
        executed_actions=[],
    )


def _matches_query(node: dict[str, Any], query_text: str) -> bool:
    lowered = query_text.lower()
    if lowered.startswith("find nodes where"):
        text = lowered
        if "type='" in text:
            expected_type = text.split("type='", 1)[1].split("'", 1)[0]
            if str(node.get("node_type", "")).lower() != expected_type:
                return False
        if "content contains '" in text:
            expected = text.split("content contains '", 1)[1].split("'", 1)[0]
            return expected in str(node.get("content", "")).lower()
        return True
    haystack = " ".join(
        [
            str(node.get("label", "")),
            str(node.get("content", "")),
            " ".join(str(tag) for tag in node.get("tags", [])),
        ]
    ).lower()
    return lowered in haystack


def execute_abhi_query(document: dict[str, Any], *, query_id: str = "", query_text: str = "") -> dict[str, Any]:
    effective_query = query_text.strip() or query_id.strip()
    if not effective_query:
        raise ValidationFailure("Provide query_text or query_id.")
    if query_text.strip():
        matched_nodes = [node for node in document.get("nodes", []) if _matches_query(node, effective_query)]
    else:
        matched_nodes = list(document.get("nodes", []))[:ABHI_CHUNK_NODE_LIMIT]
    node_ids = {str(node.get("id", "")).strip() for node in matched_nodes}
    matched_edges = [
        edge
        for edge in document.get("edges", [])
        if str(edge.get("source_id", "")).strip() in node_ids and str(edge.get("target_id", "")).strip() in node_ids
    ]
    chunk_ids = sorted({f"chunk-{index // ABHI_CHUNK_NODE_LIMIT}" for index, node in enumerate(document.get("nodes", [])) if str(node.get("id", "")).strip() in node_ids})
    compatible_nodes = []
    for node in matched_nodes:
        enriched = deepcopy(node)
        enriched["type"] = enriched.get("node_type", "")
        compatible_nodes.append(enriched)
    return {
        "query_id": query_id,
        "query": effective_query,
        "nodes": compatible_nodes,
        "edges": matched_edges,
        "chunk_ids": chunk_ids,
    }


def load_abhi_chunks(
    document: dict[str, Any],
    *,
    chunk_ids: list[str] | None = None,
    query_text: str = "",
) -> dict[str, Any]:
    nodes = list(document.get("nodes", []))
    selected_chunk_ids = [chunk_id for chunk_id in (chunk_ids or []) if str(chunk_id).strip()]
    if not selected_chunk_ids and query_text.strip():
        selected_chunk_ids = execute_abhi_query(document, query_text=query_text)["chunk_ids"]
    if not selected_chunk_ids:
        selected_chunk_ids = ["chunk-0"] if nodes else []
    selected_indexes: set[int] = set()
    for chunk_id in selected_chunk_ids:
        if not chunk_id.startswith("chunk-"):
            continue
        try:
            chunk_index = int(chunk_id.split("-", 1)[1])
        except ValueError:
            continue
        start = chunk_index * ABHI_CHUNK_NODE_LIMIT
        end = start + ABHI_CHUNK_NODE_LIMIT
        selected_indexes.update(range(start, min(end, len(nodes))))
    chunk_nodes = [nodes[index] for index in sorted(selected_indexes) if 0 <= index < len(nodes)]
    node_ids = {str(node.get("id", "")).strip() for node in chunk_nodes}
    chunk_edges = [
        edge
        for edge in document.get("edges", [])
        if str(edge.get("source_id", "")).strip() in node_ids and str(edge.get("target_id", "")).strip() in node_ids
    ]
    return {
        "chunk_ids": selected_chunk_ids,
        "nodes": chunk_nodes,
        "edges": chunk_edges,
        "available_chunk_count": max(1, (len(nodes) + ABHI_CHUNK_NODE_LIMIT - 1) // ABHI_CHUNK_NODE_LIMIT) if nodes else 0,
        "load_strategy": "chunked" if len(nodes) > ABHI_CHUNK_NODE_LIMIT else "full",
        "query": query_text,
    }


def query_abhi_file(input_path: str | Path, *, query_id: str = "", query_text: str = "", passphrase: str = "") -> AbhiQueryResult:
    source = Path(input_path).expanduser()
    document = load_abhi_document(source, passphrase=passphrase)
    payload = execute_abhi_query(document, query_id=query_id, query_text=query_text)
    return AbhiQueryResult(
        input_path=str(source),
        query_id=query_id,
        name=query_id,
        query=payload["query"],
        summary=f"{len(payload['nodes'])} nodes matched.",
        node_count=len(payload["nodes"]),
        edge_count=len(payload["edges"]),
        node_ids=[str(node.get("id", "")) for node in payload["nodes"]],
        edge_ids=[str(edge.get("id", "")) for edge in payload["edges"]],
        chunk_ids=payload["chunk_ids"],
        scanned_chunk_count=max(1, (len(document.get("nodes", [])) + ABHI_CHUNK_NODE_LIMIT - 1) // ABHI_CHUNK_NODE_LIMIT) if document.get("nodes") else 0,
        executed_actions=dispatch_abhi_event(document, event_name="on_query", persist=False, input_path=source, query_payload=payload),
    )


def load_abhi_chunk_file(
    input_path: str | Path,
    *,
    chunk_ids: list[str] | None = None,
    query_id: str = "",
    query_text: str = "",
    passphrase: str = "",
) -> AbhiChunkLoadResult:
    source = Path(input_path).expanduser()
    document = load_abhi_document(source, passphrase=passphrase)
    selection_query = query_text.strip() or query_id.strip()
    payload = load_abhi_chunks(document, chunk_ids=chunk_ids or [], query_text=selection_query)
    return AbhiChunkLoadResult(
        input_path=str(source),
        chunk_ids=payload["chunk_ids"],
        load_strategy=payload["load_strategy"],
        node_count=len(payload["nodes"]),
        edge_count=len(payload["edges"]),
        available_chunk_count=payload["available_chunk_count"],
        query=payload["query"],
        node_ids=[str(node.get("id", "")) for node in payload["nodes"]],
        edge_ids=[str(edge.get("id", "")) for edge in payload["edges"]],
    )


def dispatch_abhi_event(
    document: dict[str, Any],
    *,
    event_name: str,
    persist: bool,
    input_path: str | Path | None = None,
    query_payload: dict[str, Any] | None = None,
) -> list[str]:
    action_map = {
        "on_export": ["exported_abhi"],
        "on_import": ["imported_abhi"],
        "on_query": ["queried_abhi"],
        "on_merge": ["merged_abhi"],
    }
    return action_map.get(event_name, [])


def validate_abhi_signature(document: dict[str, Any]) -> None:
    if not document.get("manifest", {}).get("signatures", {}).get("present"):
        raise ValidationFailure("This .abhi file is not signed.")
    public_key = serialization.load_pem_public_key(document["public_key_pem"])
    if not isinstance(public_key, ed25519.Ed25519PublicKey):
        raise ValidationFailure("Unsupported ABHI public key.")
    try:
        public_key.verify(document["signature"], _signature_payload(document))
    except InvalidSignature as exc:
        raise ValidationFailure("ABHI signature verification failed.") from exc


def validate_abhi_document(document: dict[str, Any], *, input_path: str | Path) -> AbhiValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    manifest = document.get("manifest", {})
    try:
        _assert_supported_schema_version(str(manifest.get("schema_version", "")))
    except ValidationFailure as exc:
        errors.append(str(exc))
    actual_hash = compute_abhi_hash(document)
    expected_hash = str(manifest.get("content_hash", ""))
    if expected_hash and expected_hash.removeprefix("sha256:") != actual_hash.removeprefix("sha256:"):
        errors.append("Manifest content hash does not match payload.")
    if manifest.get("signatures", {}).get("present"):
        try:
            validate_abhi_signature(document)
        except ValidationFailure as exc:
            errors.append(str(exc))
    if not document.get("nodes") and not document.get("transcripts"):
        warnings.append("ABHI payload is empty.")
    return AbhiValidationResult(
        input_path=str(Path(input_path).expanduser()),
        valid=not errors,
        errors=errors,
        warnings=warnings,
        node_count=len(document.get("nodes", [])),
        edge_count=len(document.get("edges", [])),
        content_hash=_hash_with_prefix(actual_hash.removeprefix("sha256:")),
        abhi_spec_version=str(manifest.get("schema_version", "")) or ABHI_SPEC_VERSION,
        embedding_count=sum(1 for node in document.get("nodes", []) if str(node.get("embedding_b64", "")).strip()),
        encrypted=bool(manifest.get("encryption", {}).get("enabled")),
        encryption_algorithm=str(manifest.get("encryption", {}).get("algorithm", "")),
    )


def _decode_embedding(embedding_b64: str) -> bytes | None:
    if not embedding_b64.strip():
        return None
    return base64.b64decode(embedding_b64.encode("ascii"))


def _namespace_id(namespace: str, value: str) -> str:
    return f"{namespace}:{value}" if namespace.strip() else value


def abhi_to_snapshot(
    document: dict[str, Any],
    *,
    fallback_tenant_id: str,
    namespace: str = "",
    read_only: bool = False,
    reembed_on_import: bool = False,
) -> dict[str, Any]:
    manifest = document.get("manifest", {})
    node_id_map = {
        str(node.get("id", "")): _namespace_id(namespace, str(node.get("id", "")))
        for node in document.get("nodes", [])
        if str(node.get("id", "")).strip()
    }
    nodes: list[dict[str, Any]] = []
    for raw_node in document.get("nodes", []):
        metadata = deepcopy(raw_node.get("metadata", {}))
        if read_only:
            metadata["abhi_read_only"] = True
        nodes.append(
            {
                **deepcopy(raw_node),
                "id": node_id_map.get(str(raw_node.get("id", "")), str(raw_node.get("id", ""))),
                "tenant_id": str(manifest.get("tenant", "")) or fallback_tenant_id,
                "embedding": None if reembed_on_import else _decode_embedding(str(raw_node.get("embedding_b64", ""))),
                "metadata": metadata,
            }
        )
    edges: list[dict[str, Any]] = []
    for raw_edge in document.get("edges", []):
        edges.append(
            {
                **deepcopy(raw_edge),
                "id": _namespace_id(namespace, str(raw_edge.get("id", ""))),
                "tenant_id": str(manifest.get("tenant", "")) or fallback_tenant_id,
                "source_id": node_id_map.get(str(raw_edge.get("source_id", "")), str(raw_edge.get("source_id", ""))),
                "target_id": node_id_map.get(str(raw_edge.get("target_id", "")), str(raw_edge.get("target_id", ""))),
            }
        )
    transcripts: list[dict[str, Any]] = []
    for row in document.get("transcripts", []):
        transcripts.append(
            {
                **deepcopy(row),
                "id": _namespace_id(namespace, str(row.get("id", ""))),
                "tenant_id": str(manifest.get("tenant", "")) or fallback_tenant_id,
                "embedding": None if reembed_on_import else _decode_embedding(str(row.get("embedding_b64", ""))),
            }
        )
    windows = []
    for raw_window in document.get("context_windows", []):
        windows.append(
            {
                **deepcopy(raw_window),
                "id": _namespace_id(namespace, str(raw_window.get("id", ""))),
                "tenant_id": str(manifest.get("tenant", "")) or fallback_tenant_id,
            }
        )
    return {
        "schema_version": ABHI_MAJOR_VERSION,
        "tenant_id": str(manifest.get("tenant", "")) or fallback_tenant_id,
        "embedding_model_id": str(manifest.get("embedding_model_id", "")),
        "embedding_dim": int(manifest.get("embedding_dim", 0) or 0),
        "nodes": nodes,
        "edges": edges,
        "transcripts": transcripts,
        "context_windows": windows,
        "context_window_edges": deepcopy(manifest.get("context_window_edges", [])),
        "repos": deepcopy(manifest.get("repos", [])),
        "ui": deepcopy(manifest.get("ui", {})),
    }


def compute_abhi_hash(document: dict[str, Any]) -> str:
    manifest = deepcopy(document.get("manifest", {}))
    manifest.pop("content_hash", None)
    manifest.pop("created_at", None)
    manifest.pop("export_context", None)
    manifest["signatures"] = {
        "algorithm": ABHI_SIGNATURE_ALGORITHM,
        "present": False,
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json(manifest))
    digest.update(_record_lines(document.get("transcripts", [])))
    digest.update(_record_lines(document.get("nodes", [])))
    digest.update(_record_lines(document.get("edges", [])))
    digest.update(_record_lines(document.get("context_windows", [])))
    return digest.hexdigest()


def is_encrypted_abhi_payload(document: dict[str, Any]) -> bool:
    return bool(document.get("manifest", {}).get("encryption", {}).get("enabled"))


def encrypt_abhi_document(document: dict[str, Any], *, passphrase: str) -> dict[str, Any]:
    encrypted = deepcopy(document)
    encrypted.setdefault("manifest", {}).setdefault("encryption", {})
    encrypted["manifest"]["encryption"] = {
        "enabled": True,
        "algorithm": ABHI_ENCRYPTION_ALGORITHM,
    }
    return encrypted


def decrypt_abhi_document(document: dict[str, Any], *, passphrase: str) -> dict[str, Any]:
    return deepcopy(document)


def diff_abhi_files(*, input_path_a: str | Path, input_path_b: str | Path, passphrase: str = "") -> AbhiDiffResult:
    document_a = load_abhi_document(input_path_a, passphrase=passphrase)
    document_b = load_abhi_document(input_path_b, passphrase=passphrase)
    return diff_abhi_documents(document_a, document_b, input_path_a=input_path_a, input_path_b=input_path_b)


def merge_abhi_files(
    *,
    base_input_path: str | Path,
    left_input_path: str | Path,
    right_input_path: str | Path,
    output_path: str | Path,
    merge_strategy: str = "prefer_right",
    passphrase: str = "",
) -> AbhiMergeResult:
    return merge_abhi_documents(
        load_abhi_document(base_input_path, passphrase=passphrase),
        load_abhi_document(left_input_path, passphrase=passphrase),
        load_abhi_document(right_input_path, passphrase=passphrase),
        base_input_path=base_input_path,
        left_input_path=left_input_path,
        right_input_path=right_input_path,
        output_path=output_path,
        merge_strategy=merge_strategy,
        passphrase=passphrase,
    )


def _with_compat_views(document: dict[str, Any]) -> dict[str, Any]:
    compatible = deepcopy(document)
    manifest = compatible.get("manifest", {})
    nodes = []
    for node in compatible.get("nodes", []):
        enriched = deepcopy(node)
        enriched["type"] = enriched.get("node_type", "")
        metadata = deepcopy(enriched.get("metadata", {}))
        if str(enriched.get("agent_id", "")).strip():
            metadata.setdefault("source_app", enriched["agent_id"])
        enriched["metadata"] = metadata
        nodes.append(enriched)
    edges = []
    for edge in compatible.get("edges", []):
        enriched = deepcopy(edge)
        enriched["type"] = enriched.get("relationship", "")
        edges.append(enriched)
    compatible["graph"] = {"nodes": nodes, "edges": edges}
    compatible["integrity"] = {
        "content_hash": str(manifest.get("content_hash", "")),
        "schema_version": str(manifest.get("schema_version", ABHI_SPEC_VERSION)),
        "abhi_spec_version": str(manifest.get("schema_version", ABHI_SPEC_VERSION)),
    }
    compatible["schema"] = {"manifest": deepcopy(manifest)}
    compatible["waggle"] = {
        "tenant_id": str(manifest.get("tenant", "")),
        "schema_version": ABHI_MAJOR_VERSION,
        "context_windows": deepcopy(compatible.get("context_windows", [])),
        "event_log": [],
    }
    compatible["embeddings"] = {
        "vectors": {
            str(node.get("id", "")): str(node.get("embedding_b64", ""))
            for node in compatible.get("nodes", [])
            if str(node.get("embedding_b64", "")).strip()
        }
    }
    chunk_index: dict[str, dict[str, Any]] = {}
    chunk_payloads: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(nodes), ABHI_CHUNK_NODE_LIMIT):
        chunk_id = f"chunk-{offset // ABHI_CHUNK_NODE_LIMIT}"
        chunk_nodes = nodes[offset : offset + ABHI_CHUNK_NODE_LIMIT]
        chunk_index[chunk_id] = {"byte_length": len(_record_lines(chunk_nodes))}
        chunk_payloads[chunk_id] = {"node_ids": [str(node.get("id", "")) for node in chunk_nodes]}
    compatible["chunks"] = {
        "chunk_index": chunk_index,
        "chunk_payloads": chunk_payloads,
        "load_strategy": "on_demand" if len(nodes) > ABHI_CHUNK_NODE_LIMIT else "full",
    }
    compatible["ui"] = deepcopy(manifest.get("ui", {}))
    compatible["queries"] = {}
    compatible["events"] = {}
    compatible["versions"] = []
    return compatible
