#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[3] / "rust_copilot_metrics_fixture"
OUTPUT = Path(__file__).resolve().parents[1] / "fixtures" / "rust_copilot_metrics_fixture.rust_analyzer_baseline.json"

class LspClient:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            ["rust-analyzer"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.next_id = 1

    def _send(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload)
        data = f"Content-Length: {len(body)}\r\n\r\n{body}".encode("utf-8")
        assert self.proc.stdin is not None
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def _read_message(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        headers: dict[str, str] = {}
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("rust-analyzer closed stdout")
            text = line.decode("utf-8").rstrip("\r\n")
            if text == "":
                break
            k, v = text.split(":", 1)
            headers[k.lower().strip()] = v.strip()

        content_length = int(headers.get("content-length", "0"))
        payload = self.proc.stdout.read(content_length)
        return json.loads(payload.decode("utf-8"))

    def request(self, method: str, params: dict[str, Any]) -> Any:
        req_id = self.next_id
        self.next_id += 1
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})

        while True:
            message = self._read_message()
            if message.get("id") != req_id:
                continue
            if "error" in message:
                raise RuntimeError(json.dumps(message["error"]))
            return message.get("result")

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})


def path_to_uri(path: Path) -> str:
    raw = str(path)
    if not raw.startswith("/"):
        raw = "/" + raw
    encoded = (
        raw.replace("%", "%25")
        .replace(" ", "%20")
        .replace("#", "%23")
        .replace("?", "%3F")
    )
    return f"file://{encoded}"


def lsp_locations(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and payload.get("uri"):
        items = [payload]
    else:
        items = []

    out = []
    for item in items:
        uri = item.get("uri")
        range_obj = item.get("range") or {}
        start = (range_obj.get("start") or {}).get("line")
        end = (range_obj.get("end") or {}).get("line")
        if uri is None or start is None or end is None:
            continue
        out.append(
            {
                "uri": uri,
                "start_line": int(start) + 1,
                "end_line": max(int(start) + 1, int(end) + 1),
            }
        )
    return out


def collect_document_symbols(items: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def walk(item: dict[str, Any]) -> None:
        name = item.get("name")
        kind = item.get("kind")
        range_obj = item.get("range") or {}
        start = (range_obj.get("start") or {}).get("line")
        end = (range_obj.get("end") or {}).get("line")
        if name is not None and kind is not None and start is not None and end is not None:
            out.append(
                {
                    "name": name,
                    "kind": int(kind),
                    "start_line": int(start) + 1,
                    "end_line": max(int(start) + 1, int(end) + 1),
                }
            )
        for child in item.get("children") or []:
            walk(child)

    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                walk(item)

    return out


def main() -> None:
    files = sorted(
        p
        for p in WORKSPACE.rglob("*.rs")
        if ".git" not in p.parts and "target" not in p.parts
    )

    client = LspClient()
    client.request(
        "initialize",
        {
            "processId": None,
            "rootUri": path_to_uri(WORKSPACE),
            "capabilities": {
                "textDocument": {
                    "documentSymbol": {
                        "dynamicRegistration": False,
                        "hierarchicalDocumentSymbolSupport": True,
                        "labelSupport": True,
                    }
                }
            },
        },
    )
    client.notify("initialized", {})

    raw = {
        "workspace": str(WORKSPACE),
        "files": [str(p.relative_to(WORKSPACE)) for p in files],
        "document_symbols_total": 0,
        "eligible_symbols_total": 0,
        "hover_requests_total": 0,
        "hover_failures_total": 0,
        "references_requests_total": 0,
        "references_failures_total": 0,
        "references_nonempty_total": 0,
        "references_locations_total": 0,
        "implementations_requests_total": 0,
        "implementations_failures_total": 0,
        "implementations_nonempty_total": 0,
        "implementations_locations_total": 0,
        "definitions_requests_total": 0,
        "definitions_failures_total": 0,
        "definitions_nonempty_total": 0,
        "definitions_locations_total": 0,
        "type_definitions_requests_total": 0,
        "type_definitions_failures_total": 0,
        "type_definitions_nonempty_total": 0,
        "type_definitions_locations_total": 0,
    }

    per_file: dict[str, Any] = {}
    for path in files:
        rel = str(path.relative_to(WORKSPACE))
        uri = path_to_uri(path)
        content = path.read_text()
        client.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": "rust",
                    "version": 1,
                    "text": content,
                }
            },
        )

        file_data = {"document_symbols": 0, "eligible_symbols": 0}
        try:
            doc_symbols_payload = client.request("textDocument/documentSymbol", {"textDocument": {"uri": uri}})
        except RuntimeError:
            per_file[rel] = file_data
            continue

        symbols = collect_document_symbols(doc_symbols_payload)
        file_data["document_symbols"] = len(symbols)
        raw["document_symbols_total"] += len(symbols)

        eligible = [s for s in symbols if s["kind"] in {5, 10, 11, 12, 23}]
        file_data["eligible_symbols"] = len(eligible)
        raw["eligible_symbols_total"] += len(eligible)

        for symbol in symbols:
            raw["hover_requests_total"] += 1
            try:
                client.request(
                    "textDocument/hover",
                    {
                        "textDocument": {"uri": uri},
                        "position": {"line": max(symbol["start_line"] - 1, 0), "character": 0},
                    },
                )
            except RuntimeError:
                raw["hover_failures_total"] += 1

        for symbol in eligible:
            position = {"line": max(symbol["start_line"] - 1, 0), "character": 0}

            raw["references_requests_total"] += 1
            try:
                refs = client.request(
                    "textDocument/references",
                    {
                        "textDocument": {"uri": uri},
                        "position": position,
                        "context": {"includeDeclaration": False},
                    },
                )
                ref_locs = lsp_locations(refs)
                raw["references_locations_total"] += len(ref_locs)
                if ref_locs:
                    raw["references_nonempty_total"] += 1
            except RuntimeError:
                raw["references_failures_total"] += 1

            raw["implementations_requests_total"] += 1
            try:
                imps = client.request(
                    "textDocument/implementation",
                    {"textDocument": {"uri": uri}, "position": position},
                )
                imp_locs = lsp_locations(imps)
                raw["implementations_locations_total"] += len(imp_locs)
                if imp_locs:
                    raw["implementations_nonempty_total"] += 1
            except RuntimeError:
                raw["implementations_failures_total"] += 1

            raw["definitions_requests_total"] += 1
            try:
                defs = client.request(
                    "textDocument/definition",
                    {"textDocument": {"uri": uri}, "position": position},
                )
                def_locs = lsp_locations(defs)
                raw["definitions_locations_total"] += len(def_locs)
                if def_locs:
                    raw["definitions_nonempty_total"] += 1
            except RuntimeError:
                raw["definitions_failures_total"] += 1

            raw["type_definitions_requests_total"] += 1
            try:
                type_defs = client.request(
                    "textDocument/typeDefinition",
                    {"textDocument": {"uri": uri}, "position": position},
                )
                type_def_locs = lsp_locations(type_defs)
                raw["type_definitions_locations_total"] += len(type_def_locs)
                if type_def_locs:
                    raw["type_definitions_nonempty_total"] += 1
            except RuntimeError:
                raw["type_definitions_failures_total"] += 1
        per_file[rel] = file_data

    output = {
        "source": "rust-analyzer direct LSP",
        "workspace": raw["workspace"],
        "files": raw["files"],
        "rust_analyzer_metrics": raw,
        "per_file": per_file,
        "comparison_note": "Use this as upstream baseline. Daemon extraction metrics should be interpreted as post-processing of this LSP signal.",
    }

    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
