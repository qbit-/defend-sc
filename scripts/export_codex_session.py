#!/usr/bin/env python3
"""Export a local Codex session rollout to Markdown."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sqlite3
import sys
from pathlib import Path


CODEX_HOME = Path.home() / ".codex"
STATE_DB = CODEX_HOME / "state_5.sqlite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a local Codex session to Markdown."
    )
    parser.add_argument(
        "session_id",
        nargs="?",
        help="Codex session/thread id to export. If omitted, list all session ids.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output Markdown file. Defaults to codex-session-<session_id>.md",
    )
    return parser.parse_args()


def find_thread(session_id: str) -> sqlite3.Row:
    if not STATE_DB.exists():
        raise FileNotFoundError(f"Codex state database not found: {STATE_DB}")

    con = sqlite3.connect(STATE_DB)
    con.row_factory = sqlite3.Row
    try:
        thread = con.execute(
            """
            select id, title, cwd, rollout_path, tokens_used
            from threads
            where id = ?
            """,
            (session_id,),
        ).fetchone()
    finally:
        con.close()

    if thread is None:
        raise LookupError(f"Codex session not found: {session_id}")

    return thread


def list_sessions() -> list[sqlite3.Row]:
    if not STATE_DB.exists():
        raise FileNotFoundError(f"Codex state database not found: {STATE_DB}")

    con = sqlite3.connect(STATE_DB)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            select id, updated_at, first_user_message
            from threads
            order by updated_at desc, created_at desc
            """
        ).fetchall()
    finally:
        con.close()

    return rows


def truncate_to_terminal_width(line: str) -> str:
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    if width <= 0 or len(line) <= width:
        return line
    if width <= 3:
        return line[:width]
    return line[: width - 3] + "..."


def format_updated_at(timestamp: int) -> str:
    return dt.datetime.fromtimestamp(timestamp, tz=dt.UTC).strftime("%d %b")


def format_session_header() -> str:
    return truncate_to_terminal_width(f"{'SESSION ID':36}  {'DATE':6}  FIRST USER MESSAGE")


def format_session_row(row: sqlite3.Row) -> str:
    message = " ".join((row["first_user_message"] or "").split())
    date = format_updated_at(row["updated_at"])
    return truncate_to_terminal_width(f"{row['id']}  {date:6}  {message}")


def content_text(payload: dict) -> str:
    parts = []
    for item in payload.get("content") or []:
        if item.get("type") in {"input_text", "output_text"}:
            text = item.get("text", "")
            if text:
                parts.append(text)
    return "\n\n".join(parts).strip()


def is_dialogue_message(payload: dict) -> bool:
    return payload.get("type") == "message" and payload.get("role") in {
        "user",
        "assistant",
    }


def should_skip_text(text: str) -> bool:
    return not text or text.startswith("<environment_context>")


def export_markdown(thread: sqlite3.Row, output_path: Path) -> int:
    rollout_path = Path(thread["rollout_path"])
    if not rollout_path.exists():
        raise FileNotFoundError(f"Codex rollout file not found: {rollout_path}")

    session_id = thread["id"]
    lines = [
        f"# Codex Session {session_id}",
        "",
        f"- Title: {thread['title']}",
        f"- Working directory: `{thread['cwd']}`",
        f"- Rollout: `{rollout_path}`",
        f"- Tokens used: {thread['tokens_used']}",
        "",
    ]

    exported_messages = 0
    with rollout_path.open(encoding="utf-8") as rollout:
        for raw in rollout:
            obj = json.loads(raw)
            if obj.get("type") != "response_item":
                continue

            payload = obj.get("payload", {})
            if not is_dialogue_message(payload):
                continue

            text = content_text(payload)
            if should_skip_text(text):
                continue

            role = payload["role"]
            heading = "User" if role == "user" else "Assistant"
            timestamp = obj.get("timestamp")

            lines.append(f"## {heading}")
            if timestamp:
                lines.extend(["", f"_Timestamp: {timestamp}_"])
            lines.extend(["", text, ""])
            exported_messages += 1

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return exported_messages


def main() -> int:
    args = parse_args()

    if not args.session_id:
        try:
            sessions = list_sessions()
        except (FileNotFoundError, sqlite3.Error) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print(format_session_header())
        for session in sessions:
            print(format_session_row(session))
        return 0

    output_path = Path(args.output or f"codex-session-{args.session_id}.md")

    try:
        thread = find_thread(args.session_id)
        message_count = export_markdown(thread, output_path)
    except (FileNotFoundError, json.JSONDecodeError, LookupError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {output_path} ({message_count} messages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
