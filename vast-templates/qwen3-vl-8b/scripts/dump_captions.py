#!/usr/bin/env python3
"""Dump recent captions from the filter cache. Usage:

    python dump_captions.py /workspace/openwebui-data/img_captions.db
"""
import sqlite3
import sys


def main(path: str, limit: int = 50) -> None:
	con = sqlite3.connect(path)
	rows = con.execute(
		"SELECT img_hash, substr(caption,1,80), model, created_at "
		"FROM captions ORDER BY created_at DESC LIMIT ?",
		(limit,),
	)
	for h, cap, model, ts in rows:
		print(f"{h[:12]}  [{model}]  {ts}  {cap}")


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print(__doc__)
		sys.exit(2)
	main(sys.argv[1])
