"""Optional slurmrestd client (stdlib only).

Used by `sweep` to list recent failures with less overhead than shelling out
to sacct. Falls back cleanly: every caller must handle None returns.

Endpoint selection via SLURM_DOCTOR_RESTD, e.g.
    unix:///var/run/slurmrestd/slurmrestd.socket
    http://slurmrestd:6820
Auth: JWT from `scontrol token` (AuthAltTypes=auth/jwt).
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import re
import socket
import urllib.parse

from .util import run

log = logging.getLogger("slurm_doctor.rest")

API_VERSIONS = ("v0.0.44", "v0.0.43", "v0.0.42", "v0.0.41", "v0.0.40")


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float = 10.0):
        super().__init__("localhost", timeout=timeout)
        self._unix_path = path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._unix_path)
        self.sock = sock


class RestClient:
    def __init__(self, url: str, timeout: float = 10.0):
        self.url = url
        self.timeout = timeout
        self._token: str | None = None
        self._version: str | None = None

    @classmethod
    def from_env(cls, configured: str | None) -> "RestClient | None":
        url = configured or os.environ.get("SLURM_DOCTOR_RESTD")
        if not url:
            # default probe: the well-known local socket, then localhost TCP
            for cand in ("unix:///var/run/slurmrestd/slurmrestd.socket",
                         "http://localhost:6820", "http://slurmrestd:6820"):
                client = cls(cand)
                if client.ping():
                    return client
            return None
        client = cls(url)
        return client if client.ping() else None

    # -- low-level ----------------------------------------------------------
    def _connect(self) -> http.client.HTTPConnection:
        if self.url.startswith("unix://"):
            return _UnixHTTPConnection(self.url[len("unix://"):], self.timeout)
        parsed = urllib.parse.urlparse(self.url)
        return http.client.HTTPConnection(
            parsed.hostname or "localhost", parsed.port or 6820, timeout=self.timeout
        )

    def _token_header(self) -> dict[str, str]:
        if self._token is None:
            r = run(["scontrol", "token", "lifespan=600"], timeout=10)
            m = re.search(r"SLURM_JWT=(\S+)", r.stdout)
            self._token = m.group(1) if (r.ok and m) else ""
        hdrs = {"Accept": "application/json"}
        if self._token:
            hdrs["X-SLURM-USER-TOKEN"] = self._token
            user = os.environ.get("USER") or os.environ.get("LOGNAME")
            if user:
                hdrs["X-SLURM-USER-NAME"] = user
        return hdrs

    def get(self, path: str) -> dict | None:
        try:
            conn = self._connect()
            conn.request("GET", path, headers=self._token_header())
            resp = conn.getresponse()
            body = resp.read()
            conn.close()
            if resp.status != 200:
                log.debug("restd GET %s -> %s", path, resp.status)
                return None
            return json.loads(body)
        except (OSError, ValueError) as exc:
            log.debug("restd GET %s failed: %s", path, exc)
            return None

    # -- API ----------------------------------------------------------------
    def ping(self) -> bool:
        return self.version() is not None

    def version(self) -> str | None:
        if self._version:
            return self._version
        for v in API_VERSIONS:
            if self.get(f"/slurm/{v}/ping") is not None:
                self._version = v
                log.debug("slurmrestd answering on %s (%s)", self.url, v)
                return v
        return None

    def recent_jobs(self, since_epoch: int) -> list[dict] | None:
        """All accounting jobs updated since *since_epoch* (slurmdb dump)."""
        v = self.version()
        if not v:
            return None
        data = self.get(f"/slurmdb/{v}/jobs?update_time={since_epoch}")
        if data is None:
            return None
        return data.get("jobs", [])


def rest_failed_jobids(client: RestClient, since_epoch: int, states: set[str]) -> list[str] | None:
    jobs = client.recent_jobs(since_epoch)
    if jobs is None:
        return None
    out = []
    for j in jobs:
        st = j.get("state", {})
        current = st.get("current")
        if isinstance(current, list):
            current = current[0] if current else ""
        if str(current or "").upper() in states:
            out.append(str(j.get("job_id")))
    return out
