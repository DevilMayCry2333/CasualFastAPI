"""
client — Low-level MCP client implementing JSON-RPC 2.0 over stdio or HTTP.

Transport modes:
    stdio:  spawn MCP server as subprocess, communicate via stdin/stdout
    http:   HTTP-based transport — auto-detects SSE vs direct POST:
              sse:     SSE stream (GET for events, POST for requests)
              direct:  each request = one HTTP POST to the URL

No tool-specific logic lives here. Tool semantics are in runtime.py.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Optional

from runtime_kernel.runtime.mcp.models import MCPConfig


class MCPError(RuntimeError):
    """Raised when an MCP protocol-level error occurs."""
    pass


class MCPClient:
    """JSON-RPC 2.0 client for MCP protocol (stdio / SSE / direct HTTP).

    Usage:
        client = MCPClient(config)
        client.connect()
        tools = client.discover_tools()
        result = client.call_tool("web_search", {"query": "..."})
        client.close()
    """

    def __init__(self, config: MCPConfig) -> None:
        self._config = config

        # ── Transport selection ──
        if config.url:
            self._transport: str = "http"
        elif config.command:
            self._transport = "stdio"
        else:
            raise MCPError("MCPConfig must provide either url or command")

        # ── Common state ──
        self._request_id: int = 0
        self._connected: bool = False
        self._server_info: dict = {}

        # ── Stdio state ──
        self._process: Optional[subprocess.Popen] = None

        # ── HTTP transport state (used by both SSE and direct POST) ──
        self._http_mode: str = ""           # "sse" or "direct"
        self._message_url: str = ""
        self._sse_response: Optional[Any] = None
        self._sse_thread: Optional[threading.Thread] = None
        self._sse_running: bool = False
        self._endpoint_event = threading.Event()
        self._pending_responses: dict[int, dict] = {}
        self._response_events: dict[int, threading.Event] = {}

    # ══════════════════════════════════════════════
    # Connection lifecycle
    # ══════════════════════════════════════════════

    def connect(self) -> dict:
        """Connect to the MCP server and perform the handshake.

        Dispatches to stdio or HTTP transport based on config.
        Returns the server info from the initialize response.

        Raises MCPError if connection or handshake fails.
        """
        if self._transport == "http":
            result = self._connect_http()
        else:
            result = self._connect_stdio()

        if result:
            self._connected = True
            self._server_info = result
        return result

    def _connect_stdio(self) -> dict:
        """Connect via stdio transport: spawn subprocess + handshake."""
        cmd = [self._config.command] + list(self._config.args)
        merged_env = dict(os.environ)
        if self._config.env:
            merged_env.update(self._config.env)

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=merged_env,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            raise MCPError(
                f"MCP server command not found: {self._config.command!r}. "
                f"Ensure it is installed and on PATH."
            )
        except Exception as e:
            raise MCPError(f"Failed to start MCP server: {e}")

        try:
            result = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "casual-fastapi", "version": "0.1.0"},
            })
        except Exception as e:
            self._read_stderr()
            self._cleanup_stdio()
            raise MCPError(f"MCP initialize failed: {e}")

        self._send_notification("notifications/initialized")
        return result or {}

    def _connect_http(self) -> dict:
        """Connect via HTTP transport.

        Auto-detects SSE vs direct POST mode:
            1. Try GET with SSE headers
            2. If Content-Type is text/event-stream → SSE mode
            3. Otherwise → direct HTTP POST mode

        For SSE: the GET stream stays open, POST to endpoint URL
        For direct POST: each request is a standalone HTTP POST
        """
        import requests

        url = self._config.url
        print(f"  [MCP] Connecting via HTTP: {url[:80]}...", file=sys.stderr)

        # ── 1. Probe with GET to detect transport ──
        try:
            probe = requests.get(
                url,
                stream=True,
                headers={"Accept": "text/event-stream"},
                timeout=max(10.0, self._config.timeout / 3),
            )
            probe.raise_for_status()
            content_type = probe.headers.get("Content-Type", "")
        except requests.exceptions.RequestException as e:
            # GET failed — try direct POST mode
            print(f"  [MCP] GET probe failed ({e}), using direct POST mode", file=sys.stderr)
            self._http_mode = "direct"
            self._message_url = url
            result = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "casual-fastapi", "version": "0.1.0"},
            })
            self._send_notification("notifications/initialized")
            return result or {}

        # ── 2. Detect mode from Content-Type ──
        if "text/event-stream" in content_type:
            self._http_mode = "sse"
            print(f"  [MCP] Detected SSE transport", file=sys.stderr)

            # Keep the response for streaming
            self._sse_response = probe
            self._sse_running = True
            self._sse_thread = threading.Thread(
                target=self._sse_reader, daemon=True, name="mcp-sse-reader",
            )
            self._sse_thread.start()

            # Wait for endpoint event
            if not self._endpoint_event.wait(timeout=self._config.timeout):
                self._message_url = url
                print(f"  [MCP] No endpoint event, using original URL", file=sys.stderr)
        else:
            # Not SSE — use direct HTTP POST
            self._http_mode = "direct"
            print(f"  [MCP] Detected direct HTTP POST transport", file=sys.stderr)
            probe.close()
            self._message_url = url

        # ── 3. MCP initialize handshake ──
        try:
            result = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "casual-fastapi", "version": "0.1.0"},
            })
        except Exception as e:
            self._cleanup_http()
            raise MCPError(f"MCP initialize over HTTP failed: {e}")

        self._send_notification("notifications/initialized")
        return result or {}

    def close(self) -> None:
        """Close the MCP connection and clean up resources."""
        if self._transport == "http":
            self._cleanup_http()
        else:
            self._cleanup_stdio()

    def is_connected(self) -> bool:
        """Check if the client is currently connected."""
        if not self._connected:
            return False
        if self._transport == "stdio":
            return self._process is not None and self._process.poll() is None
        return self._sse_running or self._http_mode == "direct"

    # ══════════════════════════════════════════════
    # Tool operations
    # ══════════════════════════════════════════════

    def list_tools(self) -> list[dict]:
        """Call tools/list and return the tool list."""
        if not self.is_connected():
            raise MCPError("MCP client is not connected")
        result = self._send_request("tools/list", {})
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> dict:
        """Call a tool on the MCP server."""
        if not self.is_connected():
            raise MCPError("MCP client is not connected")
        params: dict[str, Any] = {"name": name}
        if arguments:
            params["arguments"] = arguments
        return self._send_request("tools/call", params)

    # ══════════════════════════════════════════════
    # Low-level JSON-RPC (dispatched by transport)
    # ══════════════════════════════════════════════

    def _send_request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request, dispatched by transport."""
        if self._transport == "http":
            if self._http_mode == "sse":
                return self._send_request_sse(method, params)
            else:
                return self._send_request_http_direct(method, params)
        else:
            return self._send_request_stdio(method, params)

    def _send_notification(self, method: str, params: Optional[dict] = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        notification = {"jsonrpc": "2.0", "method": method}
        if params:
            notification["params"] = params

        if self._transport == "http":
            import requests
            try:
                requests.post(
                    self._message_url,
                    json=notification,
                    headers={"Content-Type": "application/json"},
                    timeout=self._config.timeout,
                )
            except requests.exceptions.RequestException:
                pass  # Fire-and-forget
        else:
            self._write_json_stdin(notification)

    # ── Stdio transport ──

    def _send_request_stdio(self, method: str, params: dict) -> dict:
        """Send JSON-RPC via stdio and wait for response."""
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        self._write_json_stdin(request)

        deadline = time.time() + self._config.timeout
        while time.time() < deadline:
            line = self._read_line_stdout()
            if line is None:
                self._read_stderr()
                raise MCPError(
                    f"MCP server closed connection while waiting for "
                    f"{method!r} (id={self._request_id})"
                )

            response = self._try_parse_line(line)
            if response is not None:
                return response

        raise MCPError(f"MCP request {method!r} timed out after {self._config.timeout}s")

    # ── SSE transport ──

    def _send_request_sse(self, method: str, params: dict) -> dict:
        """Send JSON-RPC via POST to message URL, response via SSE stream."""
        import requests

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        response_event = threading.Event()
        self._response_events[self._request_id] = response_event

        try:
            resp = requests.post(
                self._message_url,
                json=request,
                headers={"Content-Type": "application/json"},
                timeout=self._config.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            self._response_events.pop(self._request_id, None)
            raise MCPError(f"HTTP POST to MCP endpoint failed: {e}")

        if response_event.wait(timeout=self._config.timeout):
            result = self._pending_responses.pop(self._request_id, {})
            self._response_events.pop(self._request_id, None)
            if "error" in result:
                err = result["error"]
                raise MCPError(err.get("message", "Unknown MCP error"))
            return result.get("result", {})
        else:
            self._response_events.pop(self._request_id, None)
            raise MCPError(f"MCP request {method!r} timed out")

    # ── Direct HTTP POST transport (POST → JSON-RPC, handles SSE-wrapped responses) ──

    def _send_request_http_direct(self, method: str, params: dict) -> dict:
        """Send JSON-RPC via direct HTTP POST and return the response.

        Supports both:
          - Direct JSON-RPC response (Content-Type: application/json)
          - SSE-wrapped response (Content-Type: text/event-stream) — Tavily MCP
        """
        import requests

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        try:
            resp = requests.post(
                self._message_url,
                json=request,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream, application/json",
                },
                timeout=self._config.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise MCPError(f"HTTP POST to MCP endpoint failed: {e}")

        # Parse response: direct JSON or SSE-wrapped
        content_type = resp.headers.get("Content-Type", "")
        body = resp.text

        if "text/event-stream" in content_type:
            # Parse SSE format: extract the data line
            result = self._parse_sse_response(body)
        else:
            try:
                result = body
                result = json.loads(body)
            except json.JSONDecodeError as e:
                raise MCPError(f"MCP response is not valid JSON: {e}")

        if "error" in result:
            err = result["error"]
            raise MCPError(err.get("message", "Unknown MCP error"))
        return result.get("result", {})

    def _parse_sse_response(self, body: str) -> dict:
        """Parse an SSE-wrapped JSON-RPC response.

        Handles format:
            event: message
            data: {"jsonrpc": "2.0", ...}
        """
        current_event = ""
        for line in body.split("\n"):
            line = line.strip()
            if not line:
                current_event = ""
                continue
            if line.startswith("event:"):
                current_event = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
                if current_event in ("message", "") and data:
                    try:
                        return json.loads(data)
                    except json.JSONDecodeError:
                        pass
                current_event = ""
        raise MCPError("No valid JSON-RPC response found in SSE stream")

    # ══════════════════════════════════════════════
    # SSE reader thread
    # ══════════════════════════════════════════════

    def _sse_reader(self) -> None:
        """Background thread: read SSE events from the HTTP stream.

        Handles:
            event: endpoint  → POST URL for requests
            event: message   → JSON-RPC response for a request
            no event type    → fallback: parse as JSON-RPC response
        """
        current_event = ""
        try:
            for raw_line in self._sse_response.iter_lines(decode_unicode=True):
                if not self._sse_running:
                    break
                if raw_line is None:
                    continue

                line = raw_line.strip()
                if not line:
                    current_event = ""
                    continue
                if line.startswith("event:"):
                    current_event = line[6:].strip()
                elif line.startswith("data:"):
                    data = line[5:].strip()
                    evt_type = current_event

                    if evt_type == "endpoint":
                        if data.startswith("/"):
                            from urllib.parse import urlparse
                            parsed = urlparse(self._config.url)
                            self._message_url = f"{parsed.scheme}://{parsed.netloc}{data}"
                        else:
                            self._message_url = data
                        self._endpoint_event.set()
                        current_event = ""

                    elif evt_type in ("message", ""):
                        try:
                            response = json.loads(data)
                            req_id = response.get("id")
                            if req_id is not None:
                                self._pending_responses[req_id] = response
                                ev = self._response_events.get(req_id)
                                if ev:
                                    ev.set()
                        except json.JSONDecodeError:
                            pass
                        current_event = ""

        except Exception as e:
            if self._sse_running:
                print(f"  [MCP] SSE reader error: {e}", file=sys.stderr)

    # ══════════════════════════════════════════════
    # Stdio I/O helpers
    # ══════════════════════════════════════════════

    def _write_json_stdin(self, data: dict) -> None:
        if self._process is None or self._process.stdin is None:
            raise MCPError("Cannot write to MCP server (no stdin)")
        line = json.dumps(data, ensure_ascii=False) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()

    def _read_line_stdout(self) -> Optional[str]:
        if self._process is None or self._process.stdout is None:
            return None
        line = self._process.stdout.readline()
        if not line:
            return None
        return line.rstrip("\n").rstrip("\r")

    def _read_stderr(self) -> str:
        if self._process is None or self._process.stderr is None:
            return ""
        try:
            return self._process.stderr.read()
        except Exception:
            return ""

    def _try_parse_line(self, line: str) -> Optional[dict]:
        """Try to parse a JSON-RPC response line from stdout."""
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            return None
        if "id" not in response:
            return None
        if response["id"] == self._request_id:
            if "error" in response:
                err = response["error"]
                raise MCPError(err.get("message", "Unknown MCP error"))
            return response.get("result", {})
        print(
            f"  [MCP] Unexpected response id={response['id']} "
            f"(expected {self._request_id}), skipping",
            file=sys.stderr,
        )
        return None

    # ══════════════════════════════════════════════
    # Cleanup
    # ══════════════════════════════════════════════

    def _cleanup_stdio(self) -> None:
        if self._process is None:
            return
        proc = self._process
        self._process = None
        self._connected = False
        try:
            if proc.poll() is None:
                if sys.platform != "win32":
                    proc.send_signal(signal.SIGTERM)
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        except Exception:
            pass

    def _cleanup_http(self) -> None:
        """Close HTTP connection and stop SSE reader if running."""
        self._sse_running = False
        if self._sse_response:
            try:
                self._sse_response.close()
            except Exception:
                pass
            self._sse_response = None
        if self._sse_thread and self._sse_thread.is_alive():
            self._sse_thread.join(timeout=3)
        self._sse_thread = None
        self._connected = False

    def __enter__(self) -> "MCPClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
