"""
Shelly Gen2+ HTTP RPC Client

Low-level client for communicating with Shelly Gen2+ devices via JSON-RPC 2.0.
All methods POST to /rpc with a JSON body.

Transport:
  Requests are sent over HTTPS. Shelly devices use a self-signed certificate so
  hostname verification is disabled (same approach as other local-LAN integrations
  such as Philips Hue). Plain HTTP is intentionally avoided to prevent credentials
  and commands from being readable on the network.

Authentication (optional):
  Shelly Gen2+ uses HTTP digest authentication. Pass username and password to
  ShellyClient; aiohttp's DigestAuth will handle the challenge-response exchange.

References:
  https://shelly-api-docs.shelly.cloud/gen2/General/RPCProtocol
  https://shelly-api-docs.shelly.cloud/gen2/General/Authentication
"""

import logging
import ssl
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_rpc_id = 0


def _next_id() -> int:
    global _rpc_id
    _rpc_id = (_rpc_id + 1) % 100_000
    return _rpc_id


class ShellyRPCError(Exception):
    """Raised when the device returns an RPC error response."""

    def __init__(self, code: int, message: str):
        super().__init__(f"Shelly RPC error {code}: {message}")
        self.code = code
        self.message = message


class ShellyClient:
    """
    Async JSON-RPC 2.0 client for a single Shelly Gen2+ device.

    Usage::

        client = ShellyClient("192.168.1.50")
        result = await client.call("Switch.Set", {"id": 0, "on": True})
    """

    def __init__(
        self,
        host: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.host = host
        self.username = username
        self.password = password
        self.timeout = timeout
        self._url = f"https://{host}/rpc"
        # Shelly devices use self-signed TLS certs — disable hostname verification
        # while still encrypting the connection (same pattern as Philips Hue client).
        self._ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE

    async def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Call an RPC method on the device.

        Args:
            method: Method name, e.g. "Switch.Set"
            params: Optional parameters dict

        Returns:
            The ``result`` field of the RPC response.

        Raises:
            ShellyRPCError: If the device returns an error.
            aiohttp.ClientError: On network/HTTP issues.
        """
        import aiohttp

        payload: Dict[str, Any] = {"id": _next_id(), "method": method}
        if params:
            payload["params"] = params

        # Shelly Gen2+ uses HTTP Digest auth (challenge-response), not Basic auth.
        # DigestAuth prevents credentials from being sent in plaintext on the first request.
        auth = (
            aiohttp.DigestAuth(self.username, self.password)
            if self.username and self.password
            else None
        )

        logger.debug("POST %s  method=%s params=%s", self._url, method, params)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._url,
                json=payload,
                auth=auth,
                ssl=self._ssl_context,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                resp.raise_for_status()
                data: Dict[str, Any] = await resp.json(content_type=None)

        logger.debug("Response: %s", data)

        if "error" in data:
            err = data["error"]
            raise ShellyRPCError(err.get("code", -1), err.get("message", "unknown"))

        return data.get("result")
