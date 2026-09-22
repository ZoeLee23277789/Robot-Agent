"""
Robot server 的 HTTP client。
只用標準函式庫，所有阻塞呼叫都丟到 thread 裡，這樣 agent loop 可以維持 async，
跟原本 browser_use 的寫法一致。
"""

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any, Optional


def check_link(health: dict) -> None:
    """server 活著不代表機器人還連著。模擬按過停止之後，舊的 server 會變成空殼。"""
    if health.get("robot_link") == "lost":
        raise SystemExit(
            "❌ robot_server 還在，但它跟機器人的連線已經斷了 (" + str(health.get("detail")) + ")\n"
            "   修復：1) CoppeliaSim 按播放  2) 在 server 的終端機按 Ctrl+C  3) 重新執行 ./start_server.sh")


class RobotClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8765", token: str = "", action_timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.action_timeout = action_timeout

    # ---- 同步底層 ----
    def _request(self, method: str, path: str, body: Optional[dict] = None, timeout: float = 10.0):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("X-Robot-Token", self.token)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.headers.get("Content-Type", ""), resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read()

    def _json(self, method: str, path: str, body: Optional[dict] = None, timeout: float = 10.0) -> dict[str, Any]:
        _, _, raw = self._request(method, path, body, timeout)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {"ok": False, "message": raw[:200].decode("utf-8", "replace")}

    def _safe_json(self, method: str, path: str) -> dict[str, Any]:
        try:
            return self._json(method, path)
        except Exception as e:  # server 沒開、網路斷線等等，回傳錯誤而不是丟例外
            return {"ok": False, "message": f"cannot reach robot server at {self.base_url}: {e}"}

    # ---- async API ----
    async def health(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._safe_json, "GET", "/health")

    async def state(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._safe_json, "GET", "/state")

    async def actions(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._safe_json, "GET", "/actions")

    async def frame(self, raw_mode: bool = False) -> Optional[bytes]:
        """raw_mode=True 會跳過 server 的「等新畫面」機制，只用在量測相機延遲。"""
        try:
            status, ctype, raw = await asyncio.to_thread(
                self._request, "GET", "/frame.jpg?raw=1" if raw_mode else "/frame.jpg", None, 15.0)
        except Exception:
            return None
        return raw if status == 200 and ctype.startswith("image/") else None

    async def overhead_frame(self) -> Optional[bytes]:
        """房間正上方的俯視圖，要先跑 add_overhead_camera.py；沒有的話回傳 None。"""
        try:
            status, ctype, raw = await asyncio.to_thread(self._request, "GET", "/overhead.jpg", None, 15.0)
        except Exception:
            return None
        return raw if status == 200 and ctype.startswith("image/") else None

    async def act(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(
                self._json, "POST", "/action", {"name": name, "params": params}, self.action_timeout
            )
        except Exception as e:
            # 連線層出錯時不知道機器人停了沒，保險起見補送一次 stop
            await self.stop()
            return {"ok": False, "message": f"transport error: {type(e).__name__}: {e}"}

    async def reset(self) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(self._json, "POST", "/reset", {}, 30.0)
        except Exception as e:
            return {"ok": False, "message": f"reset failed: {e}"}

    async def stop(self) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(self._json, "POST", "/stop", {}, 5.0)
        except Exception as e:
            return {"ok": False, "message": f"stop failed: {e}"}
