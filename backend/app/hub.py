"""WebSocket connection hub: routes real-time events between the AI service,
the citizen display and employee / admin dashboards. No polling anywhere."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import anyio.from_thread
from fastapi import WebSocket

from .deps import Principal

log = logging.getLogger("govsign.hub")


@dataclass(eq=False)
class Conn:
    ws: WebSocket
    principal: Principal
    desks: set[int] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def is_ai(self) -> bool:
        return self.principal.kind == "kiosk" and self.principal.component == "ai"

    @property
    def is_display(self) -> bool:
        return self.principal.kind == "kiosk" and self.principal.component == "display"

    @property
    def is_staff(self) -> bool:
        return self.principal.kind == "user"


class Hub:
    def __init__(self) -> None:
        self.conns: set[Conn] = set()
        self.last_status: dict[int, dict] = {}     # desk_id -> last AI status event

    # ---- registry ------------------------------------------------------- #
    def add(self, conn: Conn) -> None:
        self.conns.add(conn)

    def remove(self, conn: Conn) -> None:
        self.conns.discard(conn)

    def ai_online(self, desk_id: int) -> bool:
        return any(c.is_ai and desk_id in c.desks for c in self.conns)

    # ---- sending -------------------------------------------------------- #
    async def send(self, conn: Conn, event: dict) -> bool:
        try:
            async with conn.lock:
                await conn.ws.send_json(event)
            return True
        except Exception:                      # connection already gone
            self.conns.discard(conn)
            return False

    async def to_desk(self, desk_id: int, event: dict, *, only: str | None = None,
                      exclude: Conn | None = None) -> int:
        """Send to every connection attached to a desk.

        only: 'ai' | 'display' | 'staff' restricts the audience.
        """
        sent = 0
        for conn in list(self.conns):
            if desk_id not in conn.desks or conn is exclude:
                continue
            if only == "ai" and not conn.is_ai:
                continue
            if only == "display" and not conn.is_display:
                continue
            if only == "staff" and not conn.is_staff:
                continue
            if await self.send(conn, event):
                sent += 1
        return sent

    async def to_admins(self, event: dict) -> int:
        sent = 0
        for conn in list(self.conns):
            if conn.is_staff and conn.principal.role == "ADMIN":
                if await self.send(conn, event):
                    sent += 1
        return sent

    async def to_all_ai(self, event: dict) -> int:
        sent = 0
        for conn in list(self.conns):
            if conn.is_ai and await self.send(conn, event):
                sent += 1
        return sent

    # ---- helpers for synchronous (threadpool) endpoints ------------------ #
    def emit_desk(self, desk_id: int, event: dict, **kw) -> None:
        try:
            anyio.from_thread.run(lambda: self.to_desk(desk_id, event, **kw))
        except RuntimeError:                   # not inside a worker thread (e.g. scripts)
            log.debug("emit_desk outside of event loop thread; dropped %s", event.get("type"))

    def emit_admins(self, event: dict) -> None:
        try:
            anyio.from_thread.run(lambda: self.to_admins(event))
        except RuntimeError:
            pass

    def emit_ai(self, event: dict) -> None:
        try:
            anyio.from_thread.run(lambda: self.to_all_ai(event))
        except RuntimeError:
            pass

    async def to_staff(self, event: dict) -> int:
        sent = 0
        for conn in list(self.conns):
            if conn.is_staff and await self.send(conn, event):
                sent += 1
        return sent

    def emit_staff(self, event: dict) -> None:
        try:
            anyio.from_thread.run(lambda: self.to_staff(event))
        except RuntimeError:
            pass
