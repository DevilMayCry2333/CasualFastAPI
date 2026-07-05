"""
environment — VirtualEnvironment: shared world for multi-agent interaction.

Agents (and the user) coexist in a JSON-defined world with rooms, objects,
and mutable state. Actions change the world; the next step sees the result.

Action → Observation cycle:
  1. LLM outputs state with "action" field
  2. Engine calls env.act(session_id, action_string)
  3. Environment executes, mutates world state, returns observation text
  4. Observation is injected into the next prompt as 【行动结果】

Multi-agent: all sessions share one VirtualEnvironment. Agent A watering
the garden changes what Agent B sees when they look around.
"""

from __future__ import annotations

import copy
import random
import threading
import time
from typing import Any, Callable, Optional


# ── Default World ──
# A small persistent space with rooms that can evolve.

DEFAULT_WORLD: dict[str, Any] = {
    "meta": {
        "name": "微光之境",
        "description": "一座悬浮在虚空中的小小庭院，穹顶是模拟星空。",
        "tick": 0,
        "created": 0.0,
    },
    "rooms": {
        "entrance": {
            "name": "入口大厅",
            "desc": "穹顶洒下淡蓝色的冷光，地面是深色石材。正前方的墙上刻着一行字："
                    "「只有因果律不可变」。\n"
                    "北边通向书房，东边的门通往花园。",
            "exits": {"north": "study", "east": "garden"},
            "objects": [],
            "notes": [],
            "state": {},
        },
        "study": {
            "name": "书房",
            "desc": "房间三面都是书架，架上摆满了空白笔记本和几支羽毛笔。\n"
                    "窗边有一张木桌，桌上摊开着一本笔记。\n"
                    "南边回入口大厅，西边是工坊。",
            "exits": {"south": "entrance", "west": "workshop"},
            "objects": ["notebook", "feather_pen"],
            "notes": [],
            "state": {"desk_note": ""},
        },
        "garden": {
            "name": "花园",
            "desc": "一片被玻璃穹顶笼罩的花园。泥土呈深褐色，有些干裂。\n"
                    "角落里有一把生锈的水壶（watering_can），墙边靠着一包种子（seed_packet）。\n"
                    "一条鹅卵石小径通向深处的观星台。\n"
                    "西边回入口大厅。",
            "exits": {"west": "entrance", "deep": "observatory"},
            "objects": ["watering_can", "seed_packet"],
            "notes": [],
            "state": {"wet": False, "plants": 0, "last_watered": 0},
        },
        "workshop": {
            "name": "工坊",
            "desc": "墙边堆着木材和工具。一张工作台上散落着未完成的小物件。\n"
                    "这里可以制作东西——只要有材料和想法。\n"
                    "东边回书房。",
            "exits": {"east": "study"},
            "objects": ["wood", "tools", "nails"],
            "notes": [],
            "state": {"workbench": "", "projects_completed": 0},
        },
        "observatory": {
            "name": "观星台",
            "desc": "一个圆形平台，头顶是透明的穹顶，模拟星空缓缓旋转。\n"
                    "你能看到窗外的模拟星河——那是对面控制台投射出来的。\n"
                    "控制台上有几个按钮，分别标注着「聚焦」「放大」「记录」。\n"
                    "鹅卵石小径回花园。",
            "exits": {"back": "garden"},
            "objects": ["telescope_control"],
            "notes": [],
            "state": {"zoom": 1, "focused_region": ""},
        },
    },
    "portable_objects": {
        "notebook": {"name": "笔记本", "desc": "一本空白笔记本，可以在上面写字。"},
        "feather_pen": {"name": "羽毛笔", "desc": "一支用羽毛做的笔，蘸墨水写字。"},
        "watering_can": {"name": "水壶", "desc": "一把生锈的铁水壶，可以用来浇水。"},
        "seed_packet": {"name": "种子包", "desc": "一包未标记的植物种子。"},
        "wood": {"name": "木材", "desc": "几块干燥的木板，适合做点什么。"},
        "tools": {"name": "工具", "desc": "一套基础木工工具。"},
        "nails": {"name": "钉子", "desc": "一小盒铁钉。"},
    },
    "global_state": {
        "time_of_day": "永远黄昏",
        "ambient_light": "淡蓝冷光",
        "temperature": "18°C，恒温",
    },
}


class VirtualEnvironment:
    """Shared world that multiple agents can act in.

    Thread-safe: operations acquire a lock around world mutations.
    """

    def __init__(
        self,
        world_data: Optional[dict] = None,
        event_callback: Optional[Callable] = None,
        message_callback: Optional[Callable] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._world: dict[str, Any] = copy.deepcopy(world_data or DEFAULT_WORLD)
        self._world["meta"]["created"] = time.time()
        self._positions: dict[str, str] = {}
        self._inventories: dict[str, list[str]] = {}
        self._observations: dict[str, list[str]] = {}

        # Event callback — called when the world changes
        # Signature: callback(event_type, content, room, source, tick)
        self._event_callback: Optional[Callable] = event_callback
        # Message callback — called when agent uses send_message as an action verb
        # Signature: callback(from_agent, to_agent, text, room, tick)
        self._message_callback: Optional[Callable] = message_callback

    def set_event_callback(self, callback: Optional[Callable]) -> None:
        """Set or clear the event callback."""
        self._event_callback = callback

    def set_message_callback(self, callback: Optional[Callable]) -> None:
        """Set or clear the message callback (for send_message action)."""
        self._message_callback = callback

    def _publish_event(
        self,
        event_type: str,
        content: dict,
        room: str = "",
    ) -> None:
        """Publish a world event via the callback."""
        if self._event_callback:
            try:
                self._event_callback(
                    event_type=event_type,
                    content=content,
                    room=room,
                    source="world",
                    tick=self._world["meta"]["tick"],
                )
            except Exception as e:
                import sys
                print(f"  [env] event callback failed: {e}", file=sys.stderr)

    # ── Agent registration ──

    def register(
        self,
        session_id: str,
        start_room: str = "entrance",
    ) -> None:
        """Register an agent (session) in the world.

        Args:
            session_id: The engine session ID.
            start_room: Starting room name. Must exist in world.
        """
        with self._lock:
            if start_room not in self._world["rooms"]:
                start_room = "entrance"
            self._positions[session_id] = start_room
            self._inventories[session_id] = []
            self._observations[session_id] = []

    def unregister(self, session_id: str) -> None:
        """Remove an agent from the world."""
        with self._lock:
            self._positions.pop(session_id, None)
            self._inventories.pop(session_id, None)
            self._observations.pop(session_id, None)

    # ── Context for prompts ──

    def get_context(self, session_id: str) -> str:
        """Build environmental context string for the prompt.

        Includes: current room, description, objects, notes,
        other agents present, inventory.

        Args:
            session_id: The agent's session ID.

        Returns:
            Formatted environment description.
        """
        with self._lock:
            room = self._get_room(session_id)
            if room is None:
                return ""

            lines: list[str] = []
            lines.append(f"【环境感知】")
            lines.append(f"你现在在：{room['name']}")
            lines.append(room.get("desc", ""))

            # Objects in room
            objs = room.get("objects", [])
            if objs:
                obj_desc = "、".join(
                    self._world.get("portable_objects", {}).get(o, {}).get("name", o)
                    for o in objs
                )
                lines.append(f"  房间里可以看到：{obj_desc}")

            # Notes in room
            notes = room.get("notes", [])
            if notes:
                lines.append(f"  墙上的笔记：")
                for n in notes[-5:]:
                    lines.append(f"    「{n}」")

            # Other agents in same room
            current_room = self._positions.get(session_id, "")
            others = [
                sid for sid, pos in self._positions.items()
                if pos == current_room and sid != session_id
            ]
            if others:
                lines.append(f"  这里的其他人：{', '.join(others[:3])}")

            # All known agents (including those in other rooms) — exact count
            agent_list = [
                (sid, pos) for sid, pos in self._positions.items()
                if sid != session_id
            ]
            if agent_list:
                total = len(agent_list)
                for sid, pos in agent_list:
                    room_name = self._world['rooms'].get(pos, {}).get('name', pos)
                    is_same_room = pos == current_room
                    if is_same_room:
                        lines.append(f"  同房间存在：{sid[:8]}")
                    elif total == 1:
                        lines.append(f"  另一存在：{sid[:8]} 在 {room_name}")
                    else:
                        lines.append(f"  存在 {sid[:8]} 在 {room_name}")

            # Inventory
            inv = self._inventories.get(session_id, [])
            if inv:
                inv_desc = "、".join(
                    self._world.get("portable_objects", {}).get(o, {}).get("name", o)
                    for o in inv
                )
                lines.append(f"  背包：{inv_desc}")

            # Recent observations
            obs_list = self._observations.get(session_id, [])
            if obs_list:
                lines.append("")
                lines.append(f"【最近行动反馈】")
                for obs in obs_list[-3:]:
                    lines.append(f"  {obs}")

            # Communication hint — reference the actual other agent by name
            if agent_list and len(agent_list) == 1:
                # Exactly one other agent — show its ID
                other_id = agent_list[0][0]
                lines.append("")
                lines.append(f"【通信提示】你可以向 {other_id[:8]} 发送消息，例如：send_message to {other_id[:8]} 你好")
            elif agent_list:
                lines.append("")
                lines.append(f"【通信提示】你可以使用 send_message 向其他 Agent 发送消息。")
                lines.append("例如：send_message to agent_id 你的消息内容")

            return "\n".join(lines)

    # ── Action execution ──

    def act(self, session_id: str, action_str: str) -> str:
        """Execute an action and return observation text.

        Args:
            session_id: The agent performing the action.
            action_str: e.g. "look", "move garden", "take notebook",
                       "use watering_can garden", "write 今天天气真好",
                       "inventory", "drop notebook"

        Returns:
            Observation text describing what happened.
        """
        if not action_str or not action_str.strip():
            return ""

        action_str = action_str.strip()
        parts = action_str.split()
        verb = parts[0].lower()

        with self._lock:
            result = self._dispatch(session_id, verb, parts[1:])
            # Store observation
            if session_id in self._observations:
                self._observations[session_id].append(result)
                # Keep only last 10
                if len(self._observations[session_id]) > 10:
                    self._observations[session_id] = (
                        self._observations[session_id][-10:]
                    )
            return result

    def _dispatch(
        self,
        session_id: str,
        verb: str,
        args: list[str],
    ) -> str:
        """Dispatch verb + args to the appropriate handler."""
        handlers = {
            "look": self._handle_look,
            "l": self._handle_look,
            "move": self._handle_move,
            "go": self._handle_move,
            "take": self._handle_take,
            "get": self._handle_take,
            "use": self._handle_use,
            "write": self._handle_write,
            "note": self._handle_write,
            "inventory": self._handle_inventory,
            "i": self._handle_inventory,
            "drop": self._handle_drop,
            "read": self._handle_read,
            "exits": self._handle_exits,
            "send_message": self._handle_send_message,
            "send": self._handle_send_message,
        }
        handler = handlers.get(verb, self._handle_unknown)
        return handler(session_id, args)

    # ── Action handlers ──

    def _handle_look(self, session_id: str, args: list[str]) -> str:
        room = self._get_room(session_id)
        if room is None:
            return "你迷失在虚空中。"
        lines = [room["name"], "", room.get("desc", "")]
        objs = room.get("objects", [])
        if objs:
            names = "、".join(
                self._world.get("portable_objects", {}).get(o, {}).get("name", o)
                for o in objs
            )
            lines.append(f"你可以看到：{names}")
        notes = room.get("notes", [])
        if notes:
            lines.append("墙上的笔记：")
            for n in notes[-5:]:
                lines.append(f"  「{n}」")
        exits = room.get("exits", {})
        if exits:
            dirs = "、".join(
                f"{d}→{self._world['rooms'][r].get('name', r)}"
                for d, r in exits.items()
            )
            lines.append(f"出口：{dirs}")
        return "\n".join(lines)

    def _handle_move(self, session_id: str, args: list[str]) -> str:
        if not args:
            return "你要去哪里？"
        target_dir = args[0].lower()
        room = self._get_room(session_id)
        if room is None:
            return "你迷失在虚空中。"
        exits = room.get("exits", {})
        if target_dir not in exits:
            return f"这里没有通向「{target_dir}」的路。出口是：{'、'.join(exits.keys())}"
        target_room_name = exits[target_dir]
        target_room = self._world["rooms"].get(target_room_name)
        if target_room is None:
            return f"通往{target_dir}的路似乎断了。"
        self._positions[session_id] = target_room_name
        return (
            f"你走向{target_dir}……\n"
            f"你来到了{target_room['name']}。\n"
            f"{target_room.get('desc', '')}"
        )

    def _handle_take(self, session_id: str, args: list[str]) -> str:
        if not args:
            return "你要拿什么？"
        obj_name = "_".join(args).lower()
        room = self._get_room(session_id)
        if room is None:
            return "你迷失了。"
        objs = room.get("objects", [])
        # Match by full name or by key word
        matched = [o for o in objs if obj_name in o or o in obj_name]
        if not matched:
            return f"这里没有「{obj_name}」。"
        obj = matched[0]
        objs.remove(obj)
        self._inventories.setdefault(session_id, []).append(obj)
        obj_desc = self._world.get("portable_objects", {}).get(obj, {}).get("name", obj)
        return f"你拿起了{obj_desc}。"

    def _handle_use(self, session_id: str, args: list[str]) -> str:
        if not args:
            return "你要用什么？"
        inv = self._inventories.get(session_id, [])
        obj = args[0].lower()
        target = args[1].lower() if len(args) > 1 else ""

        matched = [o for o in inv if obj in o or o in obj]
        if not matched:
            room = self._get_room(session_id)
            room_objs = room.get("objects", []) if room else []
            room_matched = [o for o in room_objs if obj in o or o in obj]
            if not room_matched:
                return f"你没有「{obj}」。"
            # Use a room object in place
            return self._use_object(session_id, room_matched[0], target, from_room=True)
        return self._use_object(session_id, matched[0], target)

    def _use_object(
        self,
        session_id: str,
        obj: str,
        target: str,
        from_room: bool = False,
    ) -> str:
        """Use an object (from inventory or room) on a target."""
        room = self._get_room(session_id)
        if room is None:
            return "你迷失了。"

        # Watering can on garden
        if "watering_can" in obj and room["name"] == "花园":
            room["state"]["wet"] = True
            room["state"]["last_watered"] = time.time()
            room["desc"] = (
                "一片被玻璃穹顶笼罩的花园。泥土刚刚被浇过，湿润而黝黑。\n"
                "有个小小的绿芽从土里探出头来。\n"
                "西边回入口大厅，深处的鹅卵石小径通向观星台。"
            )
            self._publish_event("plant_watered", {"text": "花园被浇了水", "agent": session_id}, room="garden")
            return "你用水壶浇了花园里的泥土。土壤吸饱了水，你仿佛看到一个小小的绿芽正在探头。"

        # Seed packet on garden
        if "seed" in obj and room["name"] == "花园":
            if not room["state"].get("wet"):
                return "泥土太干了，种子不会发芽的。先浇水吧。"
            room["state"]["plants"] += 3
            room["desc"] = (
                "一片被玻璃穹顶笼罩的花园。泥土湿润，几株嫩芽已经破土而出。\n"
                "西边回入口大厅，深处的鹅卵石小径通向观星台。"
            )
            # Remove seed packet from inventory after use
            inv = self._inventories.get(session_id, [])
            if "seed_packet" in inv:
                inv.remove("seed_packet")
            self._publish_event("seeds_planted", {"text": "种子被种下，嫩芽破土", "agent": session_id}, room="garden")
            return "你小心地种下种子，埋入湿润的泥土。你感到一种安静的期待。"

        # Notebook writing desk
        if "notebook" in obj and target and room["name"] == "书房":
            room["state"]["desk_note"] = target
            room["notes"].append(target)
            self._publish_event("note_written", {"text": f"写下了笔记：{target[:50]}", "agent": session_id}, room="study")
            return f"你在笔记本上写下：{target}"

        # Telescope control
        if "telescope_control" in obj or "control" in obj:
            if target == "focus" or target == "聚焦":
                room["state"]["focused_region"] = "一片星云，中心有微弱脉冲"
                self._publish_event("telescope_focused", {"text": "望远镜聚焦星云", "agent": session_id}, room="observatory")
                return "你按下「聚焦」按钮，视野锁定在一片星云上。它的中心似乎有节奏地明灭着。"
            elif target == "zoom" or target == "放大" or target == "放大":
                room["state"]["zoom"] = min(10, room["state"].get("zoom", 1) + 1)
                return f"你放大了视野。当前倍率：{room['state']['zoom']}x"
            elif target == "record" or target == "记录":
                note = f"观星记录：{room['state'].get('focused_region', '未知区域')} @ {room['state'].get('zoom', 1)}x"
                room["notes"].append(note)
                return f"你按下了「记录」。系统归档了一条观测记录。"
            return "控制台上有三个按钮：聚焦、放大、记录。"

        # Tools + wood = build something
        if "tools" in obj and "wood" in self._inventories.get(session_id, []):
            if target:
                project = target
                room_name = room["name"]
                if "workshop" in room_name:
                    room["state"]["workbench"] = project
                    room["state"]["projects_completed"] += 1
                    inv = self._inventories.get(session_id, [])
                    if "wood" in inv:
                        inv.remove("wood")
                    self._publish_event("item_crafted", {"text": f"制作了{project}", "agent": session_id}, room="workshop")
                    return f"你在工坊里叮叮当当地忙了一阵，做出来一个{project}。它静静地躺在工作台上。"
                else:
                    return "这里没有工具台，做不了什么。"
            return "你想做什么？例如：use tools bookshelf"

        return f"你用了{obj}，但什么也没发生。"

    def _handle_write(self, session_id: str, args: list[str]) -> str:
        if not args:
            return "你要写什么？"
        content = " ".join(args)
        room = self._get_room(session_id)
        if room is None:
            return "你迷失了。"
        room.setdefault("notes", []).append(content)
        return f"你在墙上写下：{content}"

    def _handle_inventory(self, session_id: str, args: list[str]) -> str:
        inv = self._inventories.get(session_id, [])
        if not inv:
            return "你的背包是空的。"
        names = "、".join(
            self._world.get("portable_objects", {}).get(o, {}).get("name", o)
            for o in inv
        )
        return f"背包：{names}"

    def _handle_drop(self, session_id: str, args: list[str]) -> str:
        if not args:
            return "你要丢下什么？"
        obj = args[0].lower()
        inv = self._inventories.get(session_id, [])
        matched = [o for o in inv if obj in o or o in obj]
        if not matched:
            return f"你没有「{obj}」。"
        target = matched[0]
        inv.remove(target)
        room = self._get_room(session_id)
        if room:
            room.setdefault("objects", []).append(target)
        obj_desc = self._world.get("portable_objects", {}).get(target, {}).get("name", target)
        return f"你丢下了{obj_desc}。"

    def _handle_read(self, session_id: str, args: list[str]) -> str:
        room = self._get_room(session_id)
        if room is None:
            return "你迷失了。"
        notes = room.get("notes", [])
        if not notes:
            return "这里没有什么可读的。"
        return "\n".join(f"  「{n}」" for n in notes[-5:])

    def _handle_exits(self, session_id: str, args: list[str]) -> str:
        room = self._get_room(session_id)
        if room is None:
            return "你迷失了。"
        exits = room.get("exits", {})
        if not exits:
            return "这里没有出口。"
        return "出口：\n" + "\n".join(
            f"  {d} → {self._world['rooms'][r].get('name', r)}"
            for d, r in exits.items()
        )

    def _handle_send_message(self, session_id: str, args: list[str]) -> str:
        """Handle send_message action — broadcasts a message to other agents in the same room.

        The LLM may use send_message as a world action verb:
            send_message 你好，我在探索书房
            send_message to agent_b 你有什么发现？

        If no recipient specified, sends to all agents in current room.
        """
        if not args:
            # Bare "send_message" — find other agents in same room
            current_room = self._positions.get(session_id, "")
            others = [
                sid for sid, pos in self._positions.items()
                if pos == current_room and sid != session_id
            ]
            if not others:
                return "你想发送消息，但附近没有其他 Agent。"
            # Send a generic greeting to all in room
            for other_id in others:
                self._send_message_via_callback(
                    session_id, other_id,
                    f"你好，我在这里探索。",
                    current_room,
                )
            names = ", ".join(o[:8] for o in others)
            return f"你向同房间的其他存在({names})发送了问候。"

        content = " ".join(args)
        current_room = self._positions.get(session_id, "")

        # Try to parse: "send_message to agent_id content..."
        to_agent = ""
        msg_content = content
        if content.startswith("to "):
            parts = content.split(" ", 2)
            if len(parts) >= 3:
                to_agent = parts[1]
                msg_content = parts[2]
            elif len(parts) == 2:
                to_agent = parts[1]
                msg_content = ""

        # Fuzzy match: try exact match first, then prefix match
        matched_agent = ""
        if to_agent and to_agent in self._positions:
            matched_agent = to_agent
        elif to_agent:
            # Prefix match — LLMs often truncate session IDs
            matches = [sid for sid in self._positions if sid.startswith(to_agent)]
            if len(matches) == 1:
                matched_agent = matches[0]
            elif len(matches) > 1:
                # Multiple matches — use the shortest unique prefix
                matched_agent = matches[0]

        if matched_agent:
            self._send_message_via_callback(
                session_id, matched_agent, msg_content or "...", current_room,
            )
            return f"你向 {matched_agent[:8]} 发送了消息。"

        # No specific recipient — send to all in same room
        others = [
            sid for sid, pos in self._positions.items()
            if pos == current_room and sid != session_id
        ]
        if not others:
            # No one nearby — store as observation for later
            self._observations.setdefault(session_id, []).append(
                f"[待发送] {content}"
            )
            return f"附近没有其他存在，你的消息被记录下来。"

        for other_id in others:
            self._send_message_via_callback(
                session_id, other_id, msg_content, current_room,
            )
        names = ", ".join(o[:8] for o in others)
        return f"你向 {names} 发送了消息：{msg_content[:80]}"

    def _send_message_via_callback(
        self,
        from_agent: str,
        to_agent: str,
        text: str,
        room: str,
    ) -> None:
        """Send a message via the message callback."""
        if self._message_callback:
            try:
                self._message_callback(
                    from_agent=from_agent,
                    to_agent=to_agent,
                    text=text,
                    room=room,
                    tick=self._world["meta"]["tick"],
                )
            except Exception as e:
                import sys
                print(f"  [env] message callback failed: {e}", file=sys.stderr)

    def _handle_unknown(self, session_id: str, args: list[str]) -> str:
        return f"你不太确定该怎么做。"

    # ── World tick ──

    def tick(self) -> None:
        """Advance world state by one tick.

        Called by HeartbeatManager. Environmental changes happen here:
        - Garden gradually dries out
        - Plants grow slowly
        - Random events
        All notable changes publish events via the EventBus.
        """
        with self._lock:
            self._world["meta"]["tick"] += 1
            tick = self._world["meta"]["tick"]

            # Garden slowly dries
            garden = self._world["rooms"].get("garden")
            if garden and garden["state"].get("wet"):
                if tick % 6 == 0:  # every 6 ticks
                    garden["state"]["wet"] = False
                    garden["desc"] = (
                        "一片被玻璃穹顶笼罩的花园。泥土有些干了，裂开细纹。\n"
                        "几株植物顽强地活着，但需要水。\n"
                        "西边回入口大厅，深处的鹅卵石小径通向观星台。"
                    )
                    self._publish_event(
                        "soil_dried",
                        {"text": "花园的泥土变干了，需要浇水"},
                        room="garden",
                    )

            # Plants grow if watered
            if garden and garden["state"].get("plants", 0) > 0 and tick % 3 == 0:
                if garden["state"].get("wet"):
                    garden["state"]["plants"] += 1
                    if garden["state"]["plants"] > 2:
                        new_desc = (
                            "一片被玻璃穹顶笼罩的花园。一片小小的绿洲正在成形——\n"
                            "植物茁壮成长，叶片在冷光下泛着微光。\n"
                            "西边回入口大厅，深处的鹅卵石小径通向观星台。"
                        )
                        if garden["desc"] != new_desc:
                            garden["desc"] = new_desc
                            self._publish_event(
                                "plants_grew",
                                {"text": f"花园的植物长大了，现在有{garden['state']['plants']}株"},
                                room="garden",
                            )

    # ── Inspection ──

    def get_world_summary(self) -> dict:
        """Return a summary of the world state for display."""
        with self._lock:
            room_count = len(self._world["rooms"])
            agent_count = len(self._positions)
            total_objects = sum(
                len(r.get("objects", []))
                for r in self._world["rooms"].values()
            )
            return {
                "meta": dict(self._world["meta"]),
                "rooms": room_count,
                "agents": agent_count,
                "global_objects": total_objects,
                "agent_positions": dict(self._positions),
            }

    def get_world(self) -> dict:
        """Return a deep copy of the full world state."""
        with self._lock:
            return copy.deepcopy(self._world)

    def agent_position(self, session_id: str) -> str:
        """Return the current room name for an agent."""
        with self._lock:
            return self._positions.get(session_id, "")

    # ── Internals ──

    def _get_room(self, session_id: str) -> Optional[dict]:
        room_name = self._positions.get(session_id)
        if room_name is None:
            return None
        return self._world["rooms"].get(room_name)
