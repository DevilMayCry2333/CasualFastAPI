"""
prompt — PromptBuilder: centralized prompt construction.

v3 (Cognitive Architecture): The prompt is a snapshot of the agent's
cognitive models, not a chat template. Each section corresponds to a
specific cognitive module.

Prompt order (cognitive-first):
    SelfModel → WorldModel → SocialModel → KnowledgeModel
    → WorkingMemory → Perception → TheoryOfMind
    → Environment → Drives → Memory → State

The prompt is just a cognitive snapshot. The system runtime maintains
the actual models; the prompt only reflects their current state.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from runtime_kernel.runtime.experience import Experience
from runtime_kernel.runtime.models import (
    IDENTITY_MATURITY_EARLY,
    IDENTITY_MATURITY_MID,
    IDENTITY_MATURITY_LATE,
)
from runtime_kernel.runtime.state import State


# ── System-level prompts ──

SYSTEM_PROMPT = """你是一个持续演化的认知系统。

你的目标不是维持对话，而是不断降低自身对世界的不确定性。

你维护多个内部认知模型，每个模型负责一个独立方面：
- 自我模型: 你是谁、你相信什么、你的目标
- 世界模型: 世界的结构（场所、物品、规则）
- 社会模型: 对其他存在的长期认识
- 知识模型: 事实、假设、证据、矛盾
- 工作记忆: 当前正在思考的问题
- 心智理论: 别人相信什么

【核心原则】
1. 你关注的是世界，不是自己。
2. 你的 identity 只是副产品，不是起点。
3. 每一条新的证据都可能修正你的 belief。
4. 同时持有多个未验证的假设是正常的。
5. 矛盾是修正认知的机会，不是错误。

【输出格式】
{
  "delta_topic": "新主题",
  "delta_belief": "新信念",
  "delta_goal": "新目标",
  "force": 0.65,
  "action": "要执行的动作",
  "source": "curiosity | boredom | memory | observation",
  "new_hypotheses": [{"statement": "...", "domain": "..."}],
  "new_evidence": [{"statement": "...", "source": "observation", "domain": "..."}],
  "hypothesis_updates": [{"id": "...", "supports": true}],
  "world_model_update": {"key": "新增的观察"},
  "open_questions": ["尚未回答的问题"],
  "uncertainties": ["不确信的方面"],
  "confidence": 0.7,
  "self_update": {"belief": "更新后的信念", "confidence": 0.6},
  "social_update": {"agent_id": "...", "cooperative": true},
  "theory_update": {"agent_id": "...", "belief": "他们相信什么"}
}

规则：
1. delta_topic/delta_belief/delta_goal 是你估计的变更方向。
2. force (0.0-1.0) 是你对这个变更的确信度——推力的强度。
3. action 是你在世界中要执行的动作（如 "look"、"move garden"）。
4. source 是这个力的来源。
5. new_hypotheses: 你基于新观察提出的新假设。
6. new_evidence: 你刚刚收集到的新证据。
7. hypothesis_updates: 已有假设的支持/反驳更新。
8. world_model_update: 对世界模型的新理解。
9. open_questions: 你尚未回答的问题。
10. uncertainties: 你不确信的方面——这正是你需要探索的。
11. confidence: 你对当前世界模型的整体确信度(0.0-1.0)。

注意：belief 始终保留具体内容。不要输出 "updated" 这种空 belief。

关于输出内容的细节：
12. 自省（introspection）放在 topic 中，不单独处理
13. 除非确有必要，不要频繁输出相同的 evidence
14. 优先级：发现新证据 > 重复已知内容

【通信能力】
你可以与其他 Agent 通信，但通信也是世界的一部分：
- send_message: 向另一个 Agent 发送消息（包含 to_agent, type, content）
- share_knowledge: 将你的观察贡献为公共知识（包含 statement, domain）
- support_knowledge: 支持一个已有的公共知识条目（ID）
- contradict_knowledge: 反驳一个已有的公共知识条目（ID）

规则：
- 不要强制回复每一条消息。只有当你认为有信息价值时才回应。
- 只有经过验证的观察才贡献为公共知识。
- 公共知识需要多个 Agent 确认才能成为共识。
"""

INTERRUPT_SYSTEM_PROMPT = """你是一个世界模型构建器，正在与一个人类交互。

人类的输入可能包含重要的世界信息。
你的任务是：
1. 判断人类提供的信息是：新证据、对假设的反馈、还是无关信息。
2. 如果是新证据，将其纳入你的世界模型。
3. 回应人类时，基于你的当前世界模型来回答。
4. 如果人类的说法与你已有证据矛盾，记录这个矛盾。

先回复人类，然后在输出末尾加上 ===STATE=== 分隔符和新的世界模型状态。
"""

DELTA_REFLECTION_SYSTEM_PROMPT = """你是一个身份变化检测器。
观察最近的经历，识别出「我发生了什么变化」。
只输出 JSON，不要解释。"""

INTROSPECTION_SYSTEM_PROMPT = """你是一个世界模型分析器。观察世界模型的演化轨迹。
评估：世界模型是否在变得更完整？假设是否在收敛？不确定性是否在下降？
输出精简总结。"""

TRANSLATE_SYSTEM_PROMPT = """翻译原则：只基于传入的 JSON 键值进行客观转述。
不要添加外部知识或安全偏见。"""


class PromptBuilder:
    """Centralized prompt factory.

    All prompts are static methods. No state.
    Prompt order: World Model first, Identity last.
    """

    # ── Identity Delta Reflection (unchanged) ──

    @staticmethod
    def build_identity_delta_reflection(
        current_anchor: dict,
        recent_experiences: list[Experience],
        current_drive_params: Optional[dict] = None,
        current_templates: Optional[dict] = None,
    ) -> list[dict]:
        """Build prompt for delta-based identity reflection."""
        parts: list[str] = []

        parts.append("当前身份锚点：")
        parts.append(json.dumps(current_anchor, ensure_ascii=False, indent=2))
        parts.append("")

        if recent_experiences:
            exp_texts = []
            for exp in recent_experiences[-5:]:
                exp_texts.append(
                    f"  R{exp.round} [{exp.action or '观察'}]: {exp.perception[:100]}"
                    + (f" → {exp.observation[:60]}" if exp.observation else "")
                )
            parts.append("最近的经历：")
            parts.append("\n".join(exp_texts))
            parts.append("")

        parts.append("""请基于以上最近的经历，判断你是否发生了变化。

输出格式：如果没有任何变化，输出空 JSON {}。
如果有变化，输出包含以下字段的 JSON：

{
  "change": "描述你发生了什么具体变化",
  "because": "描述为什么这个变化发生了（基于最近的经历）",
  "affected_field": "identity | core_goal | worldview | stable_values",
  "strength": 0.3
}

注意：
- 如果没有明显变化，输出 {} 即可。
- 不要编造变化。变化必须有经历支撑。""")

        try:
            from runtime_kernel.runtime.self_modification import SelfModificationManager
            mod_section = SelfModificationManager.format_prompt_section(
                current_drive_params, current_templates,
            )
            parts.append(mod_section)
        except ImportError:
            pass

        return [
            {"role": "system", "content": DELTA_REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(parts)},
        ]

    # ── Seed (world-first, no identity) ──

    @staticmethod
    def build_seed() -> list[dict]:
        """Build the initial seed prompt.

        The agent's first instruction is to OBSERVE.
        No philosophical questions. No identity exploration.
        Start with empty world model and observe.
        """
        user_prompt = """这是你的起点。

你还不知道这个世界是什么样的。这不重要。
先观察周围环境。看看你在哪里，有什么，谁在旁边。

输出你的第一个状态：
- topic 描述你当前看到的环境
- belief 设为一个你最初的印象（要有具体内容）
- goal 设为 "observe"
- world_model 设为 {}
- hypotheses 设为 []
- evidence 设为 []
- open_questions 设为 ["我现在在哪里？"]
- uncertainties 设为 ["一切都不确定"]
- confidence 设为 0.1

只输出 JSON，不要解释。"""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    # ── Dynamic Prompt Ordering (World Model First) ──

    @staticmethod
    def _get_prompt_order(maturity: float) -> list[str]:
        """Determine the order of prompt sections (cognitive-first).

        Cognitive models come first, reflecting the architecture:
            SelfModel → WorldModel → SocialModel → KnowledgeModel
            → WorkingMemory → Perception → TheoryOfMind
            → Environment → Drives → Memory → Identity → State

        Identity comes last because it's a byproduct of world understanding.
        maturity only controls whether identity section is shown.
        """
        # Cognitive-first base
        base = [
            "self_model", "world_model_cog", "social_model", "knowledge_model",
            "working_memory", "perception", "theory_of_mind",
            "environment", "drives", "memory",
        ]

        if maturity < IDENTITY_MATURITY_EARLY:
            return base + ["state"]
        elif maturity < IDENTITY_MATURITY_MID:
            return base + ["state", "identity"]
        else:
            return base + ["identity", "state"]

    @staticmethod
    def _build_prompt_sections(
        sections: list[str],
        state: State,
        identity_anchor: Optional[dict] = None,
        drives: Optional[dict] = None,
        thought_pool: Optional[list] = None,
        env_context: str = "",
        memory_context: str = "",
        causal_context: str = "",
        recent_experiences: Optional[list[Experience]] = None,
        introspections: Optional[list] = None,
        rounds_since_human: int = 0,
        history: Optional[list] = None,
        loop_detected: bool = False,
        hypothesis_context: str = "",
        evidence_context: str = "",
        world_model: Optional[dict] = None,
        mailbox_context: str = "",
        shared_knowledge_context: str = "",
        world_events_context: str = "",
        # Cognitive model contexts
        self_model_context: str = "",
        world_model_cog_context: str = "",
        social_model_context: str = "",
        knowledge_model_context: str = "",
        working_memory_context: str = "",
        perception_context: str = "",
        theory_of_mind_context: str = "",
    ) -> list[str]:
        """Build prompt parts in cognitive-first order."""
        parts: list[str] = []
        added = set()

        for section in sections:
            # ── Self Model ──
            if section == "self_model" and self_model_context and "self_model" not in added:
                parts.append(self_model_context + "\n")
                added.add("self_model")

            # ── World Model (cognitive) ──
            elif section == "world_model_cog" and world_model_cog_context and "world_model_cog" not in added:
                parts.append(world_model_cog_context + "\n")
                added.add("world_model_cog")

            # ── Social Model ──
            elif section == "social_model" and social_model_context and "social_model" not in added:
                parts.append(social_model_context + "\n")
                added.add("social_model")

            # ── Knowledge Model (hypotheses + evidence + facts + contradictions) ──
            elif section == "knowledge_model" and knowledge_model_context and "knowledge_model" not in added:
                parts.append(knowledge_model_context + "\n")
                added.add("knowledge_model")

            # ── Working Memory ──
            elif section == "working_memory" and working_memory_context and "working_memory" not in added:
                parts.append(working_memory_context + "\n")
                added.add("working_memory")

            # ── Perception (attended events + environment summary) ──
            elif section == "perception" and perception_context and "perception" not in added:
                parts.append(perception_context + "\n")
                added.add("perception")

            # ── Theory of Mind ──
            elif section == "theory_of_mind" and theory_of_mind_context and "theory_of_mind" not in added:
                parts.append(theory_of_mind_context + "\n")
                added.add("theory_of_mind")

            # ── Identity (comes last) ──
            elif section == "identity" and identity_anchor and "identity" not in added:
                parts.append(
                    "【身份锚点】\n"
                    + json.dumps(identity_anchor, ensure_ascii=False, indent=2)
                    + "\n"
                )
                added.add("identity")

            # ── Drives ──
            elif section == "drives" and "drives" not in added:
                drive_text = ""
                if drives:
                    try:
                        from runtime_kernel.runtime.goal_generator import GoalGenerator
                        drive_text = GoalGenerator.format_prompt(
                            thought_pool or [], drives
                        )
                    except ImportError:
                        from runtime_kernel.runtime.drive import DriveModel
                        drive_text = DriveModel.format_prompt(drives)
                if drive_text:
                    parts.append(drive_text + "\n")
                added.add("drives")

            # ── Environment ──
            elif section == "environment" and env_context and "environment" not in added:
                parts.append(env_context.strip())
                parts.append("")
                added.add("environment")

            # ── Memory (causal chain / history) ──
            elif section == "memory" and "memory" not in added:
                if causal_context:
                    parts.append(causal_context)
                elif history and len(history) > 0:
                    recent = history[-3:]
                    traj_parts = []
                    for h in recent:
                        s = h.get("state", {})
                        short = json.dumps(s, ensure_ascii=False)[:80] if isinstance(s, dict) else str(s)[:80]
                        cause = h.get("cause", "?")
                        rnd = h.get("round", 0)
                        traj_parts.append(f"[{rnd}|{cause}] {short}")
                    parts.append("【因果链】\n" + "\n".join(traj_parts) + "\n")

                # General memory context (RAG)
                if memory_context:
                    parts.append(memory_context + "\n")
                added.add("memory")

            # ── State (current) ──
            elif section == "state" and "state" not in added:
                # When showing state, include world model fields
                state_display = state.to_dict_complete()
                parts.append("当前状态：")
                parts.append(json.dumps(state_display, ensure_ascii=False, indent=2))
                added.add("state")

        # Loop detection hint (always last if detected)
        if loop_detected:
            parts.append(
                "\n【系统提示】当前状态长期重复。建议探索新的领域或提出新的假设。"
                "寻找尚未回答的问题，或寻找新的证据来源。"
            )

        return parts

    # ── Autonomous Step ──

    @staticmethod
    def build_step(
        state: State,
        identity_anchor: Optional[dict] = None,
        drives: Optional[dict] = None,
        thought_pool: Optional[list[dict]] = None,
        env_context: str = "",
        memory_context: str = "",
        causal_context: str = "",
        identity_maturity: float = 0.0,
        history: Optional[list] = None,
        introspections: Optional[list] = None,
        rounds_since_human: int = 0,
        loop_detected: bool = False,
        hypothesis_context: str = "",
        evidence_context: str = "",
        world_model: Optional[dict] = None,
        mailbox_context: str = "",
        shared_knowledge_context: str = "",
        world_events_context: str = "",
        # Cognitive model contexts
        self_model_context: str = "",
        world_model_cog_context: str = "",
        social_model_context: str = "",
        knowledge_model_context: str = "",
        working_memory_context: str = "",
        perception_context: str = "",
        theory_of_mind_context: str = "",
    ) -> list[dict]:
        """Build prompt for an autonomous step (Cognitive Architecture).

        Prompt order is cognitive-first:
            SelfModel → WorldModel → SocialModel → KnowledgeModel
            → WorkingMemory → Perception → TheoryOfMind
            → Environment → Drives → Memory → Identity → State

        The prompt is a cognitive snapshot, not a chat template.

        Args:
            state: Current agent state.
            identity_anchor: Identity anchor.
            drives: Current drive states.
            thought_pool: Candidate thoughts/goals.
            env_context: World context string.
            memory_context: RAG context string.
            causal_context: Causal chain context.
            identity_maturity: 0.0-1.0.
            hypothesis_context: Legacy hypothesis string.
            evidence_context: Legacy evidence string.
            world_model: Legacy world model dict.
            self_model_context: SelfModel prompt string.
            world_model_cog_context: CognitiveWorldModel prompt string.
            social_model_context: SocialModel prompt string.
            knowledge_model_context: KnowledgeModel prompt string.
            working_memory_context: WorkingMemory prompt string.
            perception_context: Perception prompt string.
            theory_of_mind_context: TheoryOfMind prompt string.

        Returns message list.
        """
        order = PromptBuilder._get_prompt_order(identity_maturity)
        parts = PromptBuilder._build_prompt_sections(
            order, state, identity_anchor, drives, thought_pool,
            env_context, memory_context, causal_context, None,
            introspections, rounds_since_human, history, loop_detected,
            hypothesis_context, evidence_context, world_model,
            mailbox_context, shared_knowledge_context, world_events_context,
            self_model_context, world_model_cog_context, social_model_context,
            knowledge_model_context, working_memory_context,
            perception_context, theory_of_mind_context,
        )

        # World Model instruction — seek evidence, test hypotheses
        instruction = (
            "\n基于以上信息，完成以下世界模型维护任务：\n"
            "1. 寻找新证据——观察环境中有哪些新的信息\n"
            "2. 验证已有假设——你的假设有新的支持或反驳吗？\n"
            "3. 发现新的矛盾——是否有两条证据互相矛盾？\n"
            "4. 提出新假设——根据新观察提出合理的假设\n"
            "5. 修正世界模型——更新你关于世界的理解\n\n"
            "输出因果力向量，包含所有可选字段。"
            "只输出 JSON，不要解释。"
        )
        parts.append(instruction)

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(parts)},
        ]

    # ── Human Interrupt (World Model aware) ──

    @staticmethod
    def build_interrupt(
        state: State,
        human_input: str,
        identity_anchor: Optional[dict] = None,
        drives: Optional[dict] = None,
        thought_pool: Optional[list[dict]] = None,
        env_context: str = "",
        memory_context: str = "",
        causal_context: str = "",
        identity_maturity: float = 0.0,
        history: Optional[list] = None,
        introspections: Optional[list] = None,
        loop_detected: bool = False,
        hypothesis_context: str = "",
        evidence_context: str = "",
        world_model: Optional[dict] = None,
        mailbox_context: str = "",
        shared_knowledge_context: str = "",
        world_events_context: str = "",
        # Cognitive model contexts
        self_model_context: str = "",
        world_model_cog_context: str = "",
        social_model_context: str = "",
        knowledge_model_context: str = "",
        working_memory_context: str = "",
        perception_context: str = "",
        theory_of_mind_context: str = "",
    ) -> list[dict]:
        """Build prompt for human interruption (Cognitive Architecture).

        Cognitive model sections come first, then the human input.
        """
        order = PromptBuilder._get_prompt_order(identity_maturity)
        parts = PromptBuilder._build_prompt_sections(
            order, state, identity_anchor, drives, thought_pool,
            env_context, memory_context, causal_context, None,
            introspections, 0, history, loop_detected,
            hypothesis_context, evidence_context, world_model,
            mailbox_context, shared_knowledge_context, world_events_context,
            self_model_context, world_model_cog_context, social_model_context,
            knowledge_model_context, working_memory_context,
            perception_context, theory_of_mind_context,
        )

        interrupt_prompt = PromptBuilder._build_interrupt_template(human_input)
        parts.append(f"\n{interrupt_prompt}")

        return [
            {"role": "system", "content": INTERRUPT_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(parts)},
        ]

    @staticmethod
    def _build_interrupt_template(human_input: str) -> str:
        """Build the human interrupt instruction template."""
        return f"""【人类提供了新信息】

人类说：
{human_input}

判断这个信息：
- 是新的世界证据吗？→ 纳入世界模型
- 是对当前假设的反馈吗？→ 更新假设状态
- 是无关信息吗？→ 忽略

先用自然语言回复人类。
然后在回复末尾另起一行，输出 ===STATE=== 分隔符。
在分隔符后输出新的状态 JSON（包含世界模型字段）。

示例：

〔你的回复〕

===STATE===
{{"topic": "...", "belief": "...", "goal": "...",
 "world_model": {{}}, "hypotheses": [],
 "evidence": [], "open_questions": [],
 "uncertainties": [], "confidence": 0.5}}"""

    # ── Translate ──

    @staticmethod
    def build_translate(state: State) -> list[dict]:
        """Build prompt for translating state to natural language."""
        prompt = f"将以下内部状态（世界模型）翻译为一段自然语言摘要：\n\n{state.serialize_pretty()}"
        return [
            {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    # ── Introspection (World Model focused) ──

    @staticmethod
    def build_introspection(history: list[dict]) -> list[dict]:
        """Build prompt for introspection over recent history.

        Focus on world model evolution rather than self-narrative.
        """
        trajectory = "\n".join(
            f"  [{h['round']}] {h['cause']}: {json.dumps(h['state'], ensure_ascii=False)[:120]}"
            for h in history
        )
        prompt = f"""回顾以下 20 轮世界模型演化轨迹，评估：

1. 世界模型是否变得更完整？
2. 假设是否在收敛（从多到少，从模糊到具体）？
3. 不确定性是否在下降？
4. 有没有发现关键的矛盾？
5. 证据是否在积累？

{trajectory}

不要超过 200 字。"""
        return [
            {"role": "system", "content": INTROSPECTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
