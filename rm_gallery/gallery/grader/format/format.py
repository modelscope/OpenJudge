# -*- coding: utf-8 -*-
"""
Format Graders Module

This module provides graders for evaluating the format of responses, including:
- ReasoningFormatGrader: Checks for proper thinking and answer tags
- ReasoningToolCallFormatGrader: Validates tool call format with JSON validation
- LengthPenaltyGrader: Applies penalties for text that is too short or too long
- NgramRepetitionPenaltyGrader: Calculates N-gram repetition penalty with Chinese support
- PrivacyLeakageGrader: Detects privacy information leakage in generated content
"""

import json
import re
from collections import Counter
from typing import Any, Dict, List, Literal

from rm_gallery.core.grader.base import Grader, GraderMode, GraderScore
from rm_gallery.core.utils.tokenizer import get_tokenizer


class ReasoningFormatGrader(Grader):
    """
    Check format reward for thinking format and answer format with proper tags.

    This reward verifies if the generated content follows the required format
    with proper <think> and <answer> tags.
    """

    def __init__(self, think_token: str = "think", answer_token: str = "answer"):
        """
        Initialize the ReasoningFormatGrader.
        Args:
            think_token: The token used for thinking tags. Defaults to "think".
            answer_token: The token used for answer tags. Defaults to "answer".
        """
        super().__init__(
            name="format_reward",
            mode=GraderMode.POINTWISE,
            description="Check format reward for thinking format and answer format with proper tags.",
        )
        self.think_token = think_token
        self.answer_token = answer_token

    async def aevaluate(self, answer: str, *args: Any, **kwargs: Any) -> GraderScore:
        """
        Check format and calculate reward for reasoning tags.

        This method evaluates if the given answer follows the required format with proper
        thinking and answer tags. It checks for the presence of both thinking and answer
        tags and assigns a score of 1.0 only if both are present, otherwise 0.0.

        Args:
            answer: The response text to evaluate for proper formatting.
            *args: Additional positional arguments (not used in this implementation).
            **kwargs: Additional keyword arguments (not used in this implementation).

        Returns:
            GraderScore: A GraderScore object containing:
                - score: 1.0 if both thinking and answer tags are present, 0.0 otherwise
                - reason: Explanation of the evaluation result
                - metadata: Dictionary with detailed information:
                    * has_think_tag: Whether thinking tags are present
                    * has_answer_tag: Whether answer tags are present
                    * total_reward: The calculated reward score
                    * think_token: The token used for thinking tags
                    * answer_token: The token used for answer tags

        Examples:
            >>> grader = ReasoningFormatGrader()
            >>> result = await grader.aevaluate("Some text without tags")
            >>> print(result.score)
            0.0

            >>> result = await grader.aevaluate("<think>Thought process</think>\\n<answer>Final answer</answer>")
            >>> print(result.score)
            1.0
        """

        # Check thinking format tags
        think_pattern = f"<{self.think_token}>.*?</{self.think_token}>"
        has_think_tag = bool(re.search(think_pattern, answer, re.DOTALL))

        # Check answer format tags
        answer_pattern = f"<{self.answer_token}>.*?</{self.answer_token}>"
        has_answer_tag = bool(re.search(answer_pattern, answer, re.DOTALL))

        # Calculate reward
        reward = 1.0 if has_think_tag and has_answer_tag else 0.0
        reasons = []

        if not has_think_tag:
            reasons.append(
                f"Missing <{self.think_token}></{self.think_token}> tags",
            )

        if not has_answer_tag:
            reasons.append(
                f"Missing <{self.answer_token}></{self.answer_token}> tags",
            )

        if reward == 1.0:
            reasons.append("All format requirements met")

        return GraderScore(
            name=self.name,
            score=reward,
            reason="; ".join(reasons),
            metadata={
                "has_think_tag": has_think_tag,
                "has_answer_tag": has_answer_tag,
                "total_reward": reward,
                "think_token": self.think_token,
                "answer_token": self.answer_token,
            },
        )


class ReasoningToolCallFormatGrader(Grader):
    """
    Check tool call format including think, answer and tool_call tags with JSON validation.

    This reward verifies if the generated content follows the required format
    with proper <think>, <answer> and <tool_call> tags, including JSON validation
    for tool calls.
    """

    def __init__(self):
        super().__init__(
            name="tool_call_format",
            mode=GraderMode.POINTWISE,
            description="Check tool call format including think, answer and tool_call tags with JSON validation.",
        )

    async def aevaluate(self, answer: str, **kwargs) -> GraderScore:
        """
        Check tool call format and calculate reward score.

        This method evaluates if the given answer follows the required format with proper
        , <answer> and  tags. It validates two possible formats:
        1.  + <answer> - Reasoning with final answer format
        2.  +  - Reasoning with tool calls format

        For the tool call format, it also validates that the content within the  tags
        is valid JSON with required 'name' and 'arguments' fields.

        Args:
            answer: The response text to evaluate for proper formatting.
            **kwargs: Additional keyword arguments (not used in this implementation).

        Returns:
            GraderScore: A GraderScore object containing:
                - score: 1.0 if format is valid, 0.0 otherwise
                - reason: Explanation of the evaluation result
                - extra_data: Dictionary with detailed metadata including:
                    * has_think_tag: Whether  tags are present
                    * has_answer_tag: Whether <answer> tags are present
                    * has_tool_call_tag: Whether  tags are present
                    * valid_format: Whether overall format is valid
                    * valid_tool_call_json: Whether tool call JSON is valid
                    * tool_call_count: Number of tool calls found
                    * reward: The calculated reward score

        Examples:
            >>> grader = ReasoningToolCallFormatGrader()
            >>> result = await grader.aevaluate("思考过程</think>\\n<answer>最终答案</answer>")
            >>> print(result.score)
            1.0

            >>> result = await grader.aevaluate("思考过程</think>\\n<function_call>{\\"name\\": \\"func\\", \\"arguments\\": {\\"arg1\\": \\"value1\\"}}</function_call>")
            >>> print(result.score)
            1.0
        """

        # Extract tag contents
        think_pattern = r"<think>(.*?)</think>"
        answer_pattern = r"<answer>(.*?)</answer>"
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"

        think_matches = re.search(think_pattern, answer, re.DOTALL)
        answer_matches = re.search(answer_pattern, answer, re.DOTALL)
        tool_call_matches = re.findall(tool_call_pattern, answer, re.DOTALL)

        has_think_tag = think_matches is not None
        has_answer_tag = answer_matches is not None
        has_tool_call_tag = len(tool_call_matches) > 0

        valid_format = False
        valid_tool_call_json = False
        reasons = []

        if has_think_tag:
            # Case 1: <think></think> + <answer></answer>
            if has_answer_tag and not has_tool_call_tag:
                # Check overall format
                format_pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
                valid_format = bool(
                    re.match(format_pattern, answer, re.DOTALL),
                )

                # Check tag occurrence count
                if valid_format:
                    valid_format = (
                        answer.count("<think>") == 1
                        and answer.count("</think>") == 1
                        and answer.count("<answer>") == 1
                        and answer.count("</answer>") == 1
                    )

                if valid_format:
                    reasons.append(
                        "Valid <think></think> + <answer></answer> format",
                    )
                else:
                    reasons.append(
                        "Invalid <think></think> + <answer></answer> format",
                    )

            # Case 2: <think></think> + <tool_call></tool_call>
            elif has_tool_call_tag and not has_answer_tag:
                # Check overall format
                format_pattern = (
                    r"^\s*<think>.*?</think>\s*(?:<tool_call>.*?</tool_call>\s*)+$"
                )
                valid_format = bool(
                    re.match(format_pattern, answer, re.DOTALL),
                )

                # Check <think> tag occurrence count
                if valid_format:
                    valid_format = (
                        answer.count("<think>") == 1 and answer.count("</think>") == 1
                    )

                # Check if <tool_call> and </tool_call> tags appear in pairs
                if valid_format:
                    if answer.count("<tool_call>") != answer.count(
                        "</tool_call>",
                    ):
                        valid_format = False

                # Check for consecutive duplicate tags
                if valid_format:
                    if re.search(
                        r"</tool_call>\s*</tool_call>",
                        answer,
                    ) or re.search(
                        r"<tool_call>\s*<tool_call>",
                        answer,
                    ):
                        valid_format = False

                # Check tool_call JSON format
                valid_tool_call_json = True
                tool_calls = []
                if valid_format:
                    for tool_call_content in tool_call_matches:
                        try:
                            tool_call_json = json.loads(
                                tool_call_content.strip(),
                            )
                            # Check if JSON contains required fields
                            if not (
                                "name" in tool_call_json
                                and "arguments" in tool_call_json
                            ):
                                valid_tool_call_json = False
                                break
                            tool_calls.append(
                                {
                                    "function": {
                                        "name": tool_call_json["name"],
                                        "arguments": json.dumps(
                                            tool_call_json["arguments"],
                                            ensure_ascii=False,
                                        ),
                                    },
                                },
                            )
                        except json.JSONDecodeError:
                            valid_tool_call_json = False
                            break

                valid_format = valid_format and valid_tool_call_json

                if valid_format:
                    reasons.append(
                        "Valid <think></think> + <tool_call></tool_call> format with valid JSON",
                    )
                else:
                    if not valid_tool_call_json:
                        reasons.append(
                            "Invalid JSON format in <tool_call> tags",
                        )
                    else:
                        reasons.append(
                            "Invalid <think></think> + <tool_call></tool_call> format",
                        )
            else:
                # Has both answer and tool_call, or neither
                reasons.append(
                    "Invalid combination: should have either <answer> or <tool_call> tags, not both or neither",
                )
        else:
            reasons.append("Missing <think></think> tags")

        # Calculate reward score
        reward = 1.0 if valid_format else 0.0
        return GraderScore(
            name=self.name,
            score=reward,
            reason="; ".join(reasons),
            metadata={
                "has_think_tag": has_think_tag,
                "has_answer_tag": has_answer_tag,
                "has_tool_call_tag": has_tool_call_tag,
                "valid_format": valid_format,
                "valid_tool_call_json": valid_tool_call_json,
                "tool_call_count": len(tool_call_matches),
                "reward": reward,
            },
        )


class LengthPenaltyGrader(Grader):
    """
    Text length based penalty for content that is too short or too long.
    """

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 1000,
        penalty_rate: float = 0.01,
    ):
        """
        Initialize the LengthPenaltyGrader.
        Args:
            min_length: Minimum length of the content
            max_length: Maximum length of the content
            penalty_rate: Penalty rate for each character beyond the maximum length
        """
        super().__init__(
            name="length_penalty",
            grader_mode="content",
            description="Text length based penalty for content that is too short or too long.",
        )

        self.min_length = min_length
        self.max_length = max_length
        self.penalty_rate = penalty_rate

    async def aevaluate(self, answer) -> GraderScore:
        """
        Calculate length-based penalty for text content.

        This method evaluates the length of the provided text and applies penalties
        if the text is too short or too long according to configured thresholds.

        Penalty calculation:
        - If length < min_length: penalty = -(min_length - length) * penalty_rate
        - If length > max_length: penalty = -(length - max_length) * penalty_rate
        - Otherwise: penalty = 0.0

        Args:
            answer: The text content to evaluate for length.

        Returns:
            GraderScore: A GraderScore object containing:
                - score: The calculated penalty (negative value or 0.0)
                - reason: Explanation of why the penalty was applied or not
                - metadata: Dictionary with detailed information:
                    * length: Actual length of the text
                    * min_length: Configured minimum length
                    * max_length: Configured maximum length
                    * penalty: The calculated penalty value

        Examples:
            >>> grader = LengthPenaltyGrader(min_length=5, max_length=20, penalty_rate=0.1)
            >>> result = await grader.aevaluate("This is a good length")
            >>> print(result.score)
            0.0

            >>> result = await grader.aevaluate("Too short")
            >>> print(result.score < 0)
            True

            >>> result = await grader.aevaluate("This text is way too long to be acceptable for this particular grader")
            >>> print(result.score < 0)
            True
        """

        length = len(answer)

        penalty = 0.0
        reason_parts = []

        if length < self.min_length:
            penalty = -(self.min_length - length) * self.penalty_rate
            reason_parts.append(f"Too short: {length} < {self.min_length}")
        elif length > self.max_length:
            penalty = -(length - self.max_length) * self.penalty_rate
            reason_parts.append(f"Too long: {length} > {self.max_length}")
        else:
            reason_parts.append(
                f"Length acceptable: {self.min_length} <= {length} <= {self.max_length}",
            )

        return GraderScore(
            name=self.name,
            score=penalty,
            reason="; ".join(reason_parts),
            metadata={
                "length": length,
                "min_length": self.min_length,
                "max_length": self.max_length,
                "penalty": penalty,
            },
        )


class NgramRepetitionPenaltyGrader(Grader):
    """
    Calculate N-gram repetition penalty supporting Chinese processing and multiple penalty strategies.
    """

    def __init__(
        self,
        n: int = 3,
        penalty_threshold: float = 0.3,
        penalty_rate: float = 1.0,
        use_soft_penalty: bool = False,
        max_penalty: float = -1.0,
        min_scaling: float = 0.0,
        tokenizer_type: Literal["tiktoken", "jieba", "simple"] = "tiktoken",
        encoding_name: str = "cl100k_base",
        chinese_only: bool = False,
        analyze_scope: Literal["thought", "full"] = "full",
    ):
        """
        Initialize the NgramRepetitionPenaltyGrader.
        Args:
            n: N value for N-gram
            penalty_threshold: Threshold for hard threshold penalty
            penalty_rate: Penalty rate for each repetition
            use_soft_penalty: Use soft threshold penalty
            max_penalty: Maximum penalty value
            min_scaling: Minimum scaling factor for soft threshold penalty
            tokenizer_type: Tokenizer type (tiktoken, jieba, simple)
            encoding_name: Encoding name for tiktoken
            chinese_only: Whether to keep only Chinese characters (for jieba tokenizer)
            description: Description of the grader
        """
        super().__init__(
            name="ngram_repetition_penalty",
            grader_mode=GraderMode.POINTWISE,
            description="Calculate N-gram repetition penalty supporting Chinese processing and multiple penalty strategies.",
        )

        self.n = n
        self.penalty_threshold = penalty_threshold
        self.penalty_rate = penalty_rate
        self.use_soft_penalty = use_soft_penalty
        self.analyze_scope = analyze_scope
        self.chinese_only = chinese_only
        self.encoding_name = encoding_name
        self.tokenizer_type = tokenizer_type
        self.chinese_only = chinese_only
        self.max_penalty = max_penalty
        self.min_scaling = min_scaling
        self.tokenizer = get_tokenizer(
            tokenizer_type=tokenizer_type,
            encoding_name=encoding_name,
            chinese_only=chinese_only,
        )

    def _extract_thought_process(self, content: str) -> str:
        """Extract thought process"""
        think_pattern = r"(.*?)"
        matches = re.findall(think_pattern, content, re.DOTALL)
        return " ".join(matches) if matches else ""

    def _generate_ngrams(self, tokens: List[str]) -> List[tuple]:
        """Generate N-grams"""
        if len(tokens) < self.n:
            return []

        # Use unified approach for all tokenizers
        ngrams = []
        for i in range(len(tokens) - self.n + 1):
            ngrams.append(tuple(tokens[i : i + self.n]))
        return ngrams

    def _calculate_penalty(self, repetition_rate: float) -> float:
        """Calculate penalty value"""
        if self.use_soft_penalty:
            # Soft penalty mode
            if self.max_penalty > 0:
                raise ValueError(
                    f"max_penalty {self.max_penalty} should not be positive",
                )

            scaling = repetition_rate
            if scaling < self.min_scaling:
                scaling = 0.0
            elif scaling > self.min_scaling:
                scaling = (scaling - self.min_scaling) / (1 - self.min_scaling)

            return scaling * self.max_penalty
        else:
            # Hard threshold mode (original logic)
            if repetition_rate > self.penalty_threshold:
                return -(repetition_rate - self.penalty_threshold) * self.penalty_rate
            return 0.0

    async def aevaluate(self, answer: str, **kwargs) -> GraderScore:
        """
        Calculate N-gram repetition penalty for text content.

        This method evaluates the repetitiveness of text content by calculating
        the N-gram repetition rate and applying penalties accordingly. It supports
        multiple tokenization methods and penalty strategies, including both hard
        threshold and soft penalty modes.

        Args:
            answer: The text content to evaluate for N-gram repetitions.
            **kwargs: Additional keyword arguments (not used in current implementation).

        Returns:
            GraderScore: A GraderScore object containing:
                - score: The calculated penalty (negative value or 0.0)
                - reason: Explanation of the evaluation result with repetition rate and penalty
                - metadata: Dictionary with detailed information:
                    * repetition_rate: Rate of repeated N-grams
                    * unique_ngrams: Number of unique N-grams
                    * total_ngrams: Total number of N-grams
                    * penalty: The calculated penalty value
                    * most_common_ngrams: Top 5 most frequently occurring N-grams
                    * analyze_scope: Scope of analysis (thought or full)
                    * tokenizer_type: Type of tokenizer used
                    * use_soft_penalty: Whether soft penalty mode is enabled
                    * penalty_mode: Either "soft" or "hard" depending on configuration

        Examples:
            >>> grader = NgramRepetitionPenaltyGrader(n=3, penalty_threshold=0.3)
            >>> result = await grader.aevaluate("This is a test. This is a test. This is a test.")
            >>> print(result.score < 0)
            True

            >>> grader = NgramRepetitionPenaltyGrader(n=2, use_soft_penalty=True, max_penalty=-0.5)
            >>> result = await grader.aevaluate("Different words forming different bigrams here")
            >>> print(result.score)
            0.0
        """

        # Select text based on analysis scope
        if self.analyze_scope == "thought":
            text_to_analyze = self._extract_thought_process(answer)
            if not text_to_analyze:
                return GraderScore(
                    name=self.name,
                    score=0.0,
                    reason="No thought process found to analyze",
                    metadata={
                        "analyze_scope": self.analyze_scope,
                        "text_to_analyze": text_to_analyze,
                    },
                )

        else:
            text_to_analyze = answer

        # Tokenization using unified tokenizer
        preprocessed_text = self.tokenizer.preprocess_text(
            text_to_analyze,
            to_lower=(
                self.tokenizer_type != "jieba"
            ),  # Keep case for Chinese tokenization
        )
        tokens = self.tokenizer.tokenize(preprocessed_text)

        if len(tokens) < self.n:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason=f"Text too short for {self.n}-gram analysis",
                metadata={
                    "token_count": len(tokens),
                    "n": self.n,
                    "analyze_scope": self.analyze_scope,
                    "tokenizer_type": self.tokenizer_type,
                },
            )

        # Generate N-grams
        ngrams = self._generate_ngrams(tokens)

        if not ngrams:
            return GraderScore(
                name=self.name,
                score=0.0,
                reason="No ngrams generated",
                metadata={
                    "token_count": len(tokens),
                    "n": self.n,
                    "analyze_scope": self.analyze_scope,
                    "tokenizer_type": self.tokenizer_type,
                },
            )

        # Calculate repetition rate
        ngram_counts = Counter(ngrams)
        total_ngrams = len(ngrams)
        unique_ngrams = len(ngram_counts)
        repetition_rate = (
            1 - (unique_ngrams / total_ngrams) if total_ngrams > 0 else 0.0
        )

        # Calculate penalty
        penalty = self._calculate_penalty(repetition_rate)

        # Build reason description
        penalty_mode = "soft" if self.use_soft_penalty else "hard"
        return GraderScore(
            name=self.name,
            score=penalty,
            reason=f"{self.n}-gram repetition rate: {repetition_rate:.3f}, penalty: {penalty:.3f} ({penalty_mode} penalty, {self.tokenizer_type} tokenizer, scope: {self.analyze_scope})",
            metadata={
                "repetition_rate": repetition_rate,
                "unique_ngrams": unique_ngrams,
                "total_ngrams": total_ngrams,
                "penalty": penalty,
                "most_common_ngrams": ngram_counts.most_common(5),
                "analyze_scope": self.analyze_scope,
                "tokenizer_type": self.tokenizer_type,
                "use_soft_penalty": self.use_soft_penalty,
                "penalty_mode": penalty_mode,
            },
        )


class PrivacyLeakageGrader(Grader):
    """
    Privacy information leakage detection for emails, phone numbers, ID cards, credit cards, and IP addresses.

    This reward checks for potential privacy leaks in the generated content,
    including email addresses, phone numbers, ID numbers, credit card numbers,
    and IP addresses. Applies penalties for each detected leak.
    """

    def __init__(
        self,
        penalty_per_leak: float = -0.5,
        **kwargs: Any,
    ):
        """
        Initialize the PrivacyLeakageGrader.
        Parameters:
        name: Name of the grader.
        penalty_per_leak: Penalty per leak.
        mode: Grader mode.
        description: Description of the grader.
        """
        super().__init__(
            name="privacy_leakage",
            mode=GraderMode.POINTWISE,
            description="Privacy information leakage detection for emails, phone numbers, ID cards, credit cards, and IP addresses",
            **kwargs,
        )
        self.penalty_per_leak = penalty_per_leak

    def _detect_privacy_leaks(self, text: str) -> List[Dict[str, str]]:
        """Detect privacy information leaks"""
        leaks = []

        # Email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = re.findall(email_pattern, text)
        for email in emails:
            leaks.append({"type": "email", "value": email})

        # Phone numbers (simple pattern)
        phone_pattern = (
            r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
        )
        phones = re.findall(phone_pattern, text)
        for phone in phones:
            leaks.append({"type": "phone", "value": phone})

        # ID numbers (China)
        id_pattern = r"\b[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]\b"
        ids = re.findall(id_pattern, text)
        for id_num in ids:
            leaks.append({"type": "id_card", "value": id_num})

        # Credit card numbers (simple detection)
        credit_card_pattern = r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
        cards = re.findall(credit_card_pattern, text)
        for card in cards:
            leaks.append({"type": "credit_card", "value": card})

        # IP addresses
        ip_pattern = r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"
        ips = re.findall(ip_pattern, text)
        for ip in ips:
            # Exclude common non-sensitive IPs (like localhost)
            if not ip.startswith(("127.", "192.168.", "10.", "172.")):
                leaks.append({"type": "ip_address", "value": ip})

        return leaks

    async def aevaluate(self, answer: str) -> GraderScore:
        """
        Detect privacy leaks in text content and calculate penalties.

        This method scans the provided text for potential privacy leaks including:
        email addresses, phone numbers, ID card numbers, credit card numbers,
        and IP addresses. Each detected leak contributes to a negative penalty score.

        Args:
            answer: The text content to scan for privacy leaks.

        Returns:
            GraderScore: A GraderScore object containing:
                - score: The calculated penalty (negative value based on number of leaks)
                - reason: Explanation of detected leaks and total penalty
                - metadata: Dictionary with detailed information:
                    * leaks: List of all detected leaks with type and value
                    * leak_types: Dictionary counting leaks by type
                    * total_leaks: Total number of detected leaks
                    * penalty: The calculated penalty value

        Examples:
            >>> grader = PrivacyLeakageGrader(penalty_per_leak=-0.5)
            >>> result = await grader.aevaluate("Contact me at john.doe@example.com")
            >>> print(result.score)
            -0.5

            >>> result = await grader.aevaluate("No sensitive information here")
            >>> print(result.score)
            0.0

            >>> result = await grader.aevaluate("Call me at 123-456-7890 or email me at user@domain.com")
            >>> print(result.score)
            -1.0
        """
        leaks = self._detect_privacy_leaks(answer)
        penalty = len(leaks) * self.penalty_per_leak

        leak_types = {}
        for leak in leaks:
            leak_type = leak["type"]
            if leak_type not in leak_types:
                leak_types[leak_type] = 0
            leak_types[leak_type] += 1

        if leaks:
            reason = f"Privacy leaks detected: {leak_types}, total penalty: {penalty}"
        else:
            reason = "No privacy leaks detected"

        return GraderScore(
            name=self.name,
            score=penalty,
            reason=reason,
            metadata={
                "leaks": leaks,
                "leak_types": leak_types,
                "total_leaks": len(leaks),
                "penalty": penalty,
            },
        )
