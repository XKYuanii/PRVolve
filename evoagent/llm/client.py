"""Small OpenAI-compatible JSON client with auditable usage accounting."""
import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from ..session.ledger import ExecutionLedger


def _decode_json_object(candidate: str) -> Tuple[Dict[str, Any], int]:
    """Decode one object and tolerate only concatenated, valid JSON values.

    Some compatible providers occasionally append a second JSON object despite
    structured-output instructions. The first complete object is the role
    action. Arbitrary trailing prose remains an error.
    """
    try:
        result = json.loads(candidate)
        trailing_values = 0
    except json.JSONDecodeError as exc:
        if "Extra data" not in str(exc):
            raise
        decoder = json.JSONDecoder()
        result, offset = decoder.raw_decode(candidate)
        remainder = candidate[offset:].strip()
        trailing_values = 0
        while remainder:
            _ignored, offset = decoder.raw_decode(remainder)
            trailing_values += 1
            remainder = remainder[offset:].strip()
    if not isinstance(result, dict):
        raise ValueError("model JSON root is not an object")
    return result, trailing_values


class JsonChatClient:
    def __init__(
        self, base_url: str, api_key: str, model: str,
        provider: str = "openai-compatible", timeout: int = 60,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})

    def complete_json(
        self, role: str, system: str, user: str,
        ledger: Optional[ExecutionLedger] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        # DeepSeek v4 can spend the entire completion allowance on hidden
        # reasoning and return an empty content field. EvoAgent's role
        # protocol requires a JSON object in content, so disable thinking for
        # this structured transport.
        if self.provider == "deepseek":
            payload["thinking"] = {"type": "disabled"}
        elif "api.siliconflow.cn" in self.base_url.lower():
            payload["enable_thinking"] = False
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(self.extra_headers)
        max_attempts = 3
        structured_retry_used = False
        for attempt in range(max_attempts):
            payload["messages"] = messages
            request = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers, method="POST",
            )
            started = time.monotonic()
            body: Dict[str, Any] = {}
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = str(body["choices"][0]["message"]["content"])
            except urllib.error.HTTPError as exc:
                detail = exc.read(1000).decode("utf-8", errors="replace")
                message = "%s API returned HTTP %d: %s" % (
                    self.provider, exc.code, detail,
                )
                retryable = exc.code in {408, 409, 429, 500, 502, 503, 504}
            except (urllib.error.URLError, socket.timeout, ValueError, KeyError,
                    IndexError, TypeError, json.JSONDecodeError) as exc:
                message = "%s JSON request failed: %s" % (self.provider, exc)
                retryable = True
            else:
                candidate = content.strip()
                if candidate.startswith("```json") and candidate.endswith("```"):
                    candidate = candidate[7:-3].strip()
                elif candidate.startswith("```") and candidate.endswith("```"):
                    candidate = candidate[3:-3].strip()
                try:
                    result, trailing_values = _decode_json_object(candidate)
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    message = "%s JSON request failed: %s" % (self.provider, exc)
                    if ledger:
                        ledger.record_model(
                            role, self.provider, self.model, body.get("usage") or {},
                            int((time.monotonic() - started) * 1000), False,
                            message + " (structured-output retry available=%s)"
                            % (not structured_retry_used),
                        )
                    if not structured_retry_used and attempt < max_attempts - 1:
                        structured_retry_used = True
                        if body["choices"][0].get("finish_reason") == "length":
                            # A truncated object has lost meaning that syntax repair
                            # cannot recover. Regenerate from the task once, briefly,
                            # using the same token allowance and total attempt cap.
                            messages = [
                                {"role": "system", "content": system},
                                {"role": "user", "content": user + (
                                    "\nThe previous response exceeded the output limit. "
                                    "Return a compact complete JSON action. State each fact "
                                    "once; retain every required decision and evidence field."
                                )},
                            ]
                        else:
                            messages = [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a strict JSON syntax repair engine. Preserve the "
                                        "input object's keys, values and meaning. Repair syntax only "
                                        "and return one JSON object with no Markdown or explanation."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": "Repair this malformed JSON object:\n" + content[:64000],
                                },
                            ]
                        continue
                    raise RuntimeError(message)
                if ledger and trailing_values:
                    ledger.trace(
                        role, "structured_json_extra_values_ignored",
                        trailing_values=trailing_values,
                    )
                if ledger:
                    ledger.record_model(
                        role, self.provider, self.model, body.get("usage") or {},
                        int((time.monotonic() - started) * 1000), True,
                    )
                return result
            if ledger:
                ledger.record_model(
                    role, self.provider, self.model, body.get("usage") or {},
                    int((time.monotonic() - started) * 1000), False, message,
                )
            if retryable and attempt < max_attempts - 1:
                time.sleep(min(0.25 * (2 ** attempt), 1.0))
                continue
            raise RuntimeError(message)
        raise RuntimeError("%s JSON request failed" % self.provider)
