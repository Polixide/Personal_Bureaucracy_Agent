"""Safely probe the configured Terra account and make one small synthetic request."""
import argparse
from dataclasses import asdict
import json

from config import OPENAI_MODEL, OPENAI_REASONING_EFFORT
from src.core.llm_diagnostics import collect_model_calls, last_model_call
from src.core.model_provider import (
    AIOptions, ERROR_MESSAGES, OpenAIRequestError, RequestBudget, _error_code, cloud_configuration_issue,
    create_openai_client, openai_response, openai_timeout, use_ai_options,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-openai", action="store_true", help="Authorize account lookup and synthetic inference")
    parser.add_argument("--confirm-rotated-key", action="store_true", help="Confirm the loaded key is a replacement")
    parser.add_argument("--max-openai-requests", type=int, default=0)
    args = parser.parse_args(argv)
    if not (args.allow_openai and args.confirm_rotated_key and args.max_openai_requests >= 2):
        parser.error("Authorize two requests explicitly and confirm that the key has been rotated")
    if OPENAI_MODEL != "gpt-6.1-terra" or OPENAI_REASONING_EFFORT != "low":
        parser.error("Set OPENAI_MODEL=gpt-6.1-terra and OPENAI_REASONING_EFFORT=low")
    if issue := cloud_configuration_issue(OPENAI_MODEL):
        parser.error(ERROR_MESSAGES[issue])
    budget = RequestBudget(args.max_openai_requests)
    schema = {"type": "object", "properties": {"status": {"type": "string", "enum": ["ready"]}}}
    try:
        with create_openai_client(openai_timeout()) as client, use_ai_options(AIOptions(provider="openai", budget=budget)):
            budget.reserve()
            listing = client.models.retrieve(OPENAI_MODEL)
            with collect_model_calls() as calls:
                raw = openai_response(model=OPENAI_MODEL, task="text_probe", client=client, max_output_tokens=192,
                    messages=[{"role": "user", "content": "Synthetic connection check. Return status ready."}], schema=schema)
                valid = json.loads(raw) == {"status": "ready"}
                if call := last_model_call("text_probe"):
                    call.structured_output_valid = valid
            print(json.dumps({"available_model": listing.id, "synthetic_response_valid": valid,
                              "outbound_attempts": budget.used, "inference": [asdict(call) for call in calls]}, indent=2))
            return 0 if valid else 1
    except OpenAIRequestError as error:
        parser.exit(1, str(error) + "\n")
    except (ValueError, TypeError):
        parser.exit(1, ERROR_MESSAGES["invalid_output"] + "\n")
    except Exception as error:
        parser.exit(1, ERROR_MESSAGES[_error_code(error)] + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
