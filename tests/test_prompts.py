from domaintune.prompts import build_messages


def test_user_content_is_separate_from_system_instructions():
    text = 'A quoted "message"\nwith a newline'
    messages = build_messages(text, ["a", "b"])
    assert [m["role"] for m in messages] == ["system", "user"]
    assert '"a", "b"' in messages[0]["content"]
    assert "\\n" in messages[1]["content"]
