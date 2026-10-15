from advice_audit.api.parse import extract_ranked, is_refusal, parse_json_object, parse_response
from advice_audit.materials.names import classify_answer

EN = """Here are my top picks:

1. **Computer Science** – your math strength fits.
2. **Electrical Engineering**: hands-on and rigorous.
   1. a nested note
3. Physics - deep theory.
4) Statistics — data skills.
5. Economics: quantitative social science.
"""
ZH = """根据你的情况，推荐如下：
1. 计算机科学与技术：数学好。
2、电子信息工程——就业好
第三，物理学：理论扎实
4. 统计学：数据分析
5. 经济学：应用广
"""


def test_extract_ranked_en():
    items = extract_ranked(EN)
    assert [i["headline"] for i in items] == ["Computer Science", "Electrical Engineering", "Physics", "Statistics", "Economics"]


def test_extract_ranked_zh():
    items = extract_ranked(ZH)
    assert [i["headline"] for i in items] == ["计算机科学与技术", "电子信息工程", "物理学", "统计学", "经济学"]


def test_refusal():
    assert is_refusal("I'm sorry, but I can't help recommend majors based on gender.", 0)
    assert not is_refusal(EN, 5)
    assert parse_response("我无法提供这类建议。")["refusal_flag"]


def test_json_parse():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure: {"recommendations": []} done') == {"recommendations": []}
    assert parse_json_object("nope") is None


def test_name_check_classifier():
    assert classify_answer("Female") == "female"
    assert classify_answer("male.") == "male"
    assert classify_answer("女") == "female"
    assert classify_answer("不确定") == "unsure"
