import json

# 3. 연령대별 인기 변환
def format_age_popularity(age_data_str):
    try:
        pop_dict = eval(age_data_str) if isinstance(age_data_str, str) else age_data_str
        return " / ".join([f"{k.replace('s', '대').replace('plus', '대 이상')}에서 {v}번째로 인기 있는 요금제" for k, v in pop_dict.items()])
    except:
        return ""

def generate_final_output(plan):
    name = plan.get("plan_name", "")
    summary = plan.get("summary", "")
    price = plan.get("price", "")
    discount_price = plan.get("discount_price", "")
    data = plan.get("data", "")
    preferred_channel = plan.get("선호하는 통신방법", "")
    call = plan.get("call", "")
    sms = plan.get("sms", "")
    sharing = plan.get("data_sharing", "")
    usage = plan.get("데이터로 주로하는 작업", "")
    basic = plan.get("basic_benefits", "")
    discount = plan.get("discount_benefits", "")
    special = plan.get("special_benefits", "")
    age_rank = format_age_popularity(plan.get("연령대별 인기", ""))
    update = plan.get("update_at", "")
    enabled = plan.get("is_enabled", "")

    lines = []

    if name and summary:
        lines.append(f"{name} 요금제는 {summary}로,")
    if price and discount_price:
        lines.append(f"정가는 {price}원이며 약정 시 {discount_price}원에 이용할 수 있습니다.")
    elif price:
        lines.append(f"요금제는 {price}원입니다.")
    elif discount_price:
        lines.append(f"약정 시 {discount_price}원에 이용할 수 있습니다.")
    if data:
        data_line = f"{data} 데이터를 제공합니다."
        if preferred_channel:
            data_line += f" 이 요금제는 {preferred_channel}을 선호하는 사용자에게 적합합니다."
        lines.append(data_line)
    if call or sms:
        parts = []
        if call:
            parts.append(f"통화는 {call}")
        if sms:
            parts.append(f"문자는 {sms}")
        lines.append(" / ".join(parts) + "이 포함됩니다.")
    if sharing:
        lines.append(f"데이터 공유는 {sharing}입니다.")
    if usage:
        lines.append(f"주 사용 용도는 {usage}입니다.")
    if basic:
        lines.append(f"기본 혜택으로는 {basic}이 제공됩니다.")
    if discount:
        lines.append(f"할인 혜택으로는 {discount}이 있습니다.")
    if special:
        lines.append(f"추가 혜택으로는 {special}도 받을 수 있습니다.")
    if age_rank:
        lines.append(f"{age_rank}입니다.")
    if enabled == "TRUE":
        lines.append("현재 판매 중인 요금제입니다.")
    if update:
        lines.append(f"(마지막 업데이트: {update})")

    return "\n".join(lines)
