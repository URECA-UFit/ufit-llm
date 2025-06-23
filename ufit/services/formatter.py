import json

def generate_final_output(plan):
    name = plan.get("plan_name", "")
    summary = plan.get("summary", "")
    monthly_fee = plan.get("monthly_fee", "")
    discount_fee = plan.get("discount_fee", "")
    data_allowance = plan.get("data_allowance", "")
    voice_allowance = plan.get("voice_allowance", "")
    sms_allowance = plan.get("sms_allowance", "")
    data_sharing = plan.get("data_sharing", "")
    extra_data = plan.get("extra_data", "")
    data_category = plan.get("data_category", "")
    basic_benefit = plan.get("basic_benefit", {})
    discount_benefit = plan.get("discount_benefit", {})
    special_benefit = plan.get("special_benefit", {})
    device_type = plan.get("device_type", "")
    social_category = plan.get("social_category", "")

    lines = []

    if name and summary:
        lines.append(f"{name} 요금제는 {summary}로,")
    
    if monthly_fee and discount_fee:
        lines.append(f"정가는 {monthly_fee:,}원이며 약정 시 {discount_fee:,}원에 이용할 수 있습니다.")
    elif monthly_fee:
        lines.append(f"요금제는 {monthly_fee:,}원입니다.")
    elif discount_fee:
        lines.append(f"약정 시 {discount_fee:,}원에 이용할 수 있습니다.")
    
    if data_allowance:
        data_line = f"{data_allowance} 데이터를 제공합니다."
        if extra_data:
            data_line += f" 추가로 {extra_data}도 포함됩니다."
        lines.append(data_line)
    
    if voice_allowance or sms_allowance:
        parts = []
        if voice_allowance:
            parts.append(f"통화는 {voice_allowance}")
        if sms_allowance:
            parts.append(f"문자는 {sms_allowance}")
        lines.append(" / ".join(parts) + "이 포함됩니다.")
    
    if data_sharing:
        lines.append(f"데이터 공유는 {data_sharing}입니다.")
    
    if data_category:
        lines.append(f"데이터 카테고리는 {data_category}입니다.")
    
    if device_type:
        lines.append(f"디바이스 타입은 {device_type}입니다.")
    
    if social_category:
        lines.append(f"사회적 카테고리는 {social_category}입니다.")
    
    if basic_benefit and isinstance(basic_benefit, dict):
        basic_text = ", ".join([f"{k}: {v}" for k, v in basic_benefit.items()])
        if basic_text:
            lines.append(f"기본 혜택으로는 {basic_text}이 제공됩니다.")
    
    if discount_benefit and isinstance(discount_benefit, dict):
        discount_text = ", ".join([f"{k}: {v}" for k, v in discount_benefit.items()])
        if discount_text:
            lines.append(f"할인 혜택으로는 {discount_text}이 있습니다.")
    
    if special_benefit and isinstance(special_benefit, dict):
        special_text = ", ".join([f"{k}: {v}" for k, v in special_benefit.items()])
        if special_text:
            lines.append(f"추가 혜택으로는 {special_text}도 받을 수 있습니다.")

    return "\n".join(lines)
