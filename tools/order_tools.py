"""
订单管理工具 — 封装对 Java 后端 /api/orders 的 HTTP 调用。
"""

import httpx
from langchain_core.tools import tool
from config import config

BASE = config.BACKEND_BASE_URL


def _get(path: str, params: dict | None = None) -> dict:
    try:
        r = httpx.get(f"{BASE}{path}", params=params, timeout=10.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        return {"code": 500, "message": f"请求失败: {e}", "data": None}


def _post(path: str, body: dict) -> dict:
    try:
        r = httpx.post(f"{BASE}{path}", json=body, timeout=10.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        return {"code": 500, "message": f"请求失败: {e}", "data": None}


def _patch(path: str) -> dict:
    try:
        r = httpx.patch(f"{BASE}{path}", timeout=10.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        return {"code": 500, "message": f"请求失败: {e}", "data": None}


def _delete(path: str) -> dict:
    try:
        r = httpx.delete(f"{BASE}{path}", timeout=10.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        return {"code": 500, "message": f"请求失败: {e}", "data": None}


# ==================== Tools ====================

@tool
def list_orders(customer_name: str = "", status: str = "") -> str:
    """
    查询订单列表。支持按客户名称或订单状态筛选。

    参数:
        customer_name: 客户名称（可选），按客户名称筛选
        status: 订单状态（可选），可选值: PENDING / CONFIRMED / CANCELLED / COMPLETED

    返回: JSON 格式的订单列表
    """
    params = {}
    if customer_name and customer_name.strip():
        params["customerName"] = customer_name.strip()
    if status and status.strip():
        params["status"] = status.strip().upper()
    result = _get("/api/orders", params=params)
    return str(result)


@tool
def get_order_by_id(order_id: int) -> str:
    """
    根据 ID 查询订单详情，包括所有订单项。

    参数:
        order_id: 订单 ID

    返回: JSON 格式的订单详情
    """
    result = _get(f"/api/orders/{order_id}")
    return str(result)


@tool
def get_order_by_no(order_no: str) -> str:
    """
    根据订单号查询订单详情。

    参数:
        order_no: 订单号，如 "ORD-20260710-0001"

    返回: JSON 格式的订单详情
    """
    result = _get(f"/api/orders/no/{order_no}")
    return str(result)


@tool
def create_order(customer_name: str, items: str) -> str:
    """
    创建一个新订单，下单时会自动校验并扣减库存。
    注意：需要先查询库存确认商品 ID 和库存充足。

    参数:
        customer_name: 客户名称
        items: 订单项列表，JSON 数组格式的字符串。
               每个订单项包含 inventoryId (商品ID) 和 quantity (数量)。
               例如: '[{"inventoryId": 1, "quantity": 2}, {"inventoryId": 3, "quantity": 1}]'

    返回: 创建结果，包含订单号和总金额
    """
    import json
    try:
        items_list = json.loads(items) if isinstance(items, str) else items
    except json.JSONDecodeError:
        return str({"code": 400, "message": "items 格式错误，请传入 JSON 数组字符串", "data": None})

    body = {
        "customerName": customer_name,
        "items": items_list,
    }
    result = _post("/api/orders", body)
    return str(result)


@tool
def cancel_order(order_id: int) -> str:
    """
    取消订单。已取消的订单不能重复取消，已完成的订单不能取消。
    取消后系统会自动将库存还原。

    参数:
        order_id: 要取消的订单 ID

    返回: 取消结果
    """
    result = _patch(f"/api/orders/{order_id}/cancel")
    return str(result)


@tool
def complete_order(order_id: int) -> str:
    """
    将订单标记为"已完成"。

    参数:
        order_id: 要完成的订单 ID

    返回: 完成结果
    """
    result = _patch(f"/api/orders/{order_id}/complete")
    return str(result)


@tool
def delete_order(order_id: int) -> str:
    """
    删除订单。如果订单处于"已确认"状态，删除前会自动还原库存。

    参数:
        order_id: 要删除的订单 ID

    返回: 删除结果
    """
    result = _delete(f"/api/orders/{order_id}")
    return str(result)


ORDER_TOOLS = [
    list_orders,
    get_order_by_id,
    get_order_by_no,
    create_order,
    cancel_order,
    complete_order,
    delete_order,
]
