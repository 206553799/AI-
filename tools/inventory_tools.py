"""
库存管理工具 — 封装对 Java 后端 /api/inventory 的 HTTP 调用。

每个函数使用 @tool 装饰器注册为 LangChain Tool，
Agent 可以根据用户意图自动选择和调用。
"""

import httpx
from langchain_core.tools import tool
from config import config

BASE = config.BACKEND_BASE_URL


def _get(path: str, params: dict | None = None) -> dict:
    """内部 GET 请求"""
    try:
        r = httpx.get(f"{BASE}{path}", params=params, timeout=10.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        return {"code": 500, "message": f"请求失败: {e}", "data": None}


def _post(path: str, body: dict) -> dict:
    """内部 POST 请求"""
    try:
        r = httpx.post(f"{BASE}{path}", json=body, timeout=10.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        return {"code": 500, "message": f"请求失败: {e}", "data": None}


def _put(path: str, body: dict) -> dict:
    """内部 PUT 请求"""
    try:
        r = httpx.put(f"{BASE}{path}", json=body, timeout=10.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        return {"code": 500, "message": f"请求失败: {e}", "data": None}


def _delete(path: str) -> dict:
    """内部 DELETE 请求"""
    try:
        r = httpx.delete(f"{BASE}{path}", timeout=10.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        return {"code": 500, "message": f"请求失败: {e}", "data": None}


# ==================== Tools ====================

@tool
def list_inventory(keyword: str = "") -> str:
    """
    查询库存列表。可以传入 keyword 关键字按商品名称模糊搜索。
    如果不传 keyword 则返回全部库存。

    参数:
        keyword: 搜索关键字（可选），按商品名称模糊匹配

    返回: JSON 格式的库存列表
    """
    params = {}
    if keyword and keyword.strip():
        params["keyword"] = keyword.strip()
    result = _get("/api/inventory", params=params)
    return str(result)


@tool
def get_inventory_by_id(inventory_id: int) -> str:
    """
    根据 ID 查询单个库存商品的详细信息。

    参数:
        inventory_id: 库存商品 ID

    返回: JSON 格式的库存详情
    """
    result = _get(f"/api/inventory/{inventory_id}")
    return str(result)


@tool
def get_inventory_by_sku(sku: str) -> str:
    """
    根据 SKU 编码查询库存商品。

    参数:
        sku: 商品 SKU 编码，如 "SKU001"

    返回: JSON 格式的库存详情
    """
    result = _get(f"/api/inventory/sku/{sku}")
    return str(result)


@tool
def create_inventory(sku: str, name: str, price: float,
                     quantity: int = 0, description: str = "",
                     category: str = "") -> str:
    """
    新增一个库存商品到系统中。

    参数:
        sku: SKU 编码（唯一标识）
        name: 商品名称
        price: 单价（元）
        quantity: 初始库存数量（默认 0）
        description: 商品描述（可选）
        category: 商品分类（可选）

    返回: 创建结果
    """
    body = {
        "sku": sku,
        "name": name,
        "price": price,
        "quantity": quantity,
        "description": description,
        "category": category,
    }
    result = _post("/api/inventory", body)
    return str(result)


@tool
def update_inventory(inventory_id: int, name: str = "",
                     price: float = 0.0, quantity: int = -1,
                     description: str = "", category: str = "") -> str:
    """
    更新库存商品信息。只需传入要修改的字段，未传入的字段不会被修改。

    参数:
        inventory_id: 库存商品 ID
        name: 新的商品名称（可选）
        price: 新的单价（可选，传 0 表示不修改）
        quantity: 新的库存数量（可选，传 -1 表示不修改）
        description: 新的描述（可选）
        category: 新的分类（可选）

    返回: 更新结果
    """
    body = {}
    if name:
        body["name"] = name
    if price > 0:
        body["price"] = price
    if quantity >= 0:
        body["quantity"] = quantity
    if description:
        body["description"] = description
    if category:
        body["category"] = category
    result = _put(f"/api/inventory/{inventory_id}", body)
    return str(result)


@tool
def delete_inventory(inventory_id: int) -> str:
    """
    删除一个库存商品。

    参数:
        inventory_id: 要删除的库存商品 ID

    返回: 删除结果
    """
    result = _delete(f"/api/inventory/{inventory_id}")
    return str(result)


# 工具列表
INVENTORY_TOOLS = [
    list_inventory,
    get_inventory_by_id,
    get_inventory_by_sku,
    create_inventory,
    update_inventory,
    delete_inventory,
]
