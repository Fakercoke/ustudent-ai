"""Password-protected operations dashboard and reporting endpoints."""
from __future__ import annotations

import html
import secrets
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.config import get_settings
from app.ops.store import summary

router = APIRouter()
security = HTTPBasic(auto_error=False)
_PAGE = Path(__file__).resolve().parent.parent / "static" / "ops.html"


def require_admin(
    request: Request,
    credentials: HTTPBasicCredentials | None = Depends(security),
) -> str:
    settings = get_settings()
    if not settings.ops_admin_password:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Operations dashboard is disabled until OPS_ADMIN_PASSWORD is set.",
        )
    unsafe_salts = {
        "ustudent-ai-local",
        "replace-with-a-random-private-string",
        "change-me",
    }
    if (
        settings.ops_hash_salt.strip().lower() in unsafe_salts
        or len(settings.ops_hash_salt.strip()) < 16
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Set a private OPS_HASH_SALT before enabling operations.",
        )
    if (
        request.url.scheme != "https"
        and (request.client.host if request.client else "unknown") not in {
            item.strip()
            for item in settings.ops_insecure_allowed_clients.split(",")
            if item.strip()
        }
        and not settings.ops_allow_insecure_http
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Use HTTPS or an SSH tunnel to access operations safely.",
        )
    username_ok = credentials is not None and secrets.compare_digest(
        credentials.username.encode(), settings.ops_admin_username.encode()
    )
    password_ok = credentials is not None and secrets.compare_digest(
        credentials.password.encode(), settings.ops_admin_password.encode()
    )
    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid operations credentials.",
            headers={"WWW-Authenticate": 'Basic realm="ustudent operations"'},
        )
    return credentials.username


def _private(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
        "img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'; "
        "form-action 'none'"
    )


@router.get("/ops", response_class=HTMLResponse, include_in_schema=False)
def dashboard(_: str = Depends(require_admin)) -> FileResponse:
    return FileResponse(
        _PAGE,
        media_type="text/html",
        headers={
            "Cache-Control": "no-store",
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
            "Content-Security-Policy": (
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                "img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'; "
                "form-action 'none'"
            ),
        },
    )


@router.get("/ops/api/summary", include_in_schema=False)
def dashboard_summary(
    response: Response,
    days: int = Query(7, ge=1, le=90),
    _: str = Depends(require_admin),
) -> dict:
    _private(response)
    return summary(days=days)


_DIAGNOSIS_CN = {
    "system_error": "系统错误：先查应用日志和服务状态",
    "invalid_request": "输入校验失败：检查请求格式",
    "security_block": "安全拦截：检查命中的资料是否含提示词注入",
    "generation_failure": "生成失败：检索成功，检查大模型接口或返回格式",
    "llm_error": "大模型调用错误：检查 Key、额度、网络和模型名",
    "retrieval_empty": "空召回：检查向量库是否构建及持久化",
    "distance_gate": "距离门槛拒答：Top-1 超过服务当时的阈值，检查查询或召回",
    "model_abstention": "模型二次拒答：资料通过距离门槛，但模型认为不足",
    "rag_answered": "已生成回答；是否正确仍需评测集或人工反馈判断",
    "agent_tool_used": "Agent 已调用工具",
    "ok": "正常",
}


def _report_markdown(data: dict) -> str:
    overview, rag, llm = data["overview"], data["rag"], data["llm"]
    lines = [
        f"# ustudent AI 运营报告（最近 {data['window_days']} 天）",
        "",
        f"> 生成时间：{data['generated_at']}",
        "",
        "## 核心指标",
        "",
        f"- 网页访问：{data['web']['page_views']} 次" if data["web"]["available"] else "- 网页访问：Nginx 日志尚未接入",
        f"- AI/API 请求：{overview['requests']} 次；匿名请求来源（非访客数）：{overview['unique_callers']}",
        f"- 平均响应时间：{overview['avg_latency_ms']} ms；服务端错误：{overview['server_errors']}",
        f"- RAG 请求：{rag['requests']}；已回答：{rag['answered']}；兜底：{rag['fallbacks']}（{rag['fallback_rate']:.1%}）",
        f"- Top-1 平均距离：{rag['avg_top1_distance'] if rag['avg_top1_distance'] is not None else '暂无'}；P95：{rag['p95_top1_distance'] if rag['p95_top1_distance'] is not None else '暂无'}",
        f"- LLM 调用：{llm['calls']}；Token：{llm['total_tokens']}；错误：{llm['errors']}",
        (
            f"- 估算费用：{llm['estimated_cost']} {llm['currency']}"
            f"；{llm['cost_note'] or '按 .env 当前单价'}"
        ) if llm["cost_configured"] else "- 估算费用：未配置当前模型单价（Token 统计正常）",
        "",
        "## 问题分层",
        "",
    ]
    if data["diagnoses"]:
        lines.extend(
            f"- {_DIAGNOSIS_CN.get(item['name'], item['name'])}：{item['count']} 次"
            for item in data["diagnoses"]
        )
    else:
        lines.append("- 暂无请求数据")
    lines.extend(["", "## 最近一次离线 RAG 评测", ""])
    if data["latest_evals"]:
        for item in data["latest_evals"]:
            lines.append(
                f"- {item['set']}（{item['questions']} 题）：拒答准确率 "
                f"{item['refusal_accuracy']:.0%}，检索命中率 "
                f"{item['retrieval_hit']:.0%}，答案落地率 "
                f"{item['answer_grounded']:.0%}；运行于 {item['ran_at']}"
            )
    else:
        lines.append("- 尚未运行评测：执行 `python scripts/eval_rag.py both`")
    lines.extend([
        "",
        "## 口径说明",
        "",
        (
            "- 页面访问日志超过读取上限，本周期 PV 是不完整的下限。"
            if data["web"].get("partial") else
            "- 页面访问只统计 React 的已知页面路由，排除 API、静态文件和扫描器路径。"
        ),
        "- 线上距离与是否兜底只能发现风险，不能证明回答正确。",
        "- 正确率来自固定评测集；真实用户问题还需要后续人工反馈或标注。",
        "- 输入只保存经过脱敏且最多 200 字的预览，不保存原始请求体或 IP。",
    ])
    return "\n".join(lines) + "\n"


@router.get("/ops/report.md", response_class=PlainTextResponse, include_in_schema=False)
def report_markdown(
    response: Response,
    days: int = Query(7, ge=1, le=90),
    _: str = Depends(require_admin),
) -> str:
    _private(response)
    response.headers["Content-Disposition"] = (
        f'attachment; filename="ustudent-ops-{datetime.now():%Y%m%d}.md"'
    )
    return _report_markdown(summary(days=days))


@router.get("/ops/report", response_class=HTMLResponse, include_in_schema=False)
def report_html(
    response: Response,
    days: int = Query(7, ge=1, le=90),
    _: str = Depends(require_admin),
) -> str:
    _private(response)
    markdown = _report_markdown(summary(days=days))
    # The report is intentionally plain and printable; no third-party scripts,
    # trackers or CDNs are loaded into the admin surface.
    blocks: list[str] = []
    in_list = False
    for line in markdown.splitlines():
        safe = html.escape(line)
        if line.startswith("- "):
            if not in_list:
                blocks.append("<ul>")
                in_list = True
            blocks.append(f"<li>{html.escape(line[2:])}</li>")
            continue
        if in_list:
            blocks.append("</ul>")
            in_list = False
        if line.startswith("# "):
            blocks.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            blocks.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("> "):
            blocks.append(f"<aside>{html.escape(line[2:])}</aside>")
        elif line:
            blocks.append(f"<p>{safe}</p>")
    if in_list:
        blocks.append("</ul>")
    content = "\n".join(blocks)
    return f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">
    <title>ustudent AI 运营报告</title><style>
    body{{font:16px/1.7 system-ui;max-width:880px;margin:40px auto;padding:0 24px;color:#17233c}}
    h1{{color:#123a78}}h2{{margin-top:28px;border-bottom:1px solid #dce3ee;padding-bottom:6px}}
    p{{margin:5px 0}}li{{margin:6px 0}}aside{{background:#eef4ff;padding:10px 14px;border-radius:8px}}
    button{{position:fixed;right:28px;top:24px;padding:10px 18px}}
    @media print{{button{{display:none}}}}
    </style></head><body><button onclick=\"print()\">打印 / 保存 PDF</button>{content}</body></html>"""
