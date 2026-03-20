#!/usr/bin/env python3
"""Collect Linux Do daily top topics and build pending translation datasets.

Output:
1. A topic archive file with topic metadata and raw post content.
2. A flattened pending dataset where each post is a translation task with
   empty output, ready for a later translation/review step.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://linux.do"
DEFAULT_OUTPUT_DIR = "temp/pending"
DEFAULT_PERIOD = "daily"
DEFAULT_LIMIT = 10
DEFAULT_BATCH_SIZE = 20
DEFAULT_TIMEOUT = 30
DEFAULT_SLEEP_SECONDS = 1.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://linux.do/top?period=daily",
}

PENDING_INSTRUCTION = (
    "将下面这段来自 Linux Do 社区讨论的中文内容翻译成英文，"
    "保留原有语气、梗、代码、命令、链接和格式。"
)


def default_edge_user_data_dir() -> str:
    system = platform.system()
    home = Path.home()

    candidates: list[Path] = []
    if system == "Darwin":
        candidates.append(home / "Library/Application Support/Microsoft Edge")
    elif system == "Windows":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(Path(local_app_data) / "Microsoft/Edge/User Data")
    else:
        candidates.append(home / ".config/microsoft-edge")

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Linux Do daily top 10 topics via browser automation."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--topics-from-archive",
        default="",
        help=(
            "Optional existing topics archive JSON. If provided, skip loading the "
            "top page and reuse the topic list from that archive."
        ),
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--mode",
        choices=["browser"],
        default="browser",
        help="Linux Do collection runs in browser mode only.",
    )
    parser.add_argument(
        "--browser-channel",
        default="msedge",
        help="Chromium channel for browser mode, for example msedge or chrome.",
    )
    parser.add_argument(
        "--browser-user-data-dir",
        default=os.environ.get("EDGE_USER_DATA_DIR", default_edge_user_data_dir()),
        help=(
            "Optional Chromium user data directory. Use this to launch Edge with an "
            "existing profile so cookies and local storage can be reused."
        ),
    )
    parser.add_argument(
        "--browser-profile-directory",
        default=os.environ.get("EDGE_PROFILE_DIRECTORY", ""),
        help=(
            "Optional profile directory inside the user data dir, for example "
            "'Default' or 'Profile 1'."
        ),
    )
    parser.add_argument(
        "--browser-cdp-url",
        default=os.environ.get("EDGE_CDP_URL", ""),
        help=(
            "Optional CDP endpoint such as http://127.0.0.1:9222. Use this to attach "
            "to a running Edge instance and reuse its live session."
        ),
    )
    parser.add_argument(
        "--browser-headless",
        action="store_true",
        help="Run browser mode in headless mode. Linux Do may require headed mode.",
    )
    parser.add_argument(
        "--max-posts-per-topic",
        type=int,
        default=0,
        help="0 means fetch all posts in a topic.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Small delay between requests to reduce pressure on the forum.",
    )
    return parser.parse_args()


def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def request_json(
    base_url: str,
    path: str,
    params: dict[str, Any] | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    cookie_header: str = "",
) -> dict[str, Any]:
    query = urlencode(params or {}, doseq=True)
    url = f"{base_url.rstrip('/')}{path}"
    if query:
        url = f"{url}?{query}"

    headers = dict(HEADERS)
    if cookie_header:
        headers["Cookie"] = cookie_header

    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} for {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc}") from exc


class BrowserJSONClient:
    def __init__(
        self,
        base_url: str,
        timeout: int,
        channel: str,
        period: str,
        headless: bool,
        user_data_dir: str = "",
        profile_directory: str = "",
        cdp_url: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.channel = channel
        self.period = period
        self.headless = headless
        self.user_data_dir = str(Path(user_data_dir).expanduser()) if user_data_dir else ""
        self.profile_directory = profile_directory
        self.cdp_url = cdp_url
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._connected_over_cdp = False
        self._persistent_context = False
        self._temp_user_data_dir: Path | None = None

    def _profile_snapshot_dir(self) -> str:
        if not self.user_data_dir:
            raise RuntimeError("No browser user data dir is available to clone.")

        source_root = Path(self.user_data_dir)
        if not source_root.exists():
            raise RuntimeError(f"Browser user data dir does not exist: {source_root}")

        profile_directory = self.profile_directory or "Default"
        source_profile = source_root / profile_directory
        if not source_profile.exists():
            raise RuntimeError(f"Browser profile does not exist: {source_profile}")

        temp_root = Path(
            tempfile.mkdtemp(prefix="ldot-edge-profile-", dir="/tmp")
        )
        self._temp_user_data_dir = temp_root

        local_state = source_root / "Local State"
        if local_state.exists():
            shutil.copy2(local_state, temp_root / "Local State")

        target_profile = temp_root / profile_directory
        target_profile.mkdir(parents=True, exist_ok=True)

        names_to_copy = [
            "Cookies",
            "Cookies-journal",
            "Preferences",
            "Secure Preferences",
            "Login Data",
            "Login Data-journal",
            "Local Storage",
            "Session Storage",
            "IndexedDB",
            "WebStorage",
            "Storage",
            "Network",
        ]
        for name in names_to_copy:
            source_path = source_profile / name
            target_path = target_profile / name
            if not source_path.exists():
                continue
            if source_path.is_dir():
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            else:
                shutil.copy2(source_path, target_path)

        return str(temp_root)

    def _launch_persistent_context(self, user_data_dir: str) -> Any:
        launch_args: list[str] = []
        if self.profile_directory:
            launch_args.append(f"--profile-directory={self.profile_directory}")
        return self._playwright.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            channel=self.channel,
            headless=self.headless,
            locale="zh-CN",
            args=launch_args,
        )

    def __enter__(self) -> "BrowserJSONClient":
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Browser mode requires the 'playwright' package. "
                "Install it first, for example: pip install -r requirements-collector.txt"
            ) from exc

        self._playwright = sync_playwright().start()
        if self.cdp_url:
            self._browser = self._playwright.chromium.connect_over_cdp(
                self.cdp_url,
                timeout=self.timeout * 1000,
            )
            self._connected_over_cdp = True
            if self._browser.contexts:
                self._context = self._browser.contexts[0]
            else:
                self._context = self._browser.new_context(locale="zh-CN")
        elif self.user_data_dir:
            try:
                self._context = self._launch_persistent_context(self.user_data_dir)
            except Exception as exc:
                message = str(exc)
                if "ProcessSingleton" not in message:
                    raise
                snapshot_dir = self._profile_snapshot_dir()
                self._context = self._launch_persistent_context(snapshot_dir)
            self._persistent_context = True
        else:
            self._browser = self._playwright.chromium.launch(
                channel=self.channel,
                headless=self.headless,
            )
            self._context = self._browser.new_context(locale="zh-CN")

        if self._context is None:
            raise RuntimeError("Browser context could not be created.")
        self._page = self._context.new_page()
        self._page.goto(
            f"{self.base_url}/top?period={self.period}",
            wait_until="domcontentloaded",
            timeout=self.timeout * 1000,
        )

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            title = self._page.title()
            cookie_names = {
                cookie["name"] for cookie in self._context.cookies(self.base_url)
            }
            if "cf_clearance" in cookie_names or "LINUX DO" in title:
                break
            self._page.wait_for_timeout(1000)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._page is not None:
            self._page.close()
        if self._context is not None and not self._connected_over_cdp:
            self._context.close()
        if self._browser is not None and not self._connected_over_cdp:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        if self._temp_user_data_dir is not None:
            shutil.rmtree(self._temp_user_data_dir, ignore_errors=True)

    def request_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._page is None:
            raise RuntimeError("Browser client is not started.")

        query = urlencode(params or {}, doseq=True)
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"

        result = self._page.evaluate(
            """async (url) => {
                const response = await fetch(url, { credentials: "include" });
                const text = await response.text();
                return { ok: response.ok, status: response.status, text };
            }""",
            url,
        )
        if not result["ok"]:
            raise RuntimeError(f"HTTP {result['status']} for {url}")
        return json.loads(result["text"])

    def top_topics(self, limit: int) -> list[dict[str, Any]]:
        if self._page is None:
            raise RuntimeError("Browser client is not started.")

        self._page.goto(
            f"{self.base_url}/top?period={self.period}",
            wait_until="domcontentloaded",
            timeout=self.timeout * 1000,
        )
        self._page.wait_for_selector(
            "table tbody tr",
            timeout=self.timeout * 1000,
        )
        return self._page.evaluate(
            """(limit) => {
                const rows = Array.from(
                    document.querySelectorAll("table tbody tr")
                )
                    .filter((row) => row.querySelector('a[href^="/t/"]'))
                    .slice(0, limit);

                return rows.map((row, index) => {
                    const titleLink = row.querySelector('a[href^="/t/"]');
                    const href = titleLink?.getAttribute("href") || "";
                    const tags = Array.from(
                        row.querySelectorAll(".discourse-tags .discourse-tag, .discourse-tag-box .discourse-tag")
                    ).map((tag) => tag.textContent?.trim()).filter(Boolean);
                    const categoryLink = row.querySelector(".badge-wrapper, .category-name");
                    const idMatch = href.match(/\\/t\\/(?:[^/]+\\/)?(\\d+)/);

                    return {
                        id: idMatch ? Number(idMatch[1]) : null,
                        title: titleLink?.textContent?.trim() || "",
                        slug: href.split("/").filter(Boolean).slice(-2, -1)[0] || "topic",
                        url: href,
                        tags,
                        category_name: categoryLink?.textContent?.trim() || "",
                        rank: index + 1,
                    };
                }).filter((item) => item.id);
            }""",
            limit,
        )

    def scrape_topic(
        self,
        topic_summary: dict[str, Any],
        max_posts: int = 0,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if self._page is None:
            raise RuntimeError("Browser client is not started.")

        topic_url = topic_summary.get("url") or f"/t/{topic_summary.get('id')}"
        if topic_url.startswith("/"):
            topic_url = f"{self.base_url}{topic_url}"

        self._page.goto(
            topic_url,
            wait_until="domcontentloaded",
            timeout=self.timeout * 1000,
        )
        self._page.wait_for_selector(
            "article[data-post-id]",
            timeout=self.timeout * 1000,
        )
        self._page.wait_for_timeout(1500)

        if max_posts > 0:
            for _ in range(max_posts * 2):
                post_count = self._page.locator("article[data-post-id]").count()
                if post_count >= max_posts:
                    break
                self._page.mouse.wheel(0, 3000)
                self._page.wait_for_timeout(500)

        payload = self._page.evaluate(
            """(maxPosts) => {
                const title =
                    document.querySelector("h1 a.fancy-title, h1 .fancy-title, h1")?.textContent?.trim() || "";
                const tagNodes = Array.from(
                    document.querySelectorAll(".discourse-tags .discourse-tag, .discourse-tag-box .discourse-tag")
                );
                const categoryNode = document.querySelector(".badge-category__name, .category-name");
                const articles = Array.from(document.querySelectorAll("article[data-post-id]"));
                const limited = maxPosts > 0 ? articles.slice(0, maxPosts) : articles;

                const posts = limited.map((article) => {
                    const timeNode = article.querySelector("time");
                    const usernameNode =
                        article.querySelector(".topic-meta-data .username a, .topic-meta-data .username, a[data-user-card]");
                    const cooked = article.querySelector(".cooked");
                    return {
                        id: Number(article.getAttribute("data-post-id")),
                        post_id: Number(article.getAttribute("data-post-id")),
                        post_number: Number(article.getAttribute("data-post-number")),
                        username: usernameNode?.textContent?.trim() || "",
                        name: "",
                        created_at: timeNode?.getAttribute("datetime") || "",
                        updated_at: "",
                        reply_to_post_number: null,
                        reply_count: 0,
                        reads: 0,
                        post_type: 1,
                        raw: cooked?.innerText?.trim() || "",
                    };
                }).filter((post) => post.raw);

                return {
                    detail: {
                        id: window.location.pathname.match(/\\/t\\/(?:[^/]+\\/)?(\\d+)/)?.[1]
                            ? Number(window.location.pathname.match(/\\/t\\/(?:[^/]+\\/)?(\\d+)/)[1])
                            : null,
                        title,
                        fancy_title: title,
                        category_name: categoryNode?.textContent?.trim() || "",
                        tags: tagNodes.map((tag) => tag.textContent?.trim()).filter(Boolean),
                        posts_count: posts.length,
                        visible: true,
                        closed: !!document.querySelector(".topic-status-info .closed"),
                        archived: !!document.querySelector(".topic-status-info .archived"),
                    },
                    posts
                };
            }""",
            max_posts,
        )
        return payload["detail"], payload["posts"]

    def open_topic(self, topic_id: int, slug: str = "topic") -> None:
        if self._page is None:
            raise RuntimeError("Browser client is not started.")

        self._page.goto(
            f"{self.base_url}/t/{slug}/{topic_id}",
            wait_until="domcontentloaded",
            timeout=self.timeout * 1000,
        )
        self._page.wait_for_function(
            """(topicId) => {
                try {
                    const controller = window.Discourse?.__container__?.lookup("controller:topic");
                    return controller?.model?.id === topicId;
                } catch (error) {
                    return false;
                }
            }""",
            arg=topic_id,
            timeout=self.timeout * 1000,
        )

    def topic_metadata(self) -> dict[str, Any]:
        if self._page is None:
            raise RuntimeError("Browser client is not started.")

        return self._page.evaluate(
            """() => {
                const controller = window.Discourse.__container__.lookup("controller:topic");
                const model = controller.model;
                const postStream = model.postStream || model.post_stream;
                return {
                    id: model.id,
                    title: model.title,
                    fancy_title: model.fancy_title,
                    category_id: model.category_id,
                    tags: model.tags || [],
                    created_at: model.created_at,
                    last_posted_at: model.last_posted_at,
                    views: model.views,
                    like_count: model.like_count,
                    reply_count: model.reply_count,
                    posts_count: model.posts_count,
                    visible: model.visible,
                    closed: model.closed,
                    archived: model.archived,
                    stream_length: postStream.stream.length
                };
            }"""
        )

    def topic_posts(self, post_numbers: list[int]) -> list[dict[str, Any]]:
        if self._page is None:
            raise RuntimeError("Browser client is not started.")
        if not post_numbers:
            return []

        return self._page.evaluate(
            """async (numbers) => {
                const controller = window.Discourse.__container__.lookup("controller:topic");
                const model = controller.model;
                const postStream = model.postStream || model.post_stream;
                const loaded = await Promise.allSettled(
                    numbers.map((n) => postStream.loadPostByPostNumber(n))
                );

                return loaded
                    .filter((item) => item.status === "fulfilled" && item.value)
                    .map((item) => {
                        const post = item.value;
                        return {
                            id: post.id,
                            post_type: post.post_type,
                            post_number: post.post_number,
                            username: post.username,
                            name: post.name,
                            created_at: post.created_at,
                            updated_at: post.updated_at,
                            reply_to_post_number: post.reply_to_post_number,
                            reply_count: post.reply_count,
                            reads: post.reads,
                            raw: post.raw || "",
                        };
                    });
            }""",
            post_numbers,
        )


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def chunked(values: list[int], size: int) -> list[list[int]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def fetch_top_topics(
    fetch_json: Callable[..., dict[str, Any]],
    period: str,
    limit: int,
) -> list[dict[str, Any]]:
    payload = fetch_json("/top.json", params={"period": period})
    topics = payload.get("topic_list", {}).get("topics", [])
    return topics[:limit]


def fetch_topic(
    fetch_json: Callable[..., dict[str, Any]],
    topic_id: int,
) -> dict[str, Any]:
    return fetch_json(f"/t/{topic_id}.json")


def fetch_topic_posts(
    fetch_json: Callable[..., dict[str, Any]],
    topic_id: int,
    post_ids: list[int],
) -> list[dict[str, Any]]:
    payload = fetch_json(
        f"/t/{topic_id}/posts.json",
        params={"include_raw": "true", "post_ids[]": post_ids},
    )
    return payload.get("post_stream", {}).get("posts", [])


def build_topic_archive(
    base_url: str,
    rank: int,
    topic_summary: dict[str, Any],
    topic_detail: dict[str, Any],
    posts: list[dict[str, Any]],
) -> dict[str, Any]:
    cleaned_posts: list[dict[str, Any]] = []
    for post in posts:
        raw = normalize_text(post.get("raw"))
        if not raw:
            continue
        if post.get("post_type") != 1:
            continue
        cleaned_posts.append(
            {
                "post_id": post.get("id"),
                "post_number": post.get("post_number"),
                "username": post.get("username"),
                "name": post.get("name"),
                "created_at": post.get("created_at"),
                "updated_at": post.get("updated_at"),
                "reply_to_post_number": post.get("reply_to_post_number"),
                "reply_count": post.get("reply_count"),
                "reads": post.get("reads"),
                "raw": raw,
            }
        )

    return {
        "rank": rank,
        "topic_id": topic_summary.get("id"),
        "title": topic_detail.get("title") or topic_summary.get("title"),
        "fancy_title": topic_detail.get("fancy_title"),
        "slug": topic_summary.get("slug"),
        "url": (
            f"{base_url.rstrip('/')}{topic_summary.get('url')}"
            if str(topic_summary.get("url", "")).startswith("/")
            else topic_summary.get("url") or f"{base_url.rstrip('/')}/t/{topic_summary.get('id')}"
        ),
        "category_id": topic_summary.get("category_id") or topic_detail.get("category_name"),
        "tags": topic_detail.get("tags") or topic_summary.get("tags", []),
        "created_at": topic_summary.get("created_at"),
        "last_posted_at": topic_summary.get("last_posted_at"),
        "views": topic_summary.get("views"),
        "like_count": topic_summary.get("like_count"),
        "reply_count": topic_summary.get("reply_count"),
        "posts_count": topic_summary.get("posts_count"),
        "visible": topic_detail.get("visible"),
        "closed": topic_detail.get("closed"),
        "archived": topic_detail.get("archived"),
        "posts": cleaned_posts,
    }


def build_pending_items(
    period: str,
    capture_date: str,
    topic_archive: dict[str, Any],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for post in topic_archive["posts"]:
        items.append(
            {
                "instruction": PENDING_INSTRUCTION,
                "input": post["raw"],
                "output": "",
                "meta": {
                    "source": "linux.do",
                    "period": period,
                    "capture_date": capture_date,
                    "topic_id": topic_archive["topic_id"],
                    "topic_title": topic_archive["title"],
                    "topic_url": topic_archive["url"],
                    "topic_rank": topic_archive["rank"],
                    "category_id": topic_archive["category_id"],
                    "tags": topic_archive["tags"],
                    "post_id": post["post_id"],
                    "post_number": post["post_number"],
                    "username": post["username"],
                    "created_at": post["created_at"],
                    "reply_to_post_number": post["reply_to_post_number"],
                    "translation_status": "pending",
                },
            }
        )
    return items


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_topics_from_archive(path: str, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    topics = payload.get("topics", [])
    loaded: list[dict[str, Any]] = []
    for topic in topics[:limit]:
        loaded.append(
            {
                "id": topic.get("topic_id") or topic.get("id"),
                "title": topic.get("title"),
                "slug": topic.get("slug") or "topic",
                "url": topic.get("url"),
                "tags": topic.get("tags", []),
                "category_id": topic.get("category_id"),
                "rank": topic.get("rank"),
            }
        )
    return [topic for topic in loaded if topic.get("id")]


def main() -> int:
    args = parse_args()
    capture_date = datetime.now().date().isoformat()
    captured_at = now_utc_iso()

    browser_client: BrowserJSONClient | None = None
    fetch_json: Callable[..., dict[str, Any]]

    try:
        browser_client = BrowserJSONClient(
            args.base_url,
            args.timeout,
            args.browser_channel,
            args.period,
            args.browser_headless,
            args.browser_user_data_dir,
            args.browser_profile_directory,
            args.browser_cdp_url,
        ).__enter__()
        fetch_json = browser_client.request_json

        if args.topics_from_archive:
            top_topics = load_topics_from_archive(args.topics_from_archive, args.limit)
        else:
            top_topics = browser_client.top_topics(args.limit)

        topic_archives: list[dict[str, Any]] = []
        pending_items: list[dict[str, Any]] = []

        for index, topic_summary in enumerate(top_topics, start=1):
            topic_id = topic_summary["id"]
            fetched_posts: list[dict[str, Any]] = []
            topic_detail, fetched_posts = browser_client.scrape_topic(
                topic_summary,
                args.max_posts_per_topic,
            )
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

            topic_archive = build_topic_archive(
                args.base_url,
                index,
                topic_summary,
                topic_detail,
                fetched_posts,
            )
            topic_archives.append(topic_archive)
            pending_items.extend(
                build_pending_items(args.period, capture_date, topic_archive)
            )

            print(
                f"[{index}/{len(top_topics)}] topic_id={topic_id} "
                f"posts={len(topic_archive['posts'])} "
                f"title={topic_archive['title']}",
                file=sys.stderr,
            )

        stem = f"linuxdo-top{args.limit}-{args.period}-{capture_date}"
        output_dir = Path(args.output_dir)
        archive_path = output_dir / f"{stem}.topics.json"
        pending_path = output_dir / f"{stem}.pending.json"

        archive_payload = {
            "source": "linux.do",
            "period": args.period,
            "capture_date": capture_date,
            "captured_at": captured_at,
            "topic_limit": args.limit,
            "topic_count": len(topic_archives),
            "pending_item_count": len(pending_items),
            "topics": topic_archives,
        }

        write_json(archive_path, archive_payload)
        write_json(pending_path, pending_items)

        print(
            json.dumps(
                {
                    "archive_path": str(archive_path),
                    "pending_path": str(pending_path),
                    "topic_count": len(topic_archives),
                    "pending_item_count": len(pending_items),
                    "captured_at": captured_at,
                },
                ensure_ascii=False,
            )
        )
        return 0
    finally:
        if browser_client is not None:
            browser_client.__exit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(main())
