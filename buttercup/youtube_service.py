"""
YouTube-specific I/O helpers.

Deliberately framework-agnostic: no Streamlit imports here, so these
functions can be unit tested or reused outside the app (CLI, API, etc.).
"""
from __future__ import annotations

import json
import urllib.request
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

# Try these languages in order before falling back to whatever is available.
PREFERRED_TRANSCRIPT_LANGUAGES = ["en", "hi", "bn", "ar", "zh", "fr", "de", "es"]


def extract_video_id(url: str) -> Optional[str]:
    """Pull the 11-char video ID out of a youtube.com or youtu.be URL.

    Handles standard watch URLs, youtu.be short links, and the
    /shorts/, /embed/, /v/, /live/ path formats (with or without a
    "www."/"m." subdomain, and on youtube-nocookie.com).
    """
    if not url:
        return None
    parsed = urlparse(url)
    hostname = str(parsed.hostname or "")

    if hostname == "youtu.be":
        video_id = parsed.path.lstrip("/")[:11]
        return video_id or None

    if hostname.endswith("youtube.com") or hostname.endswith("youtube-nocookie.com"):
        # Standard watch URL: /watch?v=VIDEO_ID
        video_id = parse_qs(parsed.query).get("v", [None])[0]
        if video_id:
            return video_id[:11]

        # Path-based formats: /shorts/VIDEO_ID, /embed/VIDEO_ID, /v/VIDEO_ID, /live/VIDEO_ID
        path_parts = [p for p in parsed.path.split("/") if p]
        for prefix in ("shorts", "embed", "v", "live"):
            if len(path_parts) >= 2 and path_parts[0] == prefix:
                return path_parts[1][:11] or None

        return None

    return None


def get_video_meta(video_id: str) -> Tuple[str, str]:
    """Fetch (title, channel_name) via YouTube's oEmbed endpoint. Best-effort."""
    try:
        url = (
            "https://www.youtube.com/oembed?url="
            f"https://www.youtube.com/watch?v={video_id}&format=json"
        )
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read())
            return data.get("title", "YouTube Video"), data.get("author_name", "Unknown")
    except Exception:
        return "YouTube Video", "Unknown"


def fetch_transcript(video_id: Optional[str]) -> Tuple[List, str]:
    """
    Fetch the best-available transcript for a video.

    Returns (transcript_snippets, language_code). On failure, transcript_snippets
    is an empty list and language_code is one of "unknown" | "disabled" | "error"
    so callers can show a specific message.
    """
    if not video_id:
        return [], "unknown"

    try:
        api = YouTubeTranscriptApi()
        listing = api.list(video_id)

        for lang in PREFERRED_TRANSCRIPT_LANGUAGES:
            try:
                return listing.find_transcript([lang]).fetch(), lang
            except NoTranscriptFound:
                continue

        # Nothing in our preferred list - take whatever the video actually has.
        for transcript in listing:
            try:
                return transcript.fetch(), transcript.language_code
            except Exception:
                continue

        return [], "unknown"
    except TranscriptsDisabled:
        return [], "disabled"
    except Exception:
        return [], "error"
