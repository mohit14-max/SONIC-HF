from __future__ import annotations

import argparse
import base64
import json
import html
import logging
import os
import re
import socket
from collections.abc import Mapping
from pathlib import Path

try:
    import gradio as gr
except ImportError:  # pragma: no cover - handled at runtime
    gr = None  # type: ignore[assignment]

from config import (
    APP_NAME,
    DEFAULT_RUNTIME_PREFERENCE,
    SONIC_HOST,
    SONIC_PORT,
    SONIC_SHARE,
    VOICE_OUTPUT_DEFAULT,
)
from core.conversations import (
    UNTITLED_CHAT_TITLE,
    append_exchange,
    clear_conversation,
    conversation_messages_to_backend_history,
    create_conversation,
    delete_conversation,
    ensure_active_conversation,
    list_conversations,
    set_active_conversation,
)
from core.image_generation import ImageGenerationError, generate_sonic_image
from core.live_web import sanitize_live_web_output
from core.providers import ask_sonic
from core.tts import build_speech_payload, encode_speech_text


# Modes list removed
RUNTIME_CHOICES = [
    ("Offline", "offline"),
    ("Online", "online"),
]
WORKFLOW_CHOICES = [
    ("Chat", "chat"),
    ("Image", "image"),
]
DEFAULT_RUNTIME_SELECTION = "online" if DEFAULT_RUNTIME_PREFERENCE == "online" else "offline"
DEFAULT_WORKFLOW_SELECTION = "chat"
DEFAULT_MODEL_LABEL = "Offline engine"
DEFAULT_LAUNCH_PORT = SONIC_PORT
PORT_FALLBACK_COUNT = 5
GRADIO_BIND_HOST = "0.0.0.0"
SONIC_FAVICON_PATH = Path(__file__).with_name("sonic_favicon.svg")
SONIC_LOGO_PATH = Path(__file__).with_name("sonic_logo.png")
SONIC_BRAND_REFERENCE_PATH = Path(__file__).with_name("sonic_brand_reference.jpeg")
logger = logging.getLogger(__name__)

SONIC_CSS = """
:root {
    --sonic-font-body: "Aptos", "Segoe UI Variable Text", "Trebuchet MS", sans-serif;
    --sonic-font-display: "Bahnschrift", "Segoe UI Variable Display", "Aptos Display", sans-serif;
    --sonic-font-mono: "Cascadia Mono", "Consolas", "SFMono-Regular", monospace;
    --sonic-bg: #0c051d;
    --sonic-bg-soft: #170b36;
    --sonic-panel: rgba(22, 11, 47, 0.6);
    --sonic-panel-strong: rgba(26, 12, 56, 0.85);
    --sonic-surface: rgba(255, 255, 255, 0.05);
    --sonic-surface-hover: rgba(255, 255, 255, 0.09);
    --sonic-border: rgba(162, 82, 255, 0.22);
    --sonic-border-strong: rgba(162, 82, 255, 0.45);
    --sonic-text: #f5f3ff;
    --sonic-muted: #c0b3df;
    --sonic-muted-strong: #e9e3ff;
    --sonic-cyan: #d8b4fe;
    --sonic-blue: #a252ff;
    --sonic-indigo: #7f35ff;
    --sonic-purple: #8b35ff;
    --sonic-purple-soft: #b45cff;
    --sonic-magenta: #f04cff;
    --sonic-gold: #ffd24a;
    --sonic-success: #74d4bb;
    --sonic-warning: #f2c66d;
    --sonic-danger: #ff8a8a;
    --sonic-shadow: 0 30px 80px rgba(0, 0, 0, 0.52);
    --sonic-shadow-soft: 0 14px 34px rgba(31, 6, 66, 0.35);
    --sonic-bg-layer: radial-gradient(circle at 18% 16%, rgba(127, 53, 255, 0.24) 0%, transparent 26%),
        radial-gradient(circle at 84% 14%, rgba(180, 92, 255, 0.18) 0%, transparent 22%),
        radial-gradient(circle at 50% 46%, #22104a 0%, #0c051d 72%, #07030f 100%);
    --sonic-shell-radius: 20px;
    --sonic-card-radius: 12px;
    --sonic-control-radius: 999px;
}

* {
    box-sizing: border-box;
}

/* Ensure the main page is allowed to scroll and doesn't trap content */
html, body {
    min-height: 100vh !important;
    min-height: 100dvh !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    margin: 0 !important;
    padding: 0 !important;
    background: var(--sonic-bg-layer) !important;
    background-color: #0c051d !important;
    color: var(--sonic-text) !important;
    font-family: var(--sonic-font-body) !important;
}

body, button, input, textarea, select {
    font-family: var(--sonic-font-body) !important;
}

h1, h2, h3, h4, h5, h6,
.sonic-navbar-wordmark,
.sonic-empty-state__title {
    font-family: var(--sonic-font-display) !important;
}

/* Hide default Gradio footers */
footer, .footer, .gradio-container footer, .gradio-container .footer {
    display: none !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.02);
}

::-webkit-scrollbar-thumb {
    background: rgba(162, 82, 255, 0.25);
    border-radius: 999px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(180, 92, 255, 0.45);
}

/* Gradio container adjustments to support vertical page scrolling */
.gradio-container {
    max-width: none !important;
    padding: 0 !important;
    min-height: 100vh !important;
    min-height: 100dvh !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: visible !important;
    background: var(--sonic-bg-layer) !important;
    background-color: #0c051d !important;
}

.gradio-container > .main,
.gradio-container > .main > .wrap,
.gradio-container > .main > .wrap > main {
    min-height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: visible !important;
    background: var(--sonic-bg-layer) !important;
    background-color: #0c051d !important;
}

/* Main App Shell Layout */
#sonic-app {
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    max-width: 1600px !important;
    margin: 0 auto !important;
    padding: 24px !important;
    gap: 16px !important;
    min-height: 100vh !important;
    min-height: 100dvh !important;
    box-sizing: border-box !important;
    overflow: visible !important;
    background: transparent !important;
}

/* Top Navbar with Runtime controls */
#sonic-navbar {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: 16px 24px !important;
    border-radius: var(--sonic-shell-radius) !important;
    border: 1px solid var(--sonic-border) !important;
    background: rgba(18, 9, 43, 0.4) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    flex-shrink: 0 !important;
    margin-bottom: 8px !important;
}

.sonic-navbar-brand {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    text-decoration: none !important;
}

.sonic-navbar-wordmark {
    font-size: 1.45rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.16em !important;
    color: #ffffff !important;
    text-transform: uppercase !important;
}

#sonic-navbar-controls {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    gap: 12px !important;
    flex-wrap: wrap !important;
    min-width: 0 !important;
}

#sonic-runtime-dropdown,
#sonic-workflow-dropdown {
    min-width: 120px !important;
}

#sonic-runtime-dropdown label,
#sonic-workflow-dropdown label {
    color: var(--sonic-text) !important;
}

/* Brand Indicator */
.sonic-mark {
    position: relative;
    display: grid;
    place-items: center;
    overflow: hidden;
    border: 1px solid rgba(196, 111, 255, 0.32);
    background: radial-gradient(circle at 52% 38%, rgba(58, 169, 255, 0.2), transparent 48%), rgba(18, 2, 34, 0.84);
    box-shadow: 0 0 0 4px rgba(129, 50, 255, 0.08), 0 14px 34px rgba(113, 38, 255, 0.34);
    isolation: isolate;
}

.sonic-mark::before {
    content: "";
    position: absolute;
    inset: -14%;
    z-index: 0;
    background: radial-gradient(circle at center, rgba(180, 92, 255, 0.35), transparent 62%);
    filter: blur(10px);
}

.sonic-mark__image {
    position: relative;
    z-index: 1;
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: 50% 20%;
    transform: scale(1.4);
}

.sonic-mark__fallback {
    position: relative;
    z-index: 1;
    display: grid;
    place-items: center;
    width: 100%;
    height: 100%;
    color: #f4f9ff;
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: 0.22em;
}

.sonic-mark--small {
    width: 36px;
    height: 36px;
    border-radius: 50%;
}

.sonic-mark--large {
    width: 78px;
    height: 78px;
    border-radius: 50%;
}

/* ==========================================================================
   PREMIUM GLASSMORPHIC SEGMENTED RADIO SWITCHES (Runtime & Workflow Mode)
   ========================================================================== */

/* 1. Reset standard Gradio wrappers (containers, borders, backgrounds) */
.block:has(#sonic-runtime-dropdown),
div:has(> #sonic-runtime-dropdown),
#sonic-runtime-dropdown,
#sonic-runtime-dropdown .wrap,
#sonic-runtime-dropdown fieldset,
#sonic-runtime-dropdown .form,
#sonic-runtime-dropdown .block,
#sonic-runtime-dropdown .gradio-radio,
#sonic-runtime-dropdown .gradio-radio .wrap,
#sonic-runtime-dropdown .gradio-radio fieldset,
#sonic-runtime-dropdown [role="radiogroup"],
#sonic-runtime-dropdown .container,
#sonic-runtime-dropdown .gradio-radio > div,
#sonic-runtime-dropdown .gradio-radio > div > div,
.block:has(#sonic-workflow-dropdown),
div:has(> #sonic-workflow-dropdown),
#sonic-workflow-dropdown,
#sonic-workflow-dropdown .wrap,
#sonic-workflow-dropdown fieldset,
#sonic-workflow-dropdown .form,
#sonic-workflow-dropdown .block,
#sonic-workflow-dropdown .gradio-radio,
#sonic-workflow-dropdown .gradio-radio .wrap,
#sonic-workflow-dropdown .gradio-radio fieldset,
#sonic-workflow-dropdown [role="radiogroup"],
#sonic-workflow-dropdown .container,
#sonic-workflow-dropdown .gradio-radio > div,
#sonic-workflow-dropdown .gradio-radio > div > div {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    padding: 0 !important;
    margin: 0 !important;
    width: fit-content !important;
    min-width: 0 !important;
}

#sonic-navbar .form:has(#sonic-runtime-dropdown),
#sonic-navbar .block:has(#sonic-runtime-dropdown),
#sonic-navbar .wrap:has(#sonic-runtime-dropdown),
#sonic-navbar .form:has(#sonic-workflow-dropdown),
#sonic-navbar .block:has(#sonic-workflow-dropdown),
#sonic-navbar .wrap:has(#sonic-workflow-dropdown) {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    width: fit-content !important;
    min-width: 0 !important;
}

/* Remove default Gradio border pseudo-elements */
.block:has(#sonic-runtime-dropdown)::before,
.block:has(#sonic-runtime-dropdown)::after,
div:has(> #sonic-runtime-dropdown)::before,
div:has(> #sonic-runtime-dropdown)::after,
#sonic-runtime-dropdown::before,
#sonic-runtime-dropdown::after,
#sonic-runtime-dropdown fieldset::before,
#sonic-runtime-dropdown fieldset::after,
#sonic-runtime-dropdown .gradio-radio::before,
#sonic-runtime-dropdown .gradio-radio::after,
.block:has(#sonic-workflow-dropdown)::before,
.block:has(#sonic-workflow-dropdown)::after,
div:has(> #sonic-workflow-dropdown)::before,
div:has(> #sonic-workflow-dropdown)::after,
#sonic-workflow-dropdown::before,
#sonic-workflow-dropdown::after,
#sonic-workflow-dropdown fieldset::before,
#sonic-workflow-dropdown fieldset::after,
#sonic-workflow-dropdown .gradio-radio::before,
#sonic-workflow-dropdown .gradio-radio::after {
    content: none !important;
    display: none !important;
    border: none !important;
    box-shadow: none !important;
}

/* 2. Style the options container as a dark glass pill wrapper */
#sonic-runtime-dropdown > div,
#sonic-runtime-dropdown fieldset > div,
#sonic-runtime-dropdown .gradio-radio fieldset > div,
#sonic-runtime-dropdown .wrap > div,
#sonic-runtime-dropdown .wrap fieldset > div,
#sonic-workflow-dropdown > div,
#sonic-workflow-dropdown fieldset > div,
#sonic-workflow-dropdown .gradio-radio fieldset > div,
#sonic-workflow-dropdown .wrap > div,
#sonic-workflow-dropdown .wrap fieldset > div {
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
    padding: 4px !important;
    border-radius: 999px !important;
    background: rgba(22, 11, 47, 0.6) !important;
    border: 1px solid rgba(162, 82, 255, 0.22) !important;
    box-shadow: 0 0 15px rgba(162, 82, 255, 0.08), inset 0 1px 3px rgba(0, 0, 0, 0.4) !important;
    width: fit-content !important;
}

/* 3. Style the option labels */
#sonic-runtime-dropdown label,
#sonic-workflow-dropdown label {
    padding: 6px 20px !important;
    border-radius: 999px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    border: none !important;
    background: transparent !important;
    color: var(--sonic-muted) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    user-select: none !important;
}

/* Hover state styling */
#sonic-runtime-dropdown label:hover,
#sonic-workflow-dropdown label:hover {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.05) !important;
}

/* 4. Active state styling (with neon violet glow) */
#sonic-runtime-dropdown label:has(input:checked),
#sonic-workflow-dropdown label:has(input:checked) {
    background: linear-gradient(135deg, var(--sonic-indigo) 0%, var(--sonic-blue) 100%) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 0 14px rgba(162, 82, 255, 0.45), 0 2px 4px rgba(0, 0, 0, 0.2) !important;
}

/* 5. Hide default Gradio radio circles */
#sonic-runtime-dropdown input[type="radio"],
#sonic-workflow-dropdown input[type="radio"] {
    display: none !important;
}

/* Workspace layout */
#sonic-shell {
    display: flex !important;
    flex-direction: row !important;
    gap: 20px !important;
    flex: 1 !important;
    height: auto !important;
    min-height: 0 !important;
    overflow: visible !important;
}

/* Sidebar */
#sonic-left-sidebar {
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
    gap: 16px !important;
    align-items: stretch !important;
    background: rgba(22, 11, 47, 0.45) !important;
    border: 1px solid var(--sonic-border) !important;
    border-radius: var(--sonic-shell-radius) !important;
    padding: 20px !important;
    padding-bottom: 24px !important;
    backdrop-filter: blur(8px) !important;
    flex: 0 0 280px !important;
    width: 280px !important;
    min-width: 280px !important;
    max-width: 280px !important;
    min-height: 0 !important;
    height: auto !important;
    overflow: visible !important;
}

#sonic-sidebar-top-controls {
    flex-shrink: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
    width: 100% !important;
}

#sonic-sidebar-list-shell {
    flex: none !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: visible !important;
    gap: 8px !important;
    width: 100% !important;
}

#sonic-sidebar-footer {
    flex-shrink: 0 !important;
    margin-top: auto !important;
    padding-top: 10px !important;
    padding-bottom: 12px !important;
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
}

#sonic-new-chat-button,
#sonic-new-chat-button button,
#sonic-new-chat-button.gradio-button {
    width: 100% !important;
    padding: 10px 16px !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    text-align: center !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(168, 85, 247, 0.5) !important;
    box-shadow: 0 0 15px rgba(124, 58, 237, 0.45), 0 4px 15px rgba(127, 53, 255, 0.35) !important;
}

#sonic-new-chat-button:hover,
#sonic-new-chat-button button:hover,
#sonic-new-chat-button.gradio-button:hover {
    background: linear-gradient(135deg, #8b5cf6 0%, #b45cff 100%) !important;
    box-shadow: 0 0 22px rgba(168, 85, 247, 0.7), 0 4px 20px rgba(127, 53, 255, 0.45) !important;
    transform: translateY(-1.5px) !important;
}

#sonic-clear-chat-button,
#sonic-clear-chat-button button,
#sonic-clear-chat-button.gradio-button {
    width: 100% !important;
    padding: 10px 16px !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    text-align: center !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    background: rgba(22, 11, 47, 0.65) !important;
    color: #ffffff !important;
    border: 1px solid rgba(168, 85, 247, 0.45) !important;
    box-shadow: 0 0 10px rgba(168, 85, 247, 0.15) !important;
}

#sonic-clear-chat-button:hover,
#sonic-clear-chat-button button:hover,
#sonic-clear-chat-button.gradio-button:hover {
    background: rgba(139, 92, 246, 0.12) !important;
    border-color: rgba(168, 85, 247, 0.75) !important;
    box-shadow: 0 0 15px rgba(168, 85, 247, 0.35), 0 0 0 1px rgba(255, 255, 255, 0.1) !important;
    transform: translateY(-1px) !important;
}

/* Voice output checkbox in sidebar */
#sonic-voice-toggle {
    margin-top: 8px !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
}

#sonic-voice-toggle label {
    display: flex !important;
    align-items: center !important;
    flex-direction: row-reverse !important;
    justify-content: space-between !important;
    gap: 12px !important;
    color: #ffffff !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    padding: 10px 16px !important;
    background: rgba(22, 11, 47, 0.5) !important;
    border: 1px solid rgba(162, 82, 255, 0.25) !important;
    border-radius: 12px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

#sonic-voice-toggle label:hover {
    background: rgba(30, 15, 64, 0.75) !important;
    border-color: rgba(168, 85, 247, 0.6) !important;
    box-shadow: 0 0 12px rgba(168, 85, 247, 0.15) !important;
}

#sonic-voice-toggle label span {
    color: #ffffff !important;
    font-weight: 600 !important;
}

#sonic-voice-toggle input[type="checkbox"] {
    width: 36px !important;
    height: 20px !important;
    border-radius: 999px !important;
    border: 1px solid rgba(168, 85, 247, 0.45) !important;
    background: rgba(255, 255, 255, 0.08) !important;
    appearance: none !important;
    -webkit-appearance: none !important;
    -moz-appearance: none !important;
    cursor: pointer !important;
    position: relative !important;
    outline: none !important;
    transition: all 0.2s ease !important;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.4) !important;
    flex-shrink: 0 !important;
}

#sonic-voice-toggle input[type="checkbox"]::after {
    content: "" !important;
    position: absolute !important;
    top: 2px !important;
    left: 2px !important;
    width: 14px !important;
    height: 14px !important;
    border-radius: 50% !important;
    background: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

#sonic-voice-toggle input[type="checkbox"]:checked {
    border-color: #b45cff !important;
    background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
    box-shadow: 0 0 12px rgba(168, 85, 247, 0.65) !important;
}

#sonic-voice-toggle input[type="checkbox"]:checked::after {
    transform: translateX(16px) !important;
    background: #ffffff !important;
}

#sonic-voice-toggle input[type="checkbox"]:focus,
#sonic-voice-toggle input[type="checkbox"]:focus-visible {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.4) !important;
}

/* Conversation history */
#sonic-history-heading-host {
    margin-top: 4px !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    flex-shrink: 0 !important;
}

.sonic-history-heading {
    color: rgba(245, 243, 255, 0.78) !important;
    font-size: 0.72rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 2px 4px 0 !important;
}

#sonic-chat-history-list {
    flex: none !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: visible !important;
    width: 100% !important;
}

#sonic-chat-history-list .wrap,
#sonic-chat-history-list fieldset,
#sonic-chat-history-list .form,
#sonic-chat-history-list .block,
#sonic-chat-history-list [role="radiogroup"],
#sonic-chat-history-list .gradio-radio {
    width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
}

#sonic-chat-history-list .wrap,
#sonic-chat-history-list fieldset,
#sonic-chat-history-list [role="radiogroup"] {
    flex: none !important;
    overflow: visible !important;
}

#sonic-chat-history-list fieldset,
#sonic-chat-history-list [role="radiogroup"] {
    overflow-y: visible !important;
    padding-right: 4px !important;
}

#sonic-chat-history-list label {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    min-height: 38px !important;
    margin: 0 0 7px !important;
    padding: 9px 11px !important;
    border-radius: 10px !important;
    background: rgba(255, 255, 255, 0.035) !important;
    border: 1px solid rgba(162, 82, 255, 0.14) !important;
    color: rgba(245, 243, 255, 0.82) !important;
    font-size: 0.86rem !important;
    font-weight: 600 !important;
    line-height: 1.25 !important;
    cursor: pointer !important;
    transition: background 0.18s ease, border-color 0.18s ease, color 0.18s ease !important;
    flex-shrink: 0 !important;
}

#sonic-chat-history-list label:hover {
    background: rgba(162, 82, 255, 0.12) !important;
    border-color: rgba(180, 92, 255, 0.36) !important;
    color: #ffffff !important;
}

#sonic-chat-history-list label:has(input:checked) {
    background: rgba(127, 53, 255, 0.34) !important;
    border-color: rgba(216, 180, 254, 0.58) !important;
    color: #ffffff !important;
    box-shadow: inset 3px 0 0 rgba(216, 180, 254, 0.86), 0 8px 20px rgba(80, 30, 150, 0.18) !important;
}

#sonic-chat-history-list input[type="radio"] {
    position: absolute !important;
    opacity: 0 !important;
    width: 1px !important;
    height: 1px !important;
    pointer-events: none !important;
}

#sonic-chat-history-list span {
    min-width: 0 !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}

#sonic-delete-chat-button,
#sonic-delete-chat-button button,
#sonic-delete-chat-button.gradio-button {
    width: 100% !important;
    padding: 9px 14px !important;
    border-radius: 999px !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    color: #ffdada !important;
    background: rgba(96, 22, 50, 0.34) !important;
    border: 1px solid rgba(255, 138, 138, 0.36) !important;
    box-shadow: none !important;
}

#sonic-delete-chat-button:hover,
#sonic-delete-chat-button button:hover,
#sonic-delete-chat-button.gradio-button:hover {
    background: rgba(136, 32, 70, 0.5) !important;
    border-color: rgba(255, 138, 138, 0.68) !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}

/* Chat Stage container */
#sonic-chat-stage {
    display: flex !important;
    flex-direction: column !important;
    gap: 16px !important;
    background: rgba(22, 11, 47, 0.5) !important;
    border: 1px solid var(--sonic-border) !important;
    border-radius: var(--sonic-shell-radius) !important;
    padding: 24px !important;
    backdrop-filter: blur(10px) !important;
    flex: 1 !important;
    min-width: 0 !important;
    min-height: 0 !important;
    height: auto !important;
    overflow: visible !important;
    position: relative !important;
}

/* Chat Transcript area scroll container */
#sonic-chat-scroll {
    position: relative !important;
    flex: 1 !important;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    height: auto !important;
    overflow: visible !important;
}

/* Chatbot component layout */
#sonic-chatbot {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
    height: auto !important;
    overflow: visible !important;
}

#sonic-chatbot > .wrapper {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
    height: auto !important;
    overflow: visible !important;
    background: rgba(22, 11, 47, 0.5) !important;
}

#sonic-generated-image {
    width: 100% !important;
    max-width: 900px !important;
    margin: 0 auto !important;
    border: 1px solid rgba(162, 82, 255, 0.24) !important;
    border-radius: var(--sonic-card-radius) !important;
    background: rgba(10, 4, 24, 0.55) !important;
    box-shadow: var(--sonic-shadow-soft) !important;
    overflow: hidden !important;
}

#sonic-generated-image img {
    border-radius: inherit !important;
}

#sonic-chatbot .bubble-wrap {
    flex: 1 !important;
    overflow-y: visible !important;
    overflow-x: hidden !important;
    min-height: 0 !important;
    height: auto !important;
    padding: 4px 8px 16px !important;
    overscroll-behavior: contain;
    -webkit-overflow-scrolling: touch;
    background: rgba(22, 11, 47, 0.5) !important;
}

#sonic-chatbot .message-wrap {
    flex: 0 0 auto !important;
    min-height: 0 !important;
    height: auto !important;
    overflow: visible !important;
}

#sonic-chatbot .placeholder {
    display: none !important;
}

#sonic-chatbot .message-row {
    padding-bottom: 12px !important;
}

#sonic-chatbot .message,
#sonic-chatbot .bubble {
    border-radius: 18px !important;
}

#sonic-chatbot .message-row,
#sonic-chatbot .message,
#sonic-chatbot .bubble,
#sonic-chatbot [data-testid="bot"],
#sonic-chatbot [data-testid="user"],
#sonic-chatbot .message-content,
#sonic-chatbot .md,
#sonic-chatbot .markdown,
#sonic-chatbot .prose {
    max-height: none !important;
    overflow: visible !important;
}

/* Message bubbles styling - width fixes */
#sonic-chatbot .message {
    width: fit-content !important;
    max-width: 85% !important;
    padding: 13px 18px !important;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.22) !important;
    font-size: 0.98rem !important;
    line-height: 1.62 !important;
}

#sonic-chatbot .message.user,
#sonic-chatbot .bubble.user,
#sonic-chatbot .message-row.user .message {
    background: #7f35ff !important;
    color: #ffffff !important;
    border-radius: 18px 18px 4px 18px !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(127, 53, 255, 0.25) !important;
}

#sonic-chatbot .message.assistant,
#sonic-chatbot .bubble.assistant,
#sonic-chatbot .message-row.assistant .message {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #f5f3ff !important;
    border-radius: 18px 18px 18px 4px !important;
    border: 1px solid rgba(162, 82, 255, 0.18) !important;
    max-width: 92% !important;
}

#sonic-chatbot .markdown,
#sonic-chatbot .md,
#sonic-chatbot .prose,
#sonic-chatbot .message-content {
    color: var(--sonic-text) !important;
}

#sonic-chatbot p {
    margin-bottom: 0.62rem !important;
}

#sonic-chatbot p:last-child {
    margin-bottom: 0 !important;
}

/* Code responses container styling */
#sonic-chatbot pre {
    background: #090514 !important;
    border: 1px solid rgba(162, 82, 255, 0.25) !important;
    border-radius: 12px !important;
    padding: 1.1rem !important;
    overflow-x: auto !important;
    width: 100% !important;
    max-width: 100% !important;
    white-space: pre !important;
    word-break: normal !important;
    word-wrap: normal !important;
}

#sonic-chatbot code {
    background: rgba(25, 11, 48, 0.8) !important;
    color: #e9e3ff !important;
    border-radius: 6px !important;
    padding: 0.15rem 0.35rem !important;
    font-family: var(--sonic-font-mono) !important;
    font-size: 0.9rem !important;
}

#sonic-chatbot pre code {
    background: transparent !important;
    padding: 0 !important;
}

#sonic-chatbot a {
    color: var(--sonic-cyan) !important;
}

/* Empty State layout */
#sonic-empty-state-host {
    position: absolute !important;
    inset: 0 !important;
    z-index: 2 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    pointer-events: none !important;
    padding: 24px !important;
}

.sonic-empty-state {
    width: min(100%, 560px) !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 12px !important;
    text-align: center !important;
    pointer-events: auto !important;
    margin: auto !important;
}

.sonic-empty-state .sonic-mark {
    margin-bottom: 12px !important;
    filter: drop-shadow(0 0 25px rgba(162, 82, 255, 0.55)) !important;
}

.sonic-empty-state__title {
    color: #ffffff !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    margin-bottom: 8px !important;
    letter-spacing: -0.01em !important;
}

.sonic-empty-state__body {
    font-size: 1.05rem !important;
    line-height: 1.6 !important;
    color: #c0b3df !important;
}

.sonic-empty-state--hidden {
    display: none !important;
}

/* Composer / Message Input styling */
#sonic-composer-meta {
    width: 100% !important;
    max-width: 900px !important;
    margin: 0 auto !important;
    text-align: center !important;
    flex-shrink: 0 !important;
    min-width: 0 !important;
}

.sonic-composer-hint {
    color: rgba(192, 179, 223, 0.45) !important;
    font-size: 0.72rem !important;
}

#sonic-composer-stack {
    width: 100% !important;
    max-width: 900px !important;
    margin: 0 auto !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 10px !important;
    min-width: 0 !important;
}

#sonic-composer {
    width: 100% !important;
    max-width: 900px !important;
    margin: 0 auto !important;
    flex-shrink: 0 !important;
    min-width: 0 !important;
}

#sonic-composer-card {
    background: rgba(22, 11, 47, 0.5) !important;
    border: 1px solid rgba(162, 82, 255, 0.3) !important;
    border-radius: 999px !important;
    padding: 6px 12px 6px 20px !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    min-width: 0 !important;
    box-shadow: 0 4px 20px rgba(13, 3, 31, 0.4) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

#sonic-composer-card:focus-within {
    border-color: rgba(162, 82, 255, 0.6) !important;
    box-shadow: 0 0 15px rgba(162, 82, 255, 0.2) !important;
}

#sonic-composer-row {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    min-width: 0 !important;
    gap: 10px !important;
}

#sonic-composer-row > * {
    min-width: 0 !important;
    margin: 0 !important;
}

#sonic-composer-input {
    flex: 1 1 auto !important;
    display: flex !important;
    align-items: center !important;
    min-width: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

#sonic-composer-send {
    flex: 0 0 auto !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-width: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

#sonic-attach-button {
    flex-shrink: 0 !important;
    min-width: 0 !important;
}

#sonic-attach-button,
#sonic-attach-button button,
#sonic-attach-button .gradio-button {
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    border: 1px solid rgba(162, 82, 255, 0.28) !important;
    background: rgba(255, 255, 255, 0.04) !important;
    color: #f5f3ff !important;
    box-shadow: none !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
}

#sonic-attach-button:hover,
#sonic-attach-button button:hover,
#sonic-attach-button .gradio-button:hover {
    background: rgba(127, 53, 255, 0.2) !important;
    border-color: rgba(180, 92, 255, 0.48) !important;
    transform: translateY(-1px) !important;
}

#sonic-message-box {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #ffffff !important;
    font-size: 0.98rem !important;
    padding: 8px 0 !important;
    width: 100% !important;
    min-width: 0 !important;
    flex: 1 1 auto !important;
}

#sonic-message-box textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #ffffff !important;
    padding: 0 !important;
    resize: none !important;
    line-height: 1.45 !important;
}

#sonic-message-box textarea::placeholder {
    color: rgba(192, 179, 223, 0.6) !important;
}

#sonic-message-box textarea:focus {
    box-shadow: none !important;
}

/* Composer Send Button */
#sonic-send-button {
    background: #ffffff !important;
    color: #120024 !important;
    width: 52px !important;
    height: 52px !important;
    min-width: 52px !important;
    max-width: 52px !important;
    border-radius: 50% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 1.1rem !important;
    font-weight: bold !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 10px rgba(255, 255, 255, 0.2) !important;
    padding: 0 !important;
    flex-shrink: 0 !important;
}

#sonic-send-button:hover {
    background: #d8b4fe !important;
    transform: scale(1.05) !important;
}

#sonic-send-button button {
    background: transparent !important;
    color: inherit !important;
    border: none !important;
    box-shadow: none !important;
    width: 100% !important;
    height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 50% !important;
}

#sonic-composer-send #sonic-send-button,
#sonic-composer-send #sonic-send-button button,
#sonic-composer-send #sonic-send-button .gradio-button {
    width: 52px !important;
    height: 52px !important;
    min-width: 52px !important;
    max-width: 52px !important;
}

/* Responsive layout adjustments */
@media (max-width: 900px) {
    #sonic-app {
        padding: 12px !important;
        height: auto !important;
        min-height: 100vh !important;
        min-height: 100dvh !important;
    }
    
    #sonic-navbar {
        padding: 10px 16px !important;
    }
    
    #sonic-shell {
        flex-direction: column !important;
    }
    
    #sonic-left-sidebar {
        width: 100% !important;
        max-width: none !important;
        min-width: 0 !important;
        padding: 16px !important;
        height: auto !important;
        overflow: visible !important;
        flex: 0 0 auto !important;
    }

    #sonic-sidebar-list-shell {
        max-height: 200px !important;
    }
    
    #sonic-chat-stage {
        padding: 16px !important;
        min-height: 0 !important;
    }
}

/* TTS styling */
#sonic-chatbot .message:has(.sonic-tts-meta),
#sonic-chatbot [data-testid="bot"]:has(.sonic-tts-meta),
#sonic-chatbot .message:has(a[href^="#tts-"]),
#sonic-chatbot [data-testid="bot"]:has(a[href^="#tts-"]) {
    padding-bottom: 42px !important;
    position: relative;
}

.sonic-tts-meta,
#sonic-chatbot a[href^="#tts-"] {
    display: none !important;
}

.sonic-voice-status {
    display: none;
    grid-column: 1 / -1;
    min-height: 34px;
    align-items: center;
    padding: 8px 10px;
    border: 1px solid rgba(242, 198, 109, 0.34);
    border-radius: 8px;
    background: rgba(242, 198, 109, 0.08);
    color: rgba(255, 239, 202, 0.92);
    font-size: 0.72rem;
    line-height: 1.35;
}

.sonic-voice-status--visible {
    display: flex;
}

.sonic-speak-button {
    position: absolute;
    left: 14px;
    bottom: 9px;
    min-width: 52px;
    height: 25px;
    padding: 0 9px;
    border: 1px solid rgba(196, 111, 255, 0.25);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.045);
    color: rgba(245, 236, 255, 0.68);
    font: 600 0.66rem/1 var(--sonic-font-body);
    cursor: pointer;
    transition: color 0.18s ease, border-color 0.18s ease, background 0.18s ease;
}

.sonic-speak-button:hover {
    color: #ffffff;
    border-color: rgba(138, 231, 255, 0.5);
    background: rgba(138, 231, 255, 0.09);
}

.sonic-speak-button--active {
    color: #ffffff;
    border-color: rgba(138, 231, 255, 0.64);
    background: rgba(127, 53, 255, 0.3);
}

.sonic-speak-button:disabled {
    cursor: not-allowed;
    opacity: 0.42;
}
"""

SONIC_TTS_JS = r"""
() => {
    if (window.__sonicTts?.initialized) {
        window.__sonicTts.scan?.({ allowAutoSpeak: false });
        return;
    }

    const speech = window.speechSynthesis;
    const storageKey = "sonic.voice-output.enabled";
    const localeMap = {
        en: ["en-US", "en-GB", "en-IN"],
        hi: ["hi-IN"],
        zh: ["zh-CN", "zh-TW", "zh-HK"],
        es: ["es-ES", "es-MX", "es-US"],
        fr: ["fr-FR", "fr-CA"],
        bn: ["bn-IN", "bn-BD"],
        te: ["te-IN"],
        mr: ["mr-IN"],
        ru: ["ru-RU"],
        gu: ["gu-IN"]
    };
    const state = {
        activeButton: null,
        playbackToken: 0,
        voices: [],
        initialized: false,
        lastSeenKey: "",
        scanTimer: null,
        voiceWarning: ""
    };

    console.debug("SONIC TTS: bootstrap loaded");

    function decodeSpeech(encoded) {
        if (!encoded) return "";
        try {
            const bytes = Uint8Array.from(atob(encoded), (char) => char.charCodeAt(0));
            return new TextDecoder("utf-8").decode(bytes);
        } catch (_) {
            return "";
        }
    }

    function detectLanguage(text) {
        const value = text || "";
        if (/[\u0980-\u09ff]/.test(value)) return "bn";
        if (/[\u0c00-\u0c7f]/.test(value)) return "te";
        if (/[\u0a80-\u0aff]/.test(value)) return "gu";
        if (/[\u0400-\u04ff]/.test(value)) return "ru";
        if (/[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/.test(value)) return "zh";
        if (/[\u0900-\u097f]/.test(value)) return "hi";
        const words = value.toLowerCase().match(/[a-z\u00c0-\u00ff']+/g) || [];
        const hindi = new Set(["aaj", "abhi", "acha", "achha", "aur", "batao", "bhai", "hai", "ka", "kya", "kaun", "mein", "nahi", "tum"]);
        const spanish = new Set(["hola", "gracias", "como", "cómo", "que", "qué", "por", "favor"]);
        const french = new Set(["bonjour", "merci", "avec", "pour", "quoi", "explique"]);
        if (words.filter((word) => hindi.has(word)).length >= 2) return "hi";
        if (words.filter((word) => spanish.has(word)).length >= 2) return "es";
        if (words.filter((word) => french.has(word)).length >= 2) return "fr";
        return "en";
    }

    function fallbackSpeechText(message) {
        const clone = message.cloneNode(true);
        clone.querySelectorAll("pre, .sonic-speak-button, .sonic-tts-meta, a[href^=\"#tts-\"]").forEach((node) => node.remove());
        let text = (clone.innerText || clone.textContent || "").trim();
        text = text.replace(/(?:^|\n)\s*(?:sources?|references?)\s*:?\s*[\s\S]*$/i, "");
        text = text.replace(/https?:\/\/\S+/g, "").replace(/\s+/g, " ").trim();
        return text;
    }

    function messageForMeta(meta) {
        return (
            meta.closest(".message") ||
            meta.closest('[data-testid="bot"]') ||
            meta.closest(".message-row")
        );
    }

    function refreshVoices() {
        state.voices = speech ? speech.getVoices() : [];
        console.debug("SONIC TTS: voice inventory", state.voices.length);
        if (state.voices.length && state.voiceWarning === "no-voices") {
            setVoiceWarning("");
        }
    }

    function voiceStatusNode() {
        let node = document.querySelector("#sonic-voice-status");
        if (node) return node;

        const host = document.querySelector("#sonic-sidebar-controls")
            || document.querySelector("#sonic-control-strip")
            || document.querySelector("#sonic-app");
        if (!host) return null;

        node = document.createElement("div");
        node.id = "sonic-voice-status";
        node.className = "sonic-voice-status";
        node.setAttribute("role", "status");
        node.setAttribute("aria-live", "polite");
        host.appendChild(node);
        return node;
    }

    function setVoiceWarning(message, code = "") {
        state.voiceWarning = code;
        const node = voiceStatusNode();
        if (!node) return;

        const cleaned = (message || "").trim();
        node.textContent = cleaned;
        node.classList.toggle("sonic-voice-status--visible", Boolean(cleaned));
    }

    function chooseVoice(language, localePreferences) {
        refreshVoices();
        const preferences = localePreferences.length
            ? localePreferences
            : (localeMap[language] || localeMap.en);
        const languageBase = language.toLowerCase();
        const candidates = state.voices.filter((voice) => {
            const voiceLanguage = (voice.lang || "").toLowerCase();
            return voiceLanguage === languageBase || voiceLanguage.startsWith(languageBase + "-");
        });
        if (!candidates.length) return null;

        return candidates
            .map((voice) => {
                const voiceLanguage = (voice.lang || "").toLowerCase();
                const voiceName = (voice.name || "").toLowerCase();
                let score = 0;
                preferences.forEach((locale, index) => {
                    if (voiceLanguage === locale.toLowerCase()) score += 80 - index * 8;
                });
                if (/(natural|neural|premium|enhanced|google|microsoft|siri)/.test(voiceName)) score += 24;
                if (/(desktop|compact|novelty)/.test(voiceName)) score -= 12;
                if (voice.localService) score += 8;
                if (voice.default) score += 4;
                return { voice, score };
            })
            .sort((a, b) => b.score - a.score)[0].voice;
    }

    function chunkSpeech(text, maxLength = 220) {
        const sentences = text.match(/[^.!?。！？\n]+[.!?。！？]?/g) || [text];
        const chunks = [];
        let current = "";
        const pushCurrent = () => {
            const cleaned = current.trim();
            if (cleaned) chunks.push(cleaned);
            current = "";
        };

        sentences.forEach((sentence) => {
            const cleaned = sentence.trim();
            if (!cleaned) return;
            if ((current + " " + cleaned).trim().length <= maxLength) {
                current = (current + " " + cleaned).trim();
                return;
            }
            pushCurrent();
            if (cleaned.length <= maxLength) {
                current = cleaned;
                return;
            }
            const words = cleaned.split(/\s+/);
            words.forEach((word) => {
                if ((current + " " + word).trim().length > maxLength) pushCurrent();
                current = (current + " " + word).trim();
            });
        });
        pushCurrent();
        return chunks;
    }

    function resetActiveButton() {
        if (!state.activeButton) return;
        state.activeButton.textContent = "Speak";
        state.activeButton.classList.remove("sonic-speak-button--active");
        state.activeButton.setAttribute("aria-label", "Read this reply aloud");
        state.activeButton = null;
    }

    function stopPlayback() {
        state.playbackToken += 1;
        if (speech) speech.cancel();
        resetActiveButton();
    }

    function speakButton(button) {
        if (!speech) {
            setVoiceWarning("Voice playback is not supported in this browser.", "unsupported");
            return;
        }
        if (!button || button.disabled) return;
        if (state.activeButton === button && speech.speaking) {
            stopPlayback();
            return;
        }

        stopPlayback();
        const text = button.dataset.speech || "";
        if (!text) {
            setVoiceWarning("SONIC did not receive any text to read aloud.", "empty-text");
            return;
        }
        const language = button.dataset.language || detectLanguage(text);
        const locales = (button.dataset.locales || "").split(",").filter(Boolean);
        const selectedVoice = chooseVoice(language, locales);
        const chunks = chunkSpeech(text);
        if (!chunks.length) {
            setVoiceWarning("SONIC could not prepare this reply for voice playback.", "empty-chunks");
            return;
        }
        if (!state.voices.length) {
            setVoiceWarning("No browser voice is available yet. Try again in a moment or use a browser with speech support.", "no-voices");
        } else {
            setVoiceWarning("");
        }
        const token = state.playbackToken;
        let index = 0;

        console.debug("SONIC TTS: speak requested", {
            language,
            locales,
            chunkCount: chunks.length,
            textPreview: text.slice(0, 120)
        });

        state.activeButton = button;
        button.textContent = "Stop";
        button.classList.add("sonic-speak-button--active");
        button.setAttribute("aria-label", "Stop reading this reply");

        const playNext = () => {
            if (token !== state.playbackToken || index >= chunks.length) {
                if (token === state.playbackToken) resetActiveButton();
                return;
            }
            const utterance = new SpeechSynthesisUtterance(chunks[index]);
            utterance.lang = selectedVoice?.lang || locales[0] || (localeMap[language] || localeMap.en)[0];
            utterance.rate = 1;
            utterance.pitch = 1;
            utterance.volume = 1;
            if (selectedVoice) utterance.voice = selectedVoice;
            utterance.onend = () => {
                index += 1;
                playNext();
            };
            utterance.onerror = (event) => {
                if (event.error !== "interrupted" && event.error !== "canceled") {
                    console.warn("SONIC voice playback error:", event.error);
                    setVoiceWarning(`Voice playback failed: ${event.error || "unknown error"}.`, "playback-error");
                }
                if (token === state.playbackToken) resetActiveButton();
            };
            console.debug("SONIC TTS: speaking chunk", {
                index: index + 1,
                total: chunks.length,
                language: utterance.lang,
                voice: selectedVoice ? selectedVoice.name : "browser-default"
            });
            speech.speak(utterance);
            speech.resume();
        };
        playNext();
    }

    function decorateMessages() {
        const root = document.querySelector("#sonic-chatbot");
        if (!root) return [];
        const decorated = [];
        const metas = Array.from(root.querySelectorAll(".sonic-tts-meta, a[href^=\"#tts-\"]"));
        metas.forEach((meta) => {
            const message = messageForMeta(meta);
            if (!message || message.querySelector(":scope > .sonic-speak-button")) return;
            
            let rawPayload = "";
            if (meta.tagName.toLowerCase() === "a") {
                const href = meta.getAttribute("href") || "";
                rawPayload = decodeURIComponent(href).substring(5); // strip "#tts-"
            } else {
                rawPayload = (meta.textContent || "").trim();
            }
            
            const payloadParts = rawPayload.split("|", 2);
            const payloadLanguage = payloadParts.length === 2 ? payloadParts[0].trim() : "";
            const encodedText = payloadParts.length === 2 ? payloadParts[1].trim() : rawPayload;
            const text = decodeSpeech(encodedText) || fallbackSpeechText(message);
            if (!text) return;
            const language = payloadLanguage || meta.dataset.language || detectLanguage(text);
            const button = document.createElement("button");
            button.type = "button";
            button.className = "sonic-speak-button";
            button.textContent = "Speak";
            button.setAttribute("aria-label", "Read this reply aloud");
            button.title = "Read aloud";
            button.dataset.speech = text;
            button.dataset.language = language;
            button.dataset.locales = meta.dataset.locales || (localeMap[language] || localeMap.en).join(",");
            button.addEventListener("click", (event) => {
                event.preventDefault();
                event.stopPropagation();
                speakButton(button);
            });
            if (!speech) {
                button.disabled = true;
                button.title = "Speech playback is not supported in this browser";
                setVoiceWarning("Voice playback is not supported in this browser.", "unsupported");
            }
            message.appendChild(button);
            decorated.push(button);
            console.debug("SONIC TTS: attached speak button", { language, textPreview: text.slice(0, 80) });
        });
        return decorated;
    }

    function latestSpeakButton() {
        const buttons = document.querySelectorAll("#sonic-chatbot .sonic-speak-button");
        return buttons.length ? buttons[buttons.length - 1] : null;
    }

    function voiceToggleInput() {
        return document.querySelector('#sonic-voice-toggle input[type="checkbox"]');
    }

    function voiceEnabled() {
        return Boolean(voiceToggleInput()?.checked);
    }

    function latestKey(button) {
        if (!button) return "";
        return (button.dataset.language || "") + ":" + (button.dataset.speech || "");
    }

    function scan({ allowAutoSpeak = true } = {}) {
        decorateMessages();
        const latest = latestSpeakButton();
        const key = latestKey(latest);
        if (!state.initialized) {
            state.lastSeenKey = key;
            state.initialized = true;
            return;
        }
        if (key && key !== state.lastSeenKey) {
            state.lastSeenKey = key;
            console.debug("SONIC TTS: new assistant reply detected", { key, allowAutoSpeak, voiceEnabled: voiceEnabled() });
            if (allowAutoSpeak && voiceEnabled()) speakButton(latest);
        } else if (!key) {
            state.lastSeenKey = "";
        }
    }

    function scheduleScan() {
        window.clearTimeout(state.scanTimer);
        state.scanTimer = window.setTimeout(() => scan(), 80);
    }

    function bindToggle() {
        const input = voiceToggleInput();
        if (!input || input.dataset.sonicBound) return;
        input.dataset.sonicBound = "true";
        let stored = null;
        try {
            stored = window.localStorage.getItem(storageKey);
        } catch (_) {
            stored = null;
        }
        if (stored !== null) input.checked = stored === "true";
        input.disabled = !speech;
        if (!speech) {
            setVoiceWarning("Voice playback is not supported in this browser.", "unsupported");
        }
        input.addEventListener("change", () => {
            console.debug("SONIC TTS: voice toggle changed", { enabled: input.checked });
            try {
                window.localStorage.setItem(storageKey, String(input.checked));
            } catch (_) {
                // Storage can be unavailable in privacy-restricted browser sessions.
            }
            if (!input.checked) {
                stopPlayback();
                return;
            }
            const latest = latestSpeakButton();
            if (latest) speakButton(latest);
        });
    }

    function initialize() {
        refreshVoices();
        if (speech) speech.onvoiceschanged = refreshVoices;
        bindToggle();
        scan({ allowAutoSpeak: false });
        const app = document.querySelector("#sonic-app") || document.body;
        const observer = new MutationObserver(() => {
            bindToggle();
            scheduleScan();
        });
        observer.observe(app, { childList: true, subtree: true });
        document.addEventListener("click", (event) => {
            const target = event.target;
            if (target?.closest?.("#sonic-new-chat-button, #sonic-clear-chat-button")) {
                stopPlayback();
            }
        });
        window.__sonicTts = {
            scan,
            stop: stopPlayback,
            speakLatest: () => {
                const latest = latestSpeakButton();
                if (latest) speakButton(latest);
            },
            state,
            initialized: true
        };
        console.debug("SONIC TTS: initialized");
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initialize, { once: true });
    } else {
        window.setTimeout(initialize, 0);
    }
}
"""

def _clean_message(text: str) -> str:
    return (text or "").strip()


def _normalize_workflow_value(value: str) -> str:
    normalized_value = _clean_message(value).lower()
    if normalized_value in {"image", "generate image", "images"}:
        return "image"
    return "chat"


def _mode_placeholder(mode: str) -> str:
    if _normalize_workflow_value(mode) == "image":
        return "Describe the image you want..."
    return "Message SONIC..."


def _conversation_image_path(conversation: Mapping[str, object] | None) -> str:
    if not conversation:
        return ""

    messages = conversation.get("messages")
    if not isinstance(messages, list):
        return ""

    for message in reversed(messages):
        if not isinstance(message, Mapping):
            continue

        if str(message.get("role") or "").strip().lower() != "assistant":
            continue

        metadata = message.get("metadata")
        if not isinstance(metadata, Mapping):
            continue

        image_path = _clean_message(str(metadata.get("image_path") or ""))
        if image_path and Path(image_path).exists():
            return image_path

    return ""


def _conversation_image_update(conversation: Mapping[str, object] | None):
    image_path = _conversation_image_path(conversation)
    return gr.update(value=image_path or None, visible=bool(image_path))


def _reply_looks_like_code(reply: str) -> bool:
    cleaned_reply = (reply or "").strip()
    if not cleaned_reply:
        return False
    if cleaned_reply.startswith("```"):
        return True

    lines = [line for line in cleaned_reply.splitlines() if line.strip()]
    if not lines:
        return False

    code_hits = 0
    for line in lines[:8]:
        stripped_line = line.strip()
        if line.startswith(("    ", "\t")):
            code_hits += 1
            continue
        if re.match(
            r"^(from|import|def|class|if|elif|else|for|while|try|except|return|const|let|var|function|SELECT|<\w+)",
            stripped_line,
            flags=re.IGNORECASE,
        ):
            code_hits += 1

    return code_hits >= max(1, min(3, len(lines) // 2 or 1))


def _format_reply_for_chat(reply: str, mode: str = "assistant") -> str:
    return (reply or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")


def _compose_display_reply(reply: str, notice: str) -> str:
    cleaned_notice = _clean_message(notice)
    if cleaned_notice and reply:
        return f"{cleaned_notice}\n\n{reply}"
    if cleaned_notice:
        return cleaned_notice
    return reply


def _attach_tts_metadata(reply: str, language_code: str | None) -> str:
    payload = build_speech_payload(reply, language_code)
    if not payload.text:
        return reply

    encoded_text = encode_speech_text(payload.text)
    payload_blob = f"{payload.language_code}|{encoded_text}"
    metadata = f"[ ](#tts-{payload_blob})"
    return f"{reply}\n\n{metadata}" if reply else metadata


def _initial_runtime_state() -> dict[str, str]:
    return {
        "runtime_used": "-",
        "provider_used": "-",
        "model_used": DEFAULT_MODEL_LABEL,
        "requested_runtime": "-",
        "fallback_used": "false",
        "fallback_reason": "",
    }


def _workspace_has_content(conversation: Mapping[str, object] | None) -> bool:
    if not isinstance(conversation, Mapping):
        return False
    messages = conversation.get("messages")
    return bool(isinstance(messages, list) and messages)
def _load_embedded_brand_image(path: Path) -> str:
    if not path.exists():
        return ""

    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower())
    if not mime_type:
        return ""

    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""

    return f"data:{mime_type};base64,{encoded}"


SONIC_BRAND_DATA_URI = _load_embedded_brand_image(
    SONIC_LOGO_PATH if SONIC_LOGO_PATH.exists() else SONIC_BRAND_REFERENCE_PATH
)


def _escape_html(value: str | None) -> str:
    return html.escape(str(value or ""))


def _badge_html(label: str, value: str, tone: str = "muted") -> str:
    return (
        f"<span class='sonic-badge sonic-badge--{tone}'>"
        f"<span class='sonic-badge__label'>{_escape_html(label)}</span>"
        f"<span class='sonic-badge__value'>{_escape_html(value)}</span>"
        "</span>"
    )


def _normalize_runtime_value(value: str) -> str:
    normalized_value = _clean_message(value).lower()
    if normalized_value in {"offline", "local"}:
        return "offline"
    if normalized_value in {"online", "live", "liveweb", "live_web", "web"}:
        return "online"
    if normalized_value == "auto":
        return "auto"
    return ""


def _is_runtime_fallback(runtime_state: Mapping[str, str]) -> bool:
    fallback_used = _clean_message(str(runtime_state.get("fallback_used", ""))).lower()
    return fallback_used in {"1", "true", "yes", "on"}


def _effective_runtime_label(runtime_preference: str, runtime_state: Mapping[str, str]) -> str:
    if _is_runtime_fallback(runtime_state):
        requested_runtime = _normalize_runtime_value(str(runtime_state.get("requested_runtime", "")))
        if requested_runtime:
            return requested_runtime

    runtime_used = _normalize_runtime_value(runtime_state.get("runtime_used", ""))
    if runtime_used:
        return runtime_used
    return _normalize_runtime_value(runtime_preference)


def _runtime_status(runtime_preference: str, runtime_state: Mapping[str, str]) -> tuple[str, str]:
    provider_used = _clean_message(runtime_state.get("provider_used", "-")).lower()
    runtime_used = _normalize_runtime_value(runtime_state.get("runtime_used", ""))
    effective_runtime = runtime_used or _normalize_runtime_value(runtime_preference)

    if provider_used == "error":
        return "Needs attention", "danger"
    if _is_runtime_fallback(runtime_state):
        return "Answered with offline fallback", "warn"
    if effective_runtime == "online":
        if runtime_used == "online":
            return "Online active", "success"
        return "Using current web results", "blue"
    if effective_runtime == "offline":
        return "Ready", "muted"
    return "Ready", "blue"


def _format_runtime_preference_label(runtime_preference: str) -> str:
    normalized_preference = _normalize_runtime_value(runtime_preference)
    if normalized_preference == "offline":
        return "Offline"
    if normalized_preference == "online":
        return "Online"
    return "Offline"


def _format_connection_label(runtime_preference: str, runtime_state: Mapping[str, str]) -> str:
    runtime_used = _normalize_runtime_value(runtime_state.get("runtime_used", ""))
    provider_used = _clean_message(runtime_state.get("provider_used", "-")).lower()
    effective_runtime = runtime_used or _normalize_runtime_value(runtime_preference)

    if provider_used == "error":
        return "Attention"
    if _is_runtime_fallback(runtime_state):
        return "Offline fallback"
    if effective_runtime == "online":
        return "Online"
    if effective_runtime == "offline":
        return "Offline"
    return "Standby"


def _connection_tone(runtime_preference: str, runtime_state: Mapping[str, str]) -> str:
    provider_used = _clean_message(runtime_state.get("provider_used", "-")).lower()
    effective_runtime = _effective_runtime_label(runtime_preference, runtime_state)

    if provider_used == "error":
        return "danger"
    if _is_runtime_fallback(runtime_state):
        return "warn"
    if effective_runtime == "online":
        return "success"
    if effective_runtime == "offline":
        return "muted"
    return "blue"


def _workspace_copy(runtime_preference: str, runtime_state: Mapping[str, str]) -> tuple[str, str, str]:
    effective_runtime = _effective_runtime_label(runtime_preference, runtime_state)

    if effective_runtime == "online":
        return (
            "Online",
            "Current answers, grounded in the web.",
            "Fresh results for news, weather, scores, prices, and recent updates.",
        )

    return (
        "Workspace",
        "Private offline chat for grounded answers and assistance.",
        "Your private offline-first AI workspace.",
    )


def _render_sonic_mark_html(size_class: str = "medium") -> str:
    safe_size = _escape_html(size_class)
    if SONIC_BRAND_DATA_URI:
        return f"""
        <div class="sonic-mark sonic-mark--{safe_size}" aria-hidden="true">
            <img src="{SONIC_BRAND_DATA_URI}" alt="" class="sonic-mark__image" />
        </div>
        """

    return f"""
    <div class="sonic-mark sonic-mark--{safe_size}" aria-hidden="true">
        <div class="sonic-mark__fallback">S</div>
    </div>
    """


def _render_header_html(
    mode: str,
    runtime_preference: str,
    runtime_state: dict[str, str],
    has_messages: bool,
) -> str:
    del mode, has_messages

    status_label, status_tone = _runtime_status(runtime_preference, runtime_state)
    runtime_label = _format_runtime_preference_label(
        _effective_runtime_label(runtime_preference, runtime_state)
    )

    return f"""
    <nav class="sonic-app-header" id="sonic-header-card" aria-label="SONIC navigation">
        <a class="sonic-app-header__brand" href="#" aria-label="SONIC Home">
            {_render_sonic_mark_html("small")}
            <div class="sonic-app-header__copy">
                <div class="sonic-app-header__wordmark">SONIC</div>
            </div>
        </a>
        <div class="sonic-app-header__links" aria-label="Primary navigation">
            <a href="#" class="sonic-nav-link">Home</a>
            <a href="#" class="sonic-nav-link">About</a>
            <a href="#" class="sonic-nav-link sonic-nav-link--active">Services</a>
        </div>
        <div class="sonic-app-header__actions">
            <span class="sonic-nav-status sonic-nav-status--{status_tone}">{_escape_html(runtime_label)} &middot; {_escape_html(status_label)}</span>
            <button class="sonic-auth-button sonic-auth-button--ghost" type="button">Log In</button>
            <button class="sonic-auth-button sonic-auth-button--primary" type="button">Sign Up</button>
        </div>
    </nav>
    """



def _render_context_rail_html(
    runtime_preference: str,
    runtime_state: dict[str, str],
    has_messages: bool,
) -> str:
    del has_messages
    _, title, subtitle = _workspace_copy(runtime_preference, runtime_state)
    status_label, status_tone = _runtime_status(runtime_preference, runtime_state)

    return f"""
    <div class="sonic-workspace-panel" id="sonic-context-card" aria-label="Workspace status">
        <div class="sonic-workspace-panel__copy">
            <div class="sonic-workspace-panel__eyebrow">Workspace Controls</div>
            <div class="sonic-workspace-panel__title">{_escape_html(title)}</div>
            <div class="sonic-workspace-panel__subtitle">
                {_escape_html(subtitle)}
            </div>
        </div>
        <div class="sonic-workspace-panel__meta">
            {_badge_html("Runtime", _format_connection_label(runtime_preference, runtime_state), _connection_tone(runtime_preference, runtime_state))}
            {_badge_html("Status", status_label, status_tone)}
        </div>
    </div>
    """

def _render_empty_state_html(
    runtime_preference: str,
    runtime_state: dict[str, str],
    has_messages: bool,
) -> str:
    del runtime_preference, runtime_state
    if has_messages:
        return '<div id="sonic-empty-state" class="sonic-empty-state sonic-empty-state--hidden"></div>'

    return """
    <div id="sonic-empty-state" class="sonic-empty-state sonic-empty-state--hidden"></div>
    """



def _render_composer_hint_html() -> str:
    return ""


def _render_session_panels(
    runtime_preference: str,
    runtime_state: dict[str, str],
    has_messages: bool,
) -> tuple[str, str, str, str]:
    return (
        _render_header_html("", runtime_preference, runtime_state, has_messages),
        "",
        _render_context_rail_html(runtime_preference, runtime_state, has_messages),
        _render_empty_state_html(runtime_preference, runtime_state, has_messages),
    )


def _parse_port_value(raw_value: str | None) -> int | None:
    if raw_value is None:
        return None

    try:
        port = int(str(raw_value).strip())
    except (TypeError, ValueError):
        return None

    if 1 <= port <= 65535:
        return port
    return None


def _read_launch_env(name: str) -> str:
    return str(os.environ.get(name) or "").strip()


def _parse_bool_value(raw_value: str | None, default: bool = False) -> bool:
    if raw_value is None:
        return default

    cleaned_value = str(raw_value).strip().lower()
    if not cleaned_value:
        return default

    return cleaned_value in {"1", "true", "yes", "on"}


def _launch_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int)
    parser.add_argument("--host")
    parser.add_argument("--auto-port", action="store_true")
    share_group = parser.add_mutually_exclusive_group()
    share_group.add_argument("--share", dest="share", action="store_true")
    share_group.add_argument("--no-share", dest="share", action="store_false")
    parser.set_defaults(share=None)
    return parser


def resolve_launch_port(default_port: int = DEFAULT_LAUNCH_PORT) -> int:
    parser = _launch_arg_parser()
    args, _ = parser.parse_known_args()

    cli_port = args.port
    if cli_port is not None and 1 <= cli_port <= 65535:
        return cli_port

    env_port = _parse_port_value(_read_launch_env("SONIC_PORT"))
    if env_port is not None:
        return env_port

    env_port = _parse_port_value(_read_launch_env("PORT"))
    if env_port is not None:
        return env_port

    env_port = _parse_port_value(_read_launch_env("GRADIO_SERVER_PORT"))
    if env_port is not None:
        return env_port

    return default_port


def resolve_launch_host(default_host: str = SONIC_HOST) -> str:
    parser = _launch_arg_parser()
    args, _ = parser.parse_known_args()

    cli_host = str(args.host or "").strip()
    if cli_host:
        return cli_host

    env_host = _read_launch_env("SONIC_HOST")
    if env_host:
        return env_host

    env_host = _read_launch_env("GRADIO_SERVER_NAME")
    if env_host:
        return env_host

    return default_host


def resolve_launch_share(default_share: bool = SONIC_SHARE) -> bool:
    parser = _launch_arg_parser()
    args, _ = parser.parse_known_args()

    if args.share is not None:
        return bool(args.share)

    env_share = _read_launch_env("SONIC_SHARE")
    if env_share:
        return _parse_bool_value(env_share, default_share)

    legacy_share = _read_launch_env("GRADIO_SHARE")
    if legacy_share:
        return _parse_bool_value(legacy_share, default_share)

    return default_share


def auto_port_enabled() -> bool:
    # Auto-port fallback is enabled by default to ensure reliable launching.
    # We still accept `--auto-port` and SONIC_AUTO_PORT environment variables,
    # but they are effectively always True.
    return True


def _port_is_available(port: int, host: str = GRADIO_BIND_HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def pick_available_port(
    preferred_port: int,
    fallback_count: int = PORT_FALLBACK_COUNT,
    host: str = GRADIO_BIND_HOST,
) -> int:
    upper_port = min(65535, preferred_port + fallback_count)
    for port in range(preferred_port, upper_port + 1):
        if _port_is_available(port, host=host):
            return port
    return preferred_port


def _browser_access_host(host: str) -> str:
    cleaned_host = str(host or "").strip()
    if cleaned_host in {"0.0.0.0", "::", ""}:
        return "127.0.0.1"
    return cleaned_host


def _detect_lan_ip() -> str | None:
    candidates: list[str] = []
    for lookup_name in {socket.gethostname(), socket.getfqdn()}:
        if not lookup_name:
            continue
        try:
            _, _, addresses = socket.gethostbyname_ex(lookup_name)
        except OSError:
            continue
        candidates.extend(addresses)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            candidates.append(sock.getsockname()[0])
    except OSError:
        pass

    for candidate in candidates:
        if candidate and not candidate.startswith("127.") and candidate != "0.0.0.0":
            return candidate
    return None


def _format_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _print_launch_summary(host: str, port: int, share_enabled: bool) -> None:
    local_host = _browser_access_host(host)
    lan_ip = _detect_lan_ip()
    lines = [
        "SONIC launch settings",
        f"  Host: {host}",
        f"  Port: {port}",
        f"  Share: {'on' if share_enabled else 'off'}",
        f"  Local URL: {_format_url(local_host, port)}",
    ]
    if lan_ip and lan_ip != local_host:
        lines.append(f"  LAN URL: {_format_url(lan_ip, port)}")

    print("\n".join(lines), flush=True)


def _launch_demo_with_fallback(demo: "gr.Blocks") -> int:
    preferred_port = resolve_launch_port()
    launch_host = resolve_launch_host()
    launch_share = resolve_launch_share()
    use_auto_port = auto_port_enabled()
    launch_port = (
        pick_available_port(preferred_port, host=launch_host)
        if use_auto_port
        else preferred_port
    )

    if launch_port != preferred_port:
        print(
            f"SONIC preferred port {preferred_port} is busy; auto-port selected {launch_port}.",
            flush=True,
        )

    _print_launch_summary(launch_host, launch_port, launch_share)

    try:
        demo.launch(
            server_name=launch_host,
            server_port=launch_port,
            inbrowser=True,
            share=launch_share,
            theme=gr.themes.Base(),
            css=SONIC_CSS,
            favicon_path=str(SONIC_FAVICON_PATH) if SONIC_FAVICON_PATH.exists() else None,
        )
        return launch_port
    except OSError as exc:
        message = str(exc).lower()
        if "port" not in message and "address" not in message:
            raise

        if not use_auto_port:
            raise RuntimeError(
                f"SONIC could not launch because http://{launch_host}:{preferred_port} is busy. "
                f"Close the process using port {preferred_port}, or rerun with `--auto-port` to let SONIC choose "
                "the next available port and print the exact URL."
            ) from exc

    upper_port = min(65535, preferred_port + PORT_FALLBACK_COUNT)
    for fallback_port in range(launch_port + 1, upper_port + 1):
        if not _port_is_available(fallback_port, host=launch_host):
            continue

        print(
            f"SONIC preferred port {launch_port} is busy; auto-port selected {fallback_port}.",
            flush=True,
        )
        _print_launch_summary(launch_host, fallback_port, launch_share)
        try:
            demo.launch(
                server_name=launch_host,
                server_port=fallback_port,
                inbrowser=True,
                share=launch_share,
                theme=gr.themes.Base(),
                css=SONIC_CSS,
                favicon_path=str(SONIC_FAVICON_PATH) if SONIC_FAVICON_PATH.exists() else None,
            )
            return fallback_port
        except OSError:
            continue

    raise RuntimeError(
        f"SONIC could not find an available port from {preferred_port} to {min(65535, preferred_port + PORT_FALLBACK_COUNT)}."
    )


def to_gradio_messages(history) -> list[dict[str, str]]:
    """
    Convert SONIC history into Gradio messages format:
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
    ]
    """
    result: list[dict[str, str]] = []
    if not history:
        return result

    for item in history:
        if isinstance(item, dict):
            if "role" in item and "content" in item:
                role = str(item.get("role") or "").strip().lower()
                content = str(item.get("content") or "")
                if role in {"user", "assistant"} and content:
                    result.append({"role": role, "content": content})
                continue

            user_msg = (
                item.get("user_message")
                or item.get("user")
                or item.get("prompt")
                or item.get("input")
                or ""
            )
            assistant_msg = (
                item.get("assistant_response")
                or item.get("assistant")
                or item.get("response")
                or item.get("output")
                or ""
            )

            if user_msg:
                result.append({"role": "user", "content": str(user_msg)})
            if assistant_msg:
                result.append({"role": "assistant", "content": str(assistant_msg)})
            continue

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user_msg, assistant_msg = item[0], item[1]
            if user_msg:
                result.append({"role": "user", "content": str(user_msg)})
            if assistant_msg:
                result.append({"role": "assistant", "content": str(assistant_msg)})

    return result


def _append_ui_history(
    history: list[dict[str, str]] | None,
    user_text: str,
    assistant_text: str,
) -> list[dict[str, str]]:
    updated_history = to_gradio_messages(history)
    updated_history.append({"role": "user", "content": user_text})
    updated_history.append({"role": "assistant", "content": assistant_text})
    return updated_history


def _append_backend_history(
    history: list[dict[str, str]] | None,
    user_text: str,
    assistant_text: str,
    metadata: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    updated_history = list(history or [])
    entry = {
        "user_message": user_text,
        "assistant_response": assistant_text,
    }
    if metadata:
        entry["metadata"] = dict(metadata)
    updated_history.append(entry)
    return updated_history


def _conversation_label(conversation: Mapping[str, object]) -> str:
    title = str(conversation.get("title") or UNTITLED_CHAT_TITLE).strip() or UNTITLED_CHAT_TITLE
    if len(title) > 42:
        title = title[:39].rstrip() + "..."
    return title


def _conversation_choice_pairs() -> list[tuple[str, str]]:
    return [
        (_conversation_label(conversation), str(conversation["id"]))
        for conversation in list_conversations()
        if isinstance(conversation.get("messages"), list) and conversation.get("messages")
    ]


def _conversation_selector_update(active_conversation_id: str):
    active_conversation = ensure_active_conversation(active_conversation_id)
    choices = _conversation_choice_pairs()
    visible_choice_ids = {choice_value for _, choice_value in choices}
    active_choice_id = str(active_conversation_id)
    return gr.update(
        choices=choices,
        value=active_choice_id if active_choice_id in visible_choice_ids else None,
    )


def _conversation_to_ui_messages(conversation: Mapping[str, object] | None) -> list[dict[str, str]]:
    if not conversation:
        return []

    messages = conversation.get("messages")
    if not isinstance(messages, list):
        return []

    ui_messages: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue

        role = str(message.get("role") or "").strip().lower()
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue

        if role == "assistant":
            display_content = str(message.get("display_content") or "").strip()
            metadata = message.get("metadata")
            language_code = ""
            if isinstance(metadata, Mapping):
                language_code = str(metadata.get("language_code") or "")
            content = display_content or _attach_tts_metadata(
                _format_reply_for_chat(content, "assistant"),
                language_code,
            )

        ui_messages.append({"role": role, "content": content})

    return ui_messages


def _conversation_to_backend_history(conversation: Mapping[str, object] | None) -> list[dict[str, object]]:
    if not conversation:
        return []

    messages = conversation.get("messages")
    if not isinstance(messages, list):
        return []

    return conversation_messages_to_backend_history(messages)


def _conversation_outputs(
    conversation: Mapping[str, object],
    runtime_preference: str,
    runtime_state: dict[str, str],
    message_value: str = "",
    image_update: object | None = None,
):
    ui_messages = _conversation_to_ui_messages(conversation)
    backend_history = _conversation_to_backend_history(conversation)
    has_messages = _workspace_has_content(conversation)
    header_html, sidebar_html, context_html, empty_html = _render_session_panels(
        runtime_preference=runtime_preference,
        runtime_state=runtime_state,
        has_messages=has_messages,
    )
    active_conversation_id = str(conversation["id"])
    return (
        ui_messages,
        message_value,
        header_html,
        sidebar_html,
        context_html,
        runtime_state,
        ui_messages,
        backend_history,
        has_messages,
        empty_html,
        image_update if image_update is not None else _conversation_image_update(conversation),
        _conversation_selector_update(active_conversation_id),
        active_conversation_id,
    )


def _handle_send(
    message: str,
    active_conversation_id: str | None,
    runtime_preference: str,
    runtime_state: dict[str, str],
    has_messages: bool,
    mode: str,
) -> tuple[
    list[dict[str, str]],
    str,
    str,
    str,
    str,
    dict[str, str],
    list[dict[str, str]],
    list[dict[str, str]],
    bool,
    str,
    object,
    str,
]:
    cleaned_message = _clean_message(message)
    selected_mode = _normalize_workflow_value(mode)
    current_conversation = ensure_active_conversation(active_conversation_id)
    current_ui_history = _conversation_to_ui_messages(current_conversation)
    current_backend_history = _conversation_to_backend_history(current_conversation)
    current_has_messages = _workspace_has_content(current_conversation) or bool(has_messages)

    if not cleaned_message:
        header_html, sidebar_html, context_html, empty_html = _render_session_panels(
            runtime_preference=runtime_preference,
            runtime_state=runtime_state,
            has_messages=current_has_messages,
        )
        ui_messages = to_gradio_messages(current_ui_history)
        return (
            ui_messages,
            "",
            header_html,
            sidebar_html,
            context_html,
            runtime_state,
            ui_messages,
            current_backend_history,
            current_has_messages,
            empty_html,
            _conversation_image_update(current_conversation),
            _conversation_selector_update(str(current_conversation["id"])),
            str(current_conversation["id"]),
        )

    if selected_mode == "image":
        logger.debug(
            "Image workflow started conversation_id=%s runtime=%s prompt=%s",
            current_conversation["id"],
            runtime_preference,
            cleaned_message,
        )
        try:
            image_result = generate_sonic_image(cleaned_message)
        except (ImageGenerationError, ValueError) as exc:
            logger.warning("Image generation failed: %s", exc)
            error_reply = "Image Generation failed.nvidia image service could not complete the request. please try again with a simpler prompt or switch the image model settings."
            display_error_reply = _attach_tts_metadata(error_reply, "en")
            updated_conversation = append_exchange(
                str(current_conversation["id"]),
                cleaned_message,
                error_reply,
                metadata={"language_code": "en"},
                display_response=display_error_reply,
            )
            error_state = {
                "runtime_used": runtime_state.get("runtime_used", "-"),
                "provider_used": "error",
                "model_used": runtime_state.get("model_used", DEFAULT_MODEL_LABEL),
                "requested_runtime": runtime_preference,
                "fallback_used": "false",
                "fallback_reason": "",
            }
            return _conversation_outputs(
                updated_conversation,
                runtime_preference,
                error_state,
                message_value="",
                image_update=gr.update(value=None, visible=False),
            )
        except Exception:
            logger.exception("Unexpected image generation failure")
            error_reply = "Error: Image generation failed unexpectedly. Please try again."
            display_error_reply = _attach_tts_metadata(error_reply, "en")
            updated_conversation = append_exchange(
                str(current_conversation["id"]),
                cleaned_message,
                error_reply,
                metadata={"language_code": "en"},
                display_response=display_error_reply,
            )
            error_state = {
                "runtime_used": runtime_state.get("runtime_used", "-"),
                "provider_used": "error",
                "model_used": runtime_state.get("model_used", DEFAULT_MODEL_LABEL),
                "requested_runtime": runtime_preference,
                "fallback_used": "false",
                "fallback_reason": "",
            }
            return _conversation_outputs(
                updated_conversation,
                runtime_preference,
                error_state,
                message_value="",
                image_update=gr.update(value=None, visible=False),
            )

        final_reply = f"Generated image for: {image_result.prompt}"
        display_reply = _compose_display_reply(final_reply, "Image preview ready.")
        display_reply_with_tts = _attach_tts_metadata(display_reply, "en")
        response_metadata = {
            "language_code": "en",
            "language_label": "English",
            "language_source": "image",
            "language_confidence": "1.0",
            "runtime_used": "online",
            "provider_used": image_result.provider_used,
            "model_used": image_result.model_used,
            "notice": "Image preview ready.",
            "image_path": image_result.image_path,
            "image_url": image_result.image_url,
            "api_endpoint": image_result.api_endpoint,
            "image_prompt": image_result.prompt,
        }
        updated_conversation = append_exchange(
            str(current_conversation["id"]),
            cleaned_message,
            final_reply,
            metadata=response_metadata,
            display_response=display_reply_with_tts,
        )
        updated_state = {
            "runtime_used": "online",
            "provider_used": image_result.provider_used,
            "model_used": image_result.model_used,
            "requested_runtime": runtime_preference or "-",
            "fallback_used": "false",
            "fallback_reason": "",
            "language_code": "en",
            "language_label": "English",
        }
        return _conversation_outputs(
            updated_conversation,
            runtime_preference,
            updated_state,
            message_value="",
        )

    try:
        sonic_response = ask_sonic(
            message=cleaned_message,
            mode="assistant",
            history=current_backend_history,
            runtime_preference=runtime_preference,
        )
    except Exception as exc:
        error_reply = f"Error: {exc}"
        display_error_reply = _attach_tts_metadata(error_reply, "en")
        updated_conversation = append_exchange(
            str(current_conversation["id"]),
            cleaned_message,
            error_reply,
            metadata={"language_code": "en"},
            display_response=display_error_reply,
        )
        error_state = {
            "runtime_used": runtime_state.get("runtime_used", "-"),
            "provider_used": "error",
            "model_used": runtime_state.get("model_used", DEFAULT_MODEL_LABEL),
            "requested_runtime": runtime_preference,
            "fallback_used": "false",
            "fallback_reason": "",
        }
        return _conversation_outputs(
            updated_conversation,
            runtime_preference,
            error_state,
            message_value="",
        )

    final_reply = sonic_response["reply"]
    if str(sonic_response["runtime_used"]).lower() in {"online", "live_web"}:
        final_reply = sanitize_live_web_output(final_reply, query=cleaned_message) or (
            "I couldn't find fresh results for that query."
        )

    assistant_reply = _format_reply_for_chat(final_reply, "assistant")
    display_reply = _compose_display_reply(
        assistant_reply,
        str(sonic_response.get("notice") or ""),
    )
    display_reply_with_tts = _attach_tts_metadata(
        display_reply,
        str(sonic_response.get("language_code") or ""),
    )
    response_metadata = {
        "language_code": str(sonic_response.get("language_code") or ""),
        "language_label": str(sonic_response.get("language_label") or ""),
        "language_source": str(sonic_response.get("language_source") or ""),
        "language_confidence": str(sonic_response.get("language_confidence") or ""),
        "runtime_used": str(sonic_response.get("runtime_used") or ""),
        "provider_used": str(sonic_response.get("provider_used") or ""),
        "model_used": str(sonic_response.get("model_used") or ""),
        "notice": str(sonic_response.get("notice") or ""),
    }
    updated_conversation = append_exchange(
        str(current_conversation["id"]),
        cleaned_message,
        final_reply,
        metadata=response_metadata,
        display_response=display_reply_with_tts,
    )
    updated_state = {
        "runtime_used": sonic_response["runtime_used"],
        "provider_used": sonic_response["provider_used"],
        "model_used": sonic_response["model_used"],
        "requested_runtime": str(sonic_response.get("requested_runtime") or runtime_preference or "-"),
        "fallback_used": "true" if sonic_response.get("fallback_used") else "false",
        "fallback_reason": str(sonic_response.get("fallback_reason") or ""),
        "language_code": str(sonic_response.get("language_code") or ""),
        "language_label": str(sonic_response.get("language_label") or ""),
    }
    return _conversation_outputs(
        updated_conversation,
        runtime_preference,
        updated_state,
        message_value="",
    )


def _handle_clear(
    runtime_preference: str,
    active_conversation_id: str | None,
) -> tuple[
    list[dict[str, str]],
    str,
    str,
    str,
    str,
    dict[str, str],
    list[dict[str, str]],
    list[dict[str, str]],
    bool,
    str,
    object,
    str,
]:
    conversation = clear_conversation(active_conversation_id)
    runtime_state = _initial_runtime_state()
    return _conversation_outputs(
        conversation,
        runtime_preference,
        runtime_state,
        message_value="",
    )


def _handle_new_chat(
    runtime_preference: str,
):
    conversation = create_conversation()
    return _conversation_outputs(
        conversation,
        runtime_preference,
        _initial_runtime_state(),
        message_value="",
    )
def _handle_select_conversation(
    conversation_id: str | None,
    runtime_preference: str,
    runtime_state: dict[str, str],
):
    conversation = set_active_conversation(str(conversation_id or ""))
    return _conversation_outputs(
        conversation,
        runtime_preference,
        runtime_state,
        message_value="",
    )


def _handle_delete_conversation(
    conversation_id: str | None,
    runtime_preference: str,
    runtime_state: dict[str, str],
):
    conversation = delete_conversation(conversation_id)
    if conversation is None:
        conversation = ensure_active_conversation()
    return _conversation_outputs(
        conversation,
        runtime_preference,
        runtime_state,
        message_value="",
    )


def create_ui() -> "gr.Blocks":
    if gr is None:
        raise RuntimeError(
            "Gradio is not installed. Run `pip install -r requirements.txt` first."
        )

    initial_conversation = ensure_active_conversation()
    initial_ui_messages = _conversation_to_ui_messages(initial_conversation)
    initial_backend_history = _conversation_to_backend_history(initial_conversation)
    initial_has_messages = bool(initial_ui_messages)
    initial_image_path = _conversation_image_path(initial_conversation)
    initial_runtime_state = _initial_runtime_state()
    initial_selector_value = str(initial_conversation["id"]) if initial_has_messages else None

    with gr.Blocks(
        title=f"{APP_NAME} | Offline-first AI workspace",
        analytics_enabled=False,
        fill_width=True,
        fill_height=True,
    ) as demo:
        runtime_state = gr.State(initial_runtime_state)
        ui_history_state = gr.State(initial_ui_messages)
        backend_history_state = gr.State(initial_backend_history)
        has_messages_state = gr.State(initial_has_messages)
        active_conversation_state = gr.State(str(initial_conversation["id"]))

        with gr.Column(elem_id="sonic-app"):
            # 1. TOP NAVBAR
            with gr.Row(elem_id="sonic-navbar"):
                logo_and_brand = gr.HTML(
                    f"""
                    <div class="sonic-navbar-brand">
                        {_render_sonic_mark_html("small")}
                        <div class="sonic-navbar-wordmark">SONIC</div>
                    </div>
                    """,
                    elem_id="sonic-navbar-brand-host",
                )
                with gr.Row(elem_id="sonic-navbar-controls"):
                    # Runtime selector as horizontal radio pills on the right
                    runtime_dropdown = gr.Radio(
                        choices=RUNTIME_CHOICES,
                        value=DEFAULT_RUNTIME_SELECTION,
                        show_label=False,
                        container=False,
                        elem_id="sonic-runtime-dropdown",
                    )
                    workflow_selector = gr.Radio(
                        choices=WORKFLOW_CHOICES,
                        value=DEFAULT_WORKFLOW_SELECTION,
                        show_label=False,
                        container=False,
                        elem_id="sonic-workflow-dropdown",
                    )

            # 2. MAIN SHELL (Sidebar + Chat Stage)
            with gr.Row(elem_id="sonic-shell"):
                # Left Sidebar actions
                with gr.Column(elem_id="sonic-left-sidebar", scale=1, min_width=180):
                    with gr.Column(elem_id="sonic-sidebar-top-controls"):
                        new_chat_button = gr.Button(
                            "New chat",
                            variant="primary",
                            elem_id="sonic-new-chat-button",
                        )
                        clear_button = gr.Button(
                            "Clear chat",
                            variant="secondary",
                            elem_id="sonic-clear-chat-button",
                        )
                        voice_output_toggle = gr.Checkbox(
                            value=VOICE_OUTPUT_DEFAULT,
                            label="Voice output",
                            show_label=True,
                            interactive=True,
                            elem_id="sonic-voice-toggle",
                        )
                    with gr.Column(elem_id="sonic-sidebar-list-shell"):
                        gr.HTML(
                            '<div class="sonic-history-heading">Chats</div>',
                            elem_id="sonic-history-heading-host",
                        )
                        chat_selector = gr.Radio(
                            choices=_conversation_choice_pairs(),
                            value=initial_selector_value,
                            show_label=False,
                            container=False,
                            elem_id="sonic-chat-history-list",
                        )
                    with gr.Column(elem_id="sonic-sidebar-footer"):
                        delete_chat_button = gr.Button(
                            "Delete selected chat",
                            variant="secondary",
                            elem_id="sonic-delete-chat-button",
                        )

                # Center Chat Workspace
                with gr.Column(elem_id="sonic-chat-stage", scale=4):
                    with gr.Column(elem_id="sonic-chat-scroll"):
                        chatbot = gr.Chatbot(
                            value=initial_ui_messages,
                            label="Conversation",
                            show_label=False,
                            height="100%",
                            max_height="100%",
                            min_height=0,
                            layout="bubble",
                            render_markdown=True,
                            line_breaks=True,
                            autoscroll=True,
                            elem_id="sonic-chatbot",
                            placeholder="",
                        )
                        empty_state_panel = gr.HTML(
                            _render_empty_state_html(
                                DEFAULT_RUNTIME_SELECTION,
                                initial_runtime_state,
                                has_messages=initial_has_messages,
                            ),
                            elem_id="sonic-empty-state-host",
                        )
                        generated_image_panel = gr.Image(
                            value=initial_image_path or None,
                            label="Generated image",
                            show_label=True,
                            type="filepath",
                            interactive=False,
                            visible=bool(initial_image_path),
                            elem_id="sonic-generated-image",
                        )

                    gr.HTML(_render_composer_hint_html(), elem_id="sonic-composer-meta")

                    with gr.Column(elem_id="sonic-composer-stack"):
                        with gr.Row(elem_id="sonic-composer"):
                            with gr.Column(elem_id="sonic-composer-card"):
                                with gr.Row(elem_id="sonic-composer-row"):
                                    with gr.Column(elem_id="sonic-composer-input"):
                                        message_box = gr.Textbox(
                                            placeholder=_mode_placeholder(DEFAULT_WORKFLOW_SELECTION),
                                            lines=1,
                                            label="Message",
                                            show_label=False,
                                            container=False,
                                            elem_id="sonic-message-box",
                                        )
                                    with gr.Column(elem_id="sonic-composer-send"):
                                        send_button = gr.Button(
                                            ">",
                                            variant="primary",
                                            elem_id="sonic-send-button",
                                        )

            # Invisible panels for backward compatibility of _handle_send returns
            header_panel = gr.HTML("", visible=False, elem_id="sonic-app-header-host")
            sidebar_status_panel = gr.HTML("", visible=False)
            context_panel = gr.HTML("", visible=False, elem_id="sonic-context-host")

        conversation_outputs = [
            chatbot,
            message_box,
            header_panel,
            sidebar_status_panel,
            context_panel,
            runtime_state,
            ui_history_state,
            backend_history_state,
            has_messages_state,
            empty_state_panel,
            generated_image_panel,
            chat_selector,
            active_conversation_state,
        ]

        clear_outputs = [
            chatbot,
            message_box,
            header_panel,
            sidebar_status_panel,
            context_panel,
            runtime_state,
            ui_history_state,
            backend_history_state,
            has_messages_state,
            empty_state_panel,
            generated_image_panel,
            chat_selector,
            active_conversation_state,
        ]

        def refresh_status(
            runtime_preference: str,
            current_state: dict[str, str],
            has_messages: bool,
        ) -> tuple[str, str, str, str]:
            return _render_session_panels(
                runtime_preference,
                current_state,
                has_messages,
            )

        def refresh_message_placeholder(selected_mode: str) -> object:
            return gr.update(placeholder=_mode_placeholder(selected_mode))

        send_button.click(
            fn=_handle_send,
            inputs=[
                message_box,
                active_conversation_state,
                runtime_dropdown,
                runtime_state,
                has_messages_state,
                workflow_selector,
            ],
            outputs=conversation_outputs,
        )
        message_box.submit(
            fn=_handle_send,
            inputs=[
                message_box,
                active_conversation_state,
                runtime_dropdown,
                runtime_state,
                has_messages_state,
                workflow_selector,
            ],
            outputs=conversation_outputs,
        )

        clear_button.click(
            fn=_handle_clear,
            inputs=[runtime_dropdown, active_conversation_state],
            outputs=clear_outputs,
        )
        new_chat_button.click(
            fn=_handle_new_chat,
            inputs=[runtime_dropdown],
            outputs=clear_outputs,
        )
        chat_selector.change(
            fn=_handle_select_conversation,
            inputs=[chat_selector, runtime_dropdown, runtime_state],
            outputs=clear_outputs,
        )
        delete_chat_button.click(
            fn=_handle_delete_conversation,
            inputs=[chat_selector, runtime_dropdown, runtime_state],
            outputs=clear_outputs,
        )
        runtime_dropdown.change(
            fn=refresh_status,
            inputs=[runtime_dropdown, runtime_state, has_messages_state],
            outputs=[header_panel, sidebar_status_panel, context_panel, empty_state_panel],
        )
        workflow_selector.change(
            fn=refresh_message_placeholder,
            inputs=[workflow_selector],
            outputs=[message_box],
        )
        demo.load(
            fn=lambda: None,
            js=SONIC_TTS_JS,
            queue=False,
            api_name=False,
            outputs=[],
        )

    return demo


def main() -> None:
    demo = create_ui()
    demo.queue()
    _launch_demo_with_fallback(demo)


if __name__ == "__main__":
    main()
