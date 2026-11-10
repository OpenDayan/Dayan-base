#!/usr/bin/env python3
#***************************************************************************************
# Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
#
# Dayan-base is licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#
# See the Mulan PSL v2 for more details.
#***************************************************************************************

"""Single-file NPU SRAM swizzle visualizer and local HTTP server."""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Tuple


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VisualSwizzle</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f3f5f7; --surface: #ffffff; --surface-alt: #f8fafb;
      --ink: #18212a; --muted: #65717d; --line: #d8dee4;
      --line-strong: #aab4be; --accent: #176b87; --good: #18794e;
      --bad: #b42318; --shadow: 0 1px 3px rgba(22, 33, 43, 0.08);
    }
    * { box-sizing: border-box; letter-spacing: 0; }
    html, body { margin: 0; min-height: 100%; background: var(--bg); color: var(--ink); }
    body {
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                   "Segoe UI", sans-serif;
      font-size: 13px;
    }
    button, input, select { font: inherit; color: inherit; }
    button { cursor: pointer; }
    .app { min-height: 100vh; display: flex; flex-direction: column; }
    .topbar {
      min-height: 58px; display: flex; align-items: center; gap: 18px;
      padding: 9px 16px; background: var(--surface); border-bottom: 1px solid var(--line);
      position: relative; z-index: 5;
    }
    .brand { min-width: 220px; }
    .brand h1 { margin: 0; font-size: 20px; line-height: 1.1; font-weight: 720; }
    .brand p { margin: 3px 0 0; color: var(--muted); font-size: 11px; }
    .topbar-spacer { flex: 1; }
    .mode-switch, .width-switch {
      display: inline-flex; border: 1px solid var(--line-strong); border-radius: 5px;
      overflow: hidden; background: var(--surface);
    }
    .mode-switch button, .width-switch button {
      min-height: 32px; border: 0; border-right: 1px solid var(--line);
      padding: 0 12px; background: transparent; white-space: nowrap;
    }
    .mode-switch button:last-child, .width-switch button:last-child { border-right: 0; }
    .mode-switch button.active, .width-switch button.active {
      background: var(--accent); color: #fff; font-weight: 650;
    }
    .command {
      min-height: 34px; border: 1px solid #10566d; border-radius: 5px;
      padding: 0 16px; background: var(--accent); color: #fff; font-weight: 650;
    }
    .command:hover { background: #105b75; }
    .command.secondary { background: var(--surface); color: var(--ink); border-color: var(--line-strong); }
    .command.secondary:hover { background: var(--surface-alt); }
    .shell { flex: 1; min-height: 0; display: flex; }
    .controls {
      width: 286px; flex: 0 0 286px; background: var(--surface);
      border-right: 1px solid var(--line); overflow-y: auto; padding: 14px 14px 20px;
    }
    .control-group { border: 0; border-top: 1px solid var(--line); margin: 14px 0 0; padding: 14px 0 0; }
    .control-group:first-child { border-top: 0; margin-top: 0; padding-top: 0; }
    .control-group legend {
      width: 100%; padding: 0 0 9px; color: #34414c; font-size: 11px;
      font-weight: 750; text-transform: uppercase;
    }
    .field { display: grid; grid-template-columns: minmax(0, 1fr) 112px; align-items: center; gap: 10px; margin-bottom: 8px; }
    .field label { color: #3d4852; min-width: 0; }
    .field input, .field select {
      width: 100%; min-height: 31px; border: 1px solid var(--line-strong);
      border-radius: 4px; background: #fff; padding: 4px 8px; font-variant-numeric: tabular-nums;
    }
    .field input:focus, .field select:focus { outline: 2px solid #8bc5d6; outline-offset: 1px; }
    .fixed-value {
      min-height: 31px; display: flex; align-items: center; padding: 4px 8px;
      border: 1px solid var(--line); border-radius: 4px; background: var(--surface-alt);
      color: var(--muted); font-variant-numeric: tabular-nums;
    }
    .wide-field { display: block; margin-bottom: 10px; }
    .wide-field label { display: block; margin-bottom: 6px; color: #3d4852; }
    .wide-field .width-switch { width: 100%; }
    .wide-field .width-switch button { flex: 1; padding: 0 4px; }
    .cycle-stepper { display: grid; grid-template-columns: 31px 1fr 31px; gap: 5px; }
    .cycle-stepper button {
      border: 1px solid var(--line-strong); border-radius: 4px; background: var(--surface); font-weight: 700;
    }
    .cycle-stepper button:hover { background: var(--surface-alt); }
    .cycle-stepper input { min-width: 0; text-align: center; }
    .control-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 14px; }
    .control-actions .command { width: 100%; padding: 0 8px; }
    .search-command { width: 100%; margin-top: 4px; }
    .search-result {
      display: none; margin-top: 9px; padding: 9px; border: 1px solid var(--line-strong);
      border-radius: 5px; background: var(--surface-alt);
    }
    .search-result.visible { display: block; }
    .search-result-head { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
    .search-result-head strong { font-size: 14px; font-variant-numeric: tabular-nums; }
    .search-result-head span { color: var(--good); font-size: 10px; font-weight: 750; }
    .search-result-meta {
      margin: 5px 0 8px; color: var(--muted); font: 9px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: pre-line;
    }
    .search-result .command { width: 100%; min-height: 30px; padding: 0 8px; }
    .search-result .command:disabled { cursor: default; background: #d8dee4; border-color: #c4ccd3; color: #52606b; }
    .workspace {
      min-width: 0; flex: 1; display: grid;
      grid-template-rows: auto minmax(310px, 1fr) minmax(235px, 0.62fr);
      gap: 10px; padding: 10px; overflow: hidden;
    }
    .metrics {
      min-height: 57px; display: grid;
      grid-template-columns: repeat(5, minmax(105px, 1fr));
      background: var(--surface); border: 1px solid var(--line); border-radius: 6px; box-shadow: var(--shadow);
    }
    .metric { min-width: 0; padding: 9px 12px; border-right: 1px solid var(--line); }
    .metric:last-child { border-right: 0; }
    .metric:nth-child(5) { border-right: 0; }
    .metric-label { display: block; color: var(--muted); font-size: 10px; text-transform: uppercase; }
    .metric[data-help] { cursor: help; }
    .metric[data-help] .metric-label::after {
      content: 'i'; display: inline-flex; align-items: center; justify-content: center;
      width: 13px; height: 13px; margin-left: 5px; border: 1px solid #95a2ad;
      border-radius: 50%; color: #64717d; font-size: 8px; font-weight: 800;
      text-transform: none; vertical-align: 1px;
    }
    .metric-value {
      display: block; margin-top: 3px; font-size: 18px; font-weight: 720; line-height: 1;
      font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .metric-value.small { font-size: 12px; line-height: 1.35; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .expression-metric {
      grid-column: 1 / -1; min-height: 42px; display: flex; align-items: center; gap: 12px;
      border-top: 1px solid var(--line); border-right: 0; padding-top: 6px; padding-bottom: 6px;
    }
    .expression-metric .metric-label { flex: 0 0 190px; }
    .bit-rules { min-width: 0; flex: 1; display: flex; align-items: center; flex-wrap: wrap; row-gap: 4px; }
    .bit-rule {
      padding: 0 7px; border-left: 1px solid var(--line); color: #53616d;
      font: 9px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: nowrap;
    }
    .bit-rule:first-child { border-left: 0; padding-left: 0; }
    .bit-rule.changed { color: var(--accent); font-weight: 800; }
    .metric-value.good { color: var(--good); }
    .metric-value.bad { color: var(--bad); }
    .panel {
      min-width: 0; min-height: 0; display: flex; flex-direction: column;
      background: var(--surface); border: 1px solid var(--line); border-radius: 6px;
      box-shadow: var(--shadow); overflow: hidden;
    }
    .panel-head {
      min-height: 43px; display: flex; align-items: center; gap: 10px;
      padding: 8px 11px; border-bottom: 1px solid var(--line); background: var(--surface-alt);
    }
    .panel-title { min-width: 0; flex: 1; }
    .panel-title h2 { margin: 0; font-size: 13px; font-weight: 720; }
    .panel-title p { margin: 2px 0 0; color: var(--muted); font-size: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .panel-stat { color: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; white-space: nowrap; }
    .canvas-scroll { flex: 1; min-height: 0; overflow: auto; position: relative; background: #fff; }
    canvas { display: block; }
    .lower-grid {
      min-width: 0; min-height: 0; display: grid;
      grid-template-columns: minmax(220px, 0.8fr) minmax(390px, 1.8fr) minmax(250px, 1fr); gap: 10px;
    }
    .conflict-scroll, .pressure-wrap { flex: 1; min-height: 0; overflow: auto; position: relative; }
    .cycle-body { flex: 1; min-height: 0; overflow: auto; padding: 9px 10px 11px; }
    .cycle-grid { min-width: 330px; display: grid; gap: 5px; }
    .lane-row { display: grid; gap: 5px; min-height: 52px; }
    .lane-label {
      display: flex; align-items: center; justify-content: center; border-right: 1px solid var(--line);
      color: var(--muted); font-size: 10px; font-weight: 700; font-variant-numeric: tabular-nums;
    }
    .transaction {
      min-width: 0; border: 1px solid rgba(19, 28, 35, 0.32); border-radius: 4px;
      padding: 5px 8px; overflow: hidden;
    }
    .transaction-main { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
    .transaction-bank { font-size: 15px; font-weight: 780; font-variant-numeric: tabular-nums; }
    .transaction-word { font-size: 9px; font-weight: 700; opacity: 0.82; }
    .transaction-meta {
      margin-top: 3px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 9px; line-height: 1.2; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .error-banner {
      display: none; position: fixed; z-index: 20; left: 50%; top: 68px; transform: translateX(-50%);
      max-width: min(620px, calc(100vw - 24px)); padding: 9px 13px; border: 1px solid #d92d20;
      border-radius: 5px; background: #fff1f0; color: #8a1c13; box-shadow: 0 6px 18px rgba(76, 20, 15, 0.18);
    }
    .error-banner.visible { display: block; }
    .tooltip {
      display: none; position: fixed; z-index: 30; pointer-events: none; max-width: 340px;
      padding: 8px 10px; border: 1px solid #263744; border-radius: 4px; background: #16232d;
      color: #f7fafc; box-shadow: 0 5px 16px rgba(0, 0, 0, 0.22);
      font: 10px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: pre-wrap; overflow-wrap: anywhere;
    }
    @media (max-width: 1220px) {
      .metrics { grid-template-columns: repeat(3, 1fr); }
      .metric:nth-child(3) { border-right: 0; }
      .metric:nth-child(n+4) { border-top: 1px solid var(--line); }
    }
    @media (max-width: 1120px) {
      .workspace { overflow: auto; grid-template-rows: auto 390px auto; }
      .lower-grid { grid-template-columns: minmax(220px, 0.8fr) minmax(390px, 1.6fr); grid-template-rows: 245px 230px; }
      .pressure-panel { grid-column: 1 / -1; }
    }
    @media (max-width: 760px) {
      .topbar { align-items: flex-start; flex-wrap: wrap; gap: 9px; }
      .brand { min-width: 0; flex: 1 1 180px; }
      .topbar-spacer { display: none; }
      .mode-switch { order: 3; width: 100%; }
      .mode-switch button { flex: 1; }
      .shell { display: block; }
      .controls { width: 100%; height: auto; border-right: 0; border-bottom: 1px solid var(--line); overflow: visible; }
      #controlsForm { display: grid; grid-template-columns: 1fr 1fr; gap: 0 18px; }
      .control-actions { align-self: end; }
      .workspace { display: block; padding: 8px; overflow: visible; }
      .metrics { margin-bottom: 8px; grid-template-columns: 1fr 1fr; }
      .metric { border-top: 1px solid var(--line); }
      .metric:nth-child(-n+2) { border-top: 0; }
      .metric:nth-child(even) { border-right: 0; }
      .expression-metric { flex-direction: column; align-items: flex-start; gap: 4px; }
      .expression-metric .metric-label { flex: none; }
      .panel { min-height: 330px; margin-bottom: 8px; }
      .lower-grid { display: block; }
      .conflict-panel, .cycle-panel, .pressure-panel { min-height: 270px; }
    }
    @media (max-width: 480px) {
      #controlsForm { display: block; }
      .metrics { grid-template-columns: 1fr; }
      .metric { border-right: 0; border-top: 1px solid var(--line); }
      .metric:first-child { border-top: 0; }
      .topbar .command.secondary { display: none; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="brand"><h1>VisualSwizzle</h1><p>NPU SRAM / 字节地址模型</p></div>
      <div class="mode-switch" id="modeSwitch" role="group" aria-label="地址映射模式">
        <button type="button" data-mode="linear">线性映射</button>
        <button type="button" data-mode="swizzle" class="active">NPU Swizzle</button>
      </div>
      <div class="topbar-spacer"></div>
      <button type="button" class="command secondary" id="resetButton">重置</button>
      <button type="button" class="command" id="applyButton">应用</button>
    </header>
    <div class="shell">
      <aside class="controls">
        <form id="controlsForm">
          <fieldset class="control-group">
            <legend>张量布局</legend>
            <div class="wide-field">
              <label>元素位宽</label>
              <div class="width-switch" id="widthSwitch" role="group" aria-label="元素位宽">
                <button type="button" data-bits="32" class="active">4 B</button>
                <button type="button" data-bits="16">2 B</button>
                <button type="button" data-bits="8">1 B</button>
                <button type="button" data-bits="4">0.5 B</button>
              </div>
            </div>
            <div class="field"><label for="rows">行数</label><input id="rows" type="number" min="1" max="256" value="32"></div>
            <div class="field"><label for="rowBytes">每行数据 (B)</label><input id="rowBytes" type="number" min="32" step="32" value="256"></div>
            <div class="field"><label for="strideBytes">行跨度 (B)</label><input id="strideBytes" type="number" min="32" step="32" value="256"></div>
            <div class="field"><label for="baseAddress">起始地址</label><input id="baseAddress" type="text" value="0x00000000" spellcheck="false"></div>
            <div class="field"><label>每行元素数</label><span class="fixed-value" id="elementsPerRow">64</span></div>
            <div class="field"><label>每拍访问</label><span class="fixed-value" id="cycleShape">8 x 32 B</span></div>
          </fieldset>
          <fieldset class="control-group">
            <legend>SRAM</legend>
            <div class="field"><label for="numBanks">Bank 数</label><input id="numBanks" type="number" min="1" max="64" value="32"></div>
            <div class="field"><label>Bank 字宽</label><span class="fixed-value">32 B</span></div>
            <div class="field"><label>局部区域</label><span class="fixed-value">32 KiB</span></div>
          </fieldset>
          <fieldset class="control-group">
            <legend>Swizzle 参数</legend>
            <div class="field">
              <label for="preset">预设</label>
              <select id="preset">
                <option value="3,1,7">局部均衡 M7/B1/S3</option>
                <option value="1,3,7">均衡 M7/B3/S1</option>
                <option value="3,5,5">S3 / B5 / M5</option>
                <option value="5,5,5">S5 / B5 / M5</option>
                <option value="3,3,5">S3 / B3 / M5</option>
                <option value="custom">自定义</option>
              </select>
            </div>
            <div class="field"><label for="shiftS">S</label><input id="shiftS" type="number" min="0" max="15" value="3"></div>
            <div class="field"><label for="widthB">B</label><input id="widthB" type="number" min="0" max="15" value="1"></div>
            <div class="field"><label for="maskM">M</label><input id="maskM" type="number" min="0" max="15" value="7"></div>
            <button type="button" class="command secondary search-command" id="searchButton">搜索最优 M/B/S</button>
            <div class="search-result" id="searchResult" aria-live="polite">
              <div class="search-result-head"><strong id="searchResultConfig">-</strong><span id="searchResultStatus">-</span></div>
              <div class="search-result-meta" id="searchResultMeta">-</div>
              <button type="button" class="command" id="useSearchResult">采用此方案</button>
            </div>
          </fieldset>
          <fieldset class="control-group">
            <legend>当前拍</legend>
            <div class="field">
              <label for="rowCycle">行组</label>
              <div class="cycle-stepper">
                <button type="button" data-step="row,-1" title="上一行组" aria-label="上一行组">&lt;</button>
                <input id="rowCycle" type="number" min="0" value="0">
                <button type="button" data-step="row,1" title="下一行组" aria-label="下一行组">&gt;</button>
              </div>
            </div>
            <div class="field">
              <label for="colCycle">列组</label>
              <div class="cycle-stepper">
                <button type="button" data-step="col,-1" title="上一列组" aria-label="上一列组">&lt;</button>
                <input id="colCycle" type="number" min="0" value="0">
                <button type="button" data-step="col,1" title="下一列组" aria-label="下一列组">&gt;</button>
              </div>
            </div>
          </fieldset>
          <div class="control-actions">
            <button type="button" class="command secondary" id="sideResetButton">重置</button>
            <button type="submit" class="command">应用</button>
          </div>
        </form>
      </aside>
      <main class="workspace">
        <section class="metrics" aria-label="当前拍摘要">
          <div class="metric" tabindex="0" data-help="Swizzle 前，当前拍落到同一 Bank 的 32B 请求峰值。1x 表示无冲突；2x 表示有两个请求落到同一 Bank。"><span class="metric-label">线性峰值</span><span class="metric-value" id="linearPeak">-</span></div>
          <div class="metric" tabindex="0" data-help="Swizzle 后，当前拍落到同一 Bank 的 32B 请求峰值。1x 表示每条请求都使用不同的 Bank，没有冲突。"><span class="metric-label">Swizzle 峰值</span><span class="metric-value" id="swizzlePeak">-</span></div>
          <div class="metric" tabindex="0" data-help="相对线性映射，额外 Bank 请求数的变化。-4 表示 Swizzle 消除了 4 条需要额外 Bank 服务的请求；正数表示冲突变差。"><span class="metric-label">额外请求变化</span><span class="metric-value" id="conflictDelta">-</span></div>
          <div class="metric" tabindex="0" data-help="任意水平连续四个 8 x 32B 块所覆盖的最少 Bank 数。32/32 通过表示 32 条请求恰好各访问一个 Bank。"><span class="metric-label">连续四块覆盖</span><span class="metric-value" id="fourBlockCoverage">-</span></div>
          <div class="metric" tabindex="0" data-help="当前拍发出的 32B SRAM Bank-word 请求数。8 x 32B 产生 8 条请求；8 x 64B 产生 16 条。"><span class="metric-label">SRAM 请求数</span><span class="metric-value" id="transactionCount">-</span></div>
          <div class="metric expression-metric">
            <span class="metric-label">有效字节地址逐位规则</span>
            <div class="bit-rules" id="expression">-</div>
          </div>
        </section>
        <section class="panel map-panel">
          <div class="panel-head">
            <div class="panel-title"><h2>SRAM Bank 映射</h2><p id="mapSubtitle">-</p></div>
            <div class="panel-stat" id="mapStat">-</div>
          </div>
          <div class="canvas-scroll" id="mapScroll"><canvas id="mapCanvas"></canvas></div>
        </section>
        <div class="lower-grid">
          <section class="panel conflict-panel">
            <div class="panel-head">
              <div class="panel-title"><h2>每拍冲突图</h2><p id="conflictSubtitle">-</p></div>
              <div class="panel-stat" id="globalPeak">-</div>
            </div>
            <div class="conflict-scroll" id="conflictScroll"><canvas id="conflictCanvas"></canvas></div>
          </section>
          <section class="panel cycle-panel">
            <div class="panel-head">
              <div class="panel-title"><h2>当前拍请求</h2><p id="cycleSubtitle">-</p></div>
              <div class="panel-stat" id="cycleStat">-</div>
            </div>
            <div class="cycle-body"><div class="cycle-grid" id="cycleGrid"></div></div>
          </section>
          <section class="panel pressure-panel">
            <div class="panel-head">
              <div class="panel-title"><h2>Bank 压力</h2><p id="pressureSubtitle">-</p></div>
              <div class="panel-stat" id="pressureStat">-</div>
            </div>
            <div class="pressure-wrap" id="pressureWrap"><canvas id="pressureCanvas"></canvas></div>
          </section>
        </div>
      </main>
    </div>
  </div>
  <div class="error-banner" id="errorBanner" role="alert"></div>
  <div class="tooltip" id="tooltip"></div>
  <script>
    'use strict';
    const DEFAULTS = Object.freeze({
      mode: 'swizzle', elementBits: 32, rows: 32, rowBytes: 256,
      strideBytes: 256, baseAddress: '0x00000000', numBanks: 32,
      s: 3, b: 1, m: 7, rowCycle: 0, colCycle: 0
    });
    const BANK_WORD_BYTES = 32;
    const ROWS_PER_CYCLE = 8;
    const MAX_MASK_BITS = 8;
    const palette = [
      '#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2', '#edc948', '#b07aa1', '#ff9da7',
      '#2f6690', '#d66b20', '#328a62', '#c44545', '#3a9d9a', '#c99a16', '#8f62a8', '#dc7182',
      '#6b8fc3', '#efad55', '#79ad67', '#dd7072', '#55a8aa', '#e4c75b', '#a986bb', '#f38e9c',
      '#3f5f86', '#a95319', '#26704d', '#9f3638', '#287e7b', '#a87b0c', '#704886', '#b85669'
    ];
    const dom = {};
    const state = {
      config: null, mode: DEFAULTS.mode, mapCells: [], conflictCells: [],
      searchResult: null, resizeTimer: null
    };
    function $(id) { return document.getElementById(id); }
    function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }
    function intValue(id) { return Number.parseInt($(id).value, 10); }
    function isPowerOfTwo(value) { return value > 0 && (value & (value - 1)) === 0; }
    function accessBytes(elementBits) { return elementBits === 4 ? 64 : 32; }
    function hex(value, minDigits = 8) {
      const raw = value.toString(16).toUpperCase();
      return '0x' + raw.padStart(Math.max(minDigits, raw.length), '0');
    }
    function bankColor(bank) { return palette[bank % palette.length]; }
    function foreground(hexColor) {
      const value = Number.parseInt(hexColor.slice(1), 16);
      const r = (value >> 16) & 255, g = (value >> 8) & 255, b = value & 255;
      return (r * 299 + g * 587 + b * 114) / 1000 > 148 ? '#14202a' : '#ffffff';
    }
    function conflictColor(peak) {
      if (peak <= 1) return '#cce8d9';
      if (peak === 2) return '#f4dc82';
      if (peak === 3) return '#efad55';
      if (peak === 4) return '#df785c';
      if (peak <= 7) return '#ca4b43';
      return '#9f2d2d';
    }
    function swizzleAddress(address, config) {
      if (address < 0n) throw new Error('地址不能为负数。');
      const s = config.s & 0xF, b = config.b & 0xF, m = config.m & 0xF;
      const local = (address >> 5n) & 0x3FFn;
      const shifted = local >> BigInt(s);
      const lowMask = ((1n << BigInt(b)) - 1n) & ((1n << BigInt(MAX_MASK_BITS)) - 1n);
      const maskShift = (m - 5) & 0xF;
      const mask = (lowMask << BigInt(maskShift)) & 0x3FFn;
      const swizzledLocal = local ^ (shifted & mask);
      return (address & ~(0x3FFn << 5n)) | (swizzledLocal << 5n);
    }
    function effectiveMask(config) {
      const lowMask = ((1n << BigInt(config.b & 0xF)) - 1n) & ((1n << BigInt(MAX_MASK_BITS)) - 1n);
      return (lowMask << BigInt(((config.m & 0xF) - 5) & 0xF)) & 0x3FFn;
    }
    function effectiveAddressBitRules(config) {
      const mask = Number(effectiveMask(config));
      const rules = [{ label: '[4:0] = [4:0]', changed: false }];
      for (let outputBit = 5; outputBit <= 14; outputBit += 1) {
        const localBit = outputBit - 5;
        const sourceBit = outputBit + (config.s & 0xF);
        const changed = Boolean(mask & (1 << localBit)) && sourceBit <= 14;
        rules.push({
          outputBit, sourceBit, changed,
          label: changed ? outputBit + ' = ' + outputBit + ' ^ ' + sourceBit : outputBit + ' = ' + outputBit
        });
      }
      rules.push({ label: '[W-1:15] = [W-1:15]', changed: false });
      return rules;
    }
    function humanRuleSummary(config) {
      const changed = effectiveAddressBitRules(config).filter(rule => rule.changed);
      return changed.length
        ? changed.map(rule => 'addr[' + rule.outputBit + '] ^= addr[' + rule.sourceBit + ']').join(', ')
        : '恒等映射';
    }
    function transactionAt(config, row, byteOffset, wordInLane) {
      const address = config.base + BigInt(row * config.strideBytes + byteOffset + wordInLane * BANK_WORD_BYTES);
      const mapped = swizzleAddress(address, config);
      const linearBank = Number((address >> 5n) % BigInt(config.numBanks));
      const swizzledBank = Number((mapped >> 5n) % BigInt(config.numBanks));
      const elementStart = Math.floor((byteOffset + wordInLane * BANK_WORD_BYTES) * 8 / config.elementBits);
      const elementsInWord = Math.floor(BANK_WORD_BYTES * 8 / config.elementBits);
      return { row, wordInLane, address, mapped, linearBank, swizzledBank,
               elementStart, elementEnd: elementStart + elementsInWord - 1 };
    }
    function cycleTransactions(config, rowCycle, colCycle) {
      const firstRow = rowCycle * ROWS_PER_CYCLE;
      const firstByte = colCycle * config.accessBytes;
      const validBytes = Math.max(0, Math.min(config.accessBytes, config.rowBytes - firstByte));
      const words = Math.ceil(validBytes / BANK_WORD_BYTES);
      const result = [];
      for (let lane = 0; lane < ROWS_PER_CYCLE; lane += 1) {
        const row = firstRow + lane;
        if (row >= config.rows) break;
        for (let word = 0; word < words; word += 1) result.push(transactionAt(config, row, firstByte, word));
      }
      return result;
    }
    function conflictStats(transactions, mode, numBanks) {
      const load = Array(numBanks).fill(0);
      for (const transaction of transactions) {
        const bank = mode === 'swizzle' ? transaction.swizzledBank : transaction.linearBank;
        load[bank] += 1;
      }
      const peak = load.length ? Math.max(...load) : 0;
      return {
        load, peak, collidingBanks: load.filter(value => value > 1).length,
        excess: load.reduce((sum, value) => sum + Math.max(0, value - 1), 0),
        activeBanks: load.filter(Boolean).length
      };
    }
    function fourBlockBalance(config) {
      const sectors = config.rowBytes / BANK_WORD_BYTES;
      const target = Math.min(32, config.numBanks);
      let minimum = target;
      let windows = 0;
      for (let rowCycle = 0; rowCycle < config.rowGroups; rowCycle += 1) {
        const firstRow = rowCycle * ROWS_PER_CYCLE;
        if (firstRow + ROWS_PER_CYCLE > config.rows) continue;
        for (let start = 0; start + 4 <= sectors; start += 1) {
          const banks = new Set();
          for (let row = firstRow; row < firstRow + ROWS_PER_CYCLE; row += 1) {
            for (let column = start; column < start + 4; column += 1) {
              banks.add(transactionAt(config, row, column * BANK_WORD_BYTES, 0).swizzledBank);
            }
          }
          minimum = Math.min(minimum, banks.size);
          windows += 1;
        }
      }
      return { minimum: windows ? minimum : 0, target, windows,
               pass: windows > 0 && minimum === target };
    }
    function bitCount(value) {
      let count = 0;
      for (let remaining = value; remaining; remaining >>>= 1) count += remaining & 1;
      return count;
    }
    function compareScore(left, right) {
      for (let index = 0; index < left.length; index += 1) {
        if (left[index] !== right[index]) return left[index] - right[index];
      }
      return 0;
    }
    function evaluateMbsCandidate(baseConfig, s, b, m) {
      const config = { ...baseConfig, s, b, m };
      let globalPeak = 0, totalExcess = 0, conflictCycles = 0;
      for (let rowCycle = 0; rowCycle < config.rowGroups; rowCycle += 1) {
        for (let colCycle = 0; colCycle < config.colGroups; colCycle += 1) {
          const stats = conflictStats(cycleTransactions(config, rowCycle, colCycle), 'swizzle', config.numBanks);
          globalPeak = Math.max(globalPeak, stats.peak);
          totalExcess += stats.excess;
          if (stats.peak > 1) conflictCycles += 1;
        }
      }
      const balance = fourBlockBalance(config);
      const mask = Number(effectiveMask(config));
      const sourceWidth = Math.max(0, 10 - s);
      const activeMask = mask & (sourceWidth ? (1 << sourceWidth) - 1 : 0);
      const highestLocalBit = 31 - Math.clz32(activeMask);
      const reorderWindowBytes = activeMask ? 2 ** (highestLocalBit + 6) : BANK_WORD_BYTES;
      const xorBits = bitCount(activeMask);
      const score = [
        globalPeak, -balance.minimum, totalExcess, conflictCycles,
        reorderWindowBytes, xorBits, bitCount(mask), s, m, b
      ];
      return {
        s, b, m, mask, activeMask, globalPeak, totalExcess, conflictCycles, balance,
        reorderWindowBytes, xorBits, score
      };
    }
    function findOptimalMbs(config) {
      const bankBits = Math.log2(config.numBanks);
      const targetBits = Math.min(10, bankBits);
      // B=0 is the canonical no-op. Keep it when the linear layout is already optimal.
      let best = evaluateMbsCandidate(config, 1, 0, 5), candidates = 1;
      for (let m = 5; m < 5 + targetBits; m += 1) {
        const maxB = Math.min(MAX_MASK_BITS, targetBits - (m - 5));
        for (let b = 1; b <= maxB; b += 1) {
          for (let s = 1; s <= 9; s += 1) {
            const candidate = evaluateMbsCandidate(config, s, b, m);
            candidates += 1;
            if (!best || compareScore(candidate.score, best.score) < 0) best = candidate;
          }
        }
      }
      best.candidates = candidates;
      return best;
    }
    function configFingerprint(config) {
      return [config.elementBits, config.rows, config.rowBytes, config.strideBytes,
              config.base.toString(), config.numBanks].join('|');
    }
    function formatByteCount(bytes) {
      if (bytes >= 1024 && bytes % 1024 === 0) return (bytes / 1024) + ' KiB';
      return bytes + ' B';
    }
    function hideSearchResult() {
      state.searchResult = null;
      $('searchResult').classList.remove('visible');
      $('searchResultConfig').textContent = '-';
      $('searchResultStatus').textContent = '-';
      $('searchResultMeta').textContent = '-';
    }
    function renderSearchResult(result) {
      state.searchResult = result;
      $('searchResultConfig').textContent = 'M' + result.m + ' / B' + result.b + ' / S' + result.s;
      $('searchResultStatus').textContent = result.balance.pass && result.globalPeak === 1 ? '已均衡' : '当前最优';
      $('searchResultMeta').textContent =
        '峰值 ' + result.globalPeak + 'x | 四块覆盖 ' + result.balance.minimum + '/' + result.balance.target + '\n' +
        '额外请求 ' + result.totalExcess + ' | ' +
        (result.xorBits ? formatByteCount(result.reorderWindowBytes) + ' 重排窗口' : '不重排') + ' | ' +
        result.xorBits + ' 个 XOR 位\n' + humanRuleSummary(result) + '\n' +
        '已搜索 ' + result.candidates + ' 个候选 | ' + result.elapsedMs.toFixed(1) + ' ms';
      const alreadyActive = state.config.s === result.s && state.config.b === result.b && state.config.m === result.m;
      $('useSearchResult').disabled = alreadyActive;
      $('useSearchResult').textContent = alreadyActive ? '当前已采用' : '采用此方案';
      $('searchResult').classList.add('visible');
    }
    function searchOptimalMbs() {
      try {
        const config = parseConfig();
        state.config = config;
        renderAll();
        const button = $('searchButton');
        button.disabled = true;
        button.textContent = '搜索中...';
        window.setTimeout(() => {
          const startedAt = performance.now();
          const result = findOptimalMbs(config);
          const elapsedMs = performance.now() - startedAt;
          button.disabled = false;
          button.textContent = '搜索最优 M/B/S';
          if (!result) {
            setError('当前 SRAM 配置下没有可逆且能改变 Bank 映射的 M/B/S 候选。');
            return;
          }
          result.fingerprint = configFingerprint(config);
          result.elapsedMs = elapsedMs;
          renderSearchResult(result);
        }, 20);
      } catch (error) {
        $('searchButton').disabled = false;
        $('searchButton').textContent = '搜索最优 M/B/S';
        setError(error.message || String(error));
      }
    }
    function parseConfig() {
      const config = {
        mode: state.mode, elementBits: state.config ? state.config.elementBits : DEFAULTS.elementBits,
        rows: intValue('rows'), rowBytes: intValue('rowBytes'), strideBytes: intValue('strideBytes'),
        numBanks: intValue('numBanks'), s: intValue('shiftS'), b: intValue('widthB'),
        m: intValue('maskM'), accessBytes: 0, base: 0n,
        rowCycle: intValue('rowCycle'), colCycle: intValue('colCycle')
      };
      try { config.base = BigInt($('baseAddress').value.trim()); }
      catch (_) { throw new Error('起始地址必须是整数，例如 0x00000000。'); }
      config.accessBytes = accessBytes(config.elementBits);
      const fieldNames = {
        rows: '行数', rowBytes: '每行数据量', strideBytes: '行跨度',
        numBanks: 'Bank 数', s: 'S', b: 'B', m: 'M'
      };
      for (const key of ['rows', 'rowBytes', 'strideBytes', 'numBanks', 's', 'b', 'm']) {
        if (!Number.isInteger(config[key])) throw new Error(fieldNames[key] + '必须是整数。');
      }
      if (config.rows < 1 || config.rows > 256) throw new Error('行数必须在 [1, 256] 范围内。');
      if (config.rowBytes < 32 || config.rowBytes > 8192 || config.rowBytes % 32 !== 0) {
        throw new Error('每行数据量必须是 32B 的整数倍，且在 [32, 8192] 范围内。');
      }
      if (config.elementBits === 4 && config.rowBytes % 64 !== 0) {
        throw new Error('0.5B 元素每拍访问 8 x 64B，因此每行数据量必须是 64B 的整数倍。');
      }
      if (config.strideBytes < config.rowBytes || config.strideBytes % 32 !== 0) {
        throw new Error('行跨度必须是 32B 的整数倍，且不能小于每行数据量。');
      }
      if (config.base < 0n || config.base % 32n !== 0n) throw new Error('起始地址必须非负并按 32B 对齐。');
      if (!isPowerOfTwo(config.numBanks) || config.numBanks > 64) throw new Error('Bank 数必须是 2 的幂，且在 [1, 64] 范围内。');
      if (config.s < 0 || config.s > 15 || config.b < 0 || config.b > 15 || config.m < 0 || config.m > 15) {
        throw new Error('S、B、M 是 4-bit 无符号数，取值范围为 [0, 15]。');
      }
      config.rowGroups = Math.ceil(config.rows / ROWS_PER_CYCLE);
      config.colGroups = Math.ceil(config.rowBytes / config.accessBytes);
      config.rowCycle = clamp(Number.isInteger(config.rowCycle) ? config.rowCycle : 0, 0, config.rowGroups - 1);
      config.colCycle = clamp(Number.isInteger(config.colCycle) ? config.colCycle : 0, 0, config.colGroups - 1);
      return config;
    }
    function setError(message) {
      dom.errorBanner.textContent = message; dom.errorBanner.classList.add('visible');
      window.clearTimeout(setError.timer);
      setError.timer = window.setTimeout(() => dom.errorBanner.classList.remove('visible'), 5000);
    }
    function syncDerivedControls(config) {
      $('elementsPerRow').textContent = String(config.rowBytes * 8 / config.elementBits);
      $('cycleShape').textContent = '8 x ' + config.accessBytes + ' B';
      $('rowCycle').max = String(config.rowGroups - 1); $('colCycle').max = String(config.colGroups - 1);
      $('rowCycle').value = String(config.rowCycle); $('colCycle').value = String(config.colCycle);
      for (const button of document.querySelectorAll('#modeSwitch button')) {
        button.classList.toggle('active', button.dataset.mode === state.mode);
        button.setAttribute('aria-pressed', button.dataset.mode === state.mode ? 'true' : 'false');
      }
      for (const button of document.querySelectorAll('#widthSwitch button')) {
        const active = Number(button.dataset.bits) === config.elementBits;
        button.classList.toggle('active', active); button.setAttribute('aria-pressed', active ? 'true' : 'false');
      }
    }
    function prepareCanvas(canvas, cssWidth, cssHeight) {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.style.width = cssWidth + 'px'; canvas.style.height = cssHeight + 'px';
      canvas.width = Math.max(1, Math.round(cssWidth * ratio)); canvas.height = Math.max(1, Math.round(cssHeight * ratio));
      const context = canvas.getContext('2d');
      context.setTransform(ratio, 0, 0, ratio, 0, 0); context.imageSmoothingEnabled = true;
      return context;
    }
    function drawMap(config) {
      const wrapper = dom.mapScroll, sectors = config.rowBytes / BANK_WORD_BYTES;
      const left = 45, top = 25;
      const cellWidth = clamp(Math.max(360, wrapper.clientWidth - left - 2) / sectors, 17, 160);
      const cellHeight = clamp(Math.max(250, wrapper.clientHeight - top - 2) / config.rows, 9, 28);
      const width = Math.ceil(left + sectors * cellWidth + 2), height = Math.ceil(top + config.rows * cellHeight + 2);
      const context = prepareCanvas(dom.mapCanvas, width, height);
      context.fillStyle = '#ffffff'; context.fillRect(0, 0, width, height);
      context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
      context.textAlign = 'center'; context.textBaseline = 'middle'; state.mapCells = [];
      const tickEvery = Math.max(1, Math.ceil(48 / cellWidth));
      context.fillStyle = '#64717d';
      for (let column = 0; column < sectors; column += tickEvery) {
        context.fillText('+0x' + (column * 32).toString(16).toUpperCase().padStart(3, '0'), left + (column + 0.5) * cellWidth, 12);
      }
      const rowTickEvery = Math.max(1, Math.ceil(14 / cellHeight));
      for (let row = 0; row < config.rows; row += 1) {
        if (row % rowTickEvery === 0) {
          context.fillStyle = '#64717d'; context.textAlign = 'right';
          context.fillText('R' + row, left - 6, top + (row + 0.5) * cellHeight); context.textAlign = 'center';
        }
        for (let column = 0; column < sectors; column += 1) {
          const transaction = transactionAt(config, row, column * BANK_WORD_BYTES, 0);
          const bank = state.mode === 'swizzle' ? transaction.swizzledBank : transaction.linearBank;
          const color = bankColor(bank), x = left + column * cellWidth, y = top + row * cellHeight;
          context.fillStyle = color; context.fillRect(x + 0.5, y + 0.5, cellWidth - 1, cellHeight - 1);
          if (cellWidth >= 25 && cellHeight >= 10) {
            context.fillStyle = foreground(color);
            context.font = (cellHeight >= 15 && cellWidth >= 38 ? '10px' : '8px') + ' ui-monospace, SFMono-Regular, Menlo, monospace';
            context.fillText('B' + String(bank).padStart(2, '0'), x + cellWidth / 2, y + cellHeight / 2);
          }
          state.mapCells.push({ x, y, width: cellWidth, height: cellHeight, row, column, transaction });
        }
      }
      const cycleColumns = config.accessBytes / BANK_WORD_BYTES;
      const selectedX = left + config.colCycle * cycleColumns * cellWidth;
      const selectedY = top + config.rowCycle * ROWS_PER_CYCLE * cellHeight;
      const selectedWidth = Math.min(cycleColumns, sectors - config.colCycle * cycleColumns) * cellWidth;
      const selectedHeight = Math.min(ROWS_PER_CYCLE, config.rows - config.rowCycle * ROWS_PER_CYCLE) * cellHeight;
      context.strokeStyle = '#ffffff'; context.lineWidth = 4;
      context.strokeRect(selectedX + 1, selectedY + 1, selectedWidth - 2, selectedHeight - 2);
      context.strokeStyle = '#17232c'; context.lineWidth = 2;
      context.strokeRect(selectedX + 2, selectedY + 2, selectedWidth - 4, selectedHeight - 4);
      dom.mapCanvas._geometry = { left, top, cellWidth, cellHeight, sectors };
      $('mapSubtitle').textContent = config.rows + ' 行 x ' + sectors + ' 个 SRAM 字；每格代表一条 32B 请求';
      $('mapStat').textContent = (state.mode === 'swizzle' ? 'NPU Swizzle' : '线性映射') + ' / ' + config.numBanks + ' 个 Bank';
    }
    function buildConflictCells(config) {
      const cells = []; let globalPeak = 0;
      for (let rowCycle = 0; rowCycle < config.rowGroups; rowCycle += 1) {
        for (let colCycle = 0; colCycle < config.colGroups; colCycle += 1) {
          const stats = conflictStats(cycleTransactions(config, rowCycle, colCycle), state.mode, config.numBanks);
          globalPeak = Math.max(globalPeak, stats.peak); cells.push({ rowCycle, colCycle, stats });
        }
      }
      return { cells, globalPeak };
    }
    function drawConflict(config) {
      const wrapper = dom.conflictScroll, left = 33, top = 24;
      const cellWidth = clamp(Math.max(190, wrapper.clientWidth - left - 2) / config.colGroups, 24, 52);
      const cellHeight = clamp(Math.max(150, wrapper.clientHeight - top - 2) / config.rowGroups, 22, 43);
      const width = Math.ceil(left + config.colGroups * cellWidth + 2), height = Math.ceil(top + config.rowGroups * cellHeight + 2);
      const context = prepareCanvas(dom.conflictCanvas, width, height);
      context.fillStyle = '#ffffff'; context.fillRect(0, 0, width, height);
      context.textAlign = 'center'; context.textBaseline = 'middle';
      context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
      const built = buildConflictCells(config); state.conflictCells = [];
      for (let column = 0; column < config.colGroups; column += 1) {
        context.fillStyle = '#65717d'; context.fillText('C' + column, left + (column + 0.5) * cellWidth, 12);
      }
      for (let row = 0; row < config.rowGroups; row += 1) {
        context.fillStyle = '#65717d'; context.textAlign = 'right';
        context.fillText('R' + row, left - 6, top + (row + 0.5) * cellHeight); context.textAlign = 'center';
      }
      for (const cell of built.cells) {
        const x = left + cell.colCycle * cellWidth, y = top + cell.rowCycle * cellHeight;
        const color = conflictColor(cell.stats.peak);
        context.fillStyle = color; context.fillRect(x + 1, y + 1, cellWidth - 2, cellHeight - 2);
        context.fillStyle = cell.stats.peak >= 5 ? '#ffffff' : '#1b2730';
        context.font = '700 11px ui-monospace, SFMono-Regular, Menlo, monospace';
        context.fillText(cell.stats.peak + 'x', x + cellWidth / 2, y + cellHeight / 2);
        if (cell.rowCycle === config.rowCycle && cell.colCycle === config.colCycle) {
          context.strokeStyle = '#111b22'; context.lineWidth = 2;
          context.strokeRect(x + 1.5, y + 1.5, cellWidth - 3, cellHeight - 3);
        }
        state.conflictCells.push({ ...cell, x, y, width: cellWidth, height: cellHeight });
      }
      dom.conflictCanvas._geometry = { left, top, cellWidth, cellHeight };
      $('conflictSubtitle').textContent = (state.mode === 'swizzle' ? 'NPU Swizzle' : '线性映射') + '：每个 Bank 的请求峰值';
      $('globalPeak').textContent = '全局峰值 ' + built.globalPeak + 'x';
    }
    function drawCycle(config, transactions, selectedStats) {
      const firstRow = config.rowCycle * ROWS_PER_CYCLE, firstByte = config.colCycle * config.accessBytes;
      const words = Math.ceil(Math.min(config.accessBytes, config.rowBytes - firstByte) / BANK_WORD_BYTES);
      dom.cycleGrid.replaceChildren();
      dom.cycleGrid.style.gridTemplateRows = 'repeat(' + Math.min(ROWS_PER_CYCLE, config.rows - firstRow) + ', minmax(52px, 1fr))';
      for (let row = firstRow; row < Math.min(firstRow + ROWS_PER_CYCLE, config.rows); row += 1) {
        const lane = document.createElement('div'); lane.className = 'lane-row';
        lane.style.gridTemplateColumns = '42px repeat(' + words + ', minmax(105px, 1fr))';
        const label = document.createElement('div'); label.className = 'lane-label';
        label.textContent = 'R' + String(row).padStart(2, '0'); lane.appendChild(label);
        for (const transaction of transactions.filter(item => item.row === row)) {
          const bank = state.mode === 'swizzle' ? transaction.swizzledBank : transaction.linearBank;
          const color = bankColor(bank), tile = document.createElement('div');
          tile.className = 'transaction'; tile.style.background = color; tile.style.color = foreground(color);
          tile.title = '原始地址 ' + hex(transaction.address) + '\n映射地址 ' + hex(transaction.mapped);
          const comparison = transaction.linearBank === transaction.swizzledBank
            ? 'L' + String(transaction.linearBank).padStart(2, '0')
            : 'L' + String(transaction.linearBank).padStart(2, '0') + ' -> S' + String(transaction.swizzledBank).padStart(2, '0');
          tile.innerHTML = '<div class="transaction-main"><span class="transaction-bank">B' + String(bank).padStart(2, '0') +
            '</span><span class="transaction-word">W' + transaction.wordInLane + ' / ' + comparison + '</span></div>' +
            '<div class="transaction-meta">e' + transaction.elementStart + ':' + transaction.elementEnd + ' @ ' + hex(transaction.address) + '</div>';
          lane.appendChild(tile);
        }
        dom.cycleGrid.appendChild(lane);
      }
      const lastRow = Math.min(config.rows, firstRow + ROWS_PER_CYCLE) - 1;
      const lastByte = Math.min(config.rowBytes, firstByte + config.accessBytes) - 1;
      $('cycleSubtitle').textContent = '行 ' + firstRow + ':' + lastRow + ' / 字节偏移 ' + firstByte + ':' + lastByte;
      $('cycleStat').textContent = selectedStats.collidingBanks + ' 个冲突 Bank';
    }
    function drawPressure(config, stats) {
      const wrapper = dom.pressureWrap, left = 30, right = 8, top = 14, bottom = 28;
      const width = Math.max(250, wrapper.clientWidth), height = Math.max(165, wrapper.clientHeight);
      const context = prepareCanvas(dom.pressureCanvas, width, height);
      context.fillStyle = '#ffffff'; context.fillRect(0, 0, width, height);
      const chartWidth = width - left - right, chartHeight = height - top - bottom;
      const barWidth = chartWidth / config.numBanks, scaleMax = Math.max(2, stats.peak);
      context.strokeStyle = '#d8dee4'; context.lineWidth = 1;
      for (let level = 0; level <= scaleMax; level += 1) {
        const y = top + chartHeight - level / scaleMax * chartHeight;
        context.beginPath(); context.moveTo(left, y + 0.5); context.lineTo(width - right, y + 0.5); context.stroke();
      }
      context.textAlign = 'center'; context.textBaseline = 'top';
      context.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace';
      for (let bank = 0; bank < config.numBanks; bank += 1) {
        const load = stats.load[bank], barHeight = load / scaleMax * chartHeight;
        const x = left + bank * barWidth, y = top + chartHeight - barHeight;
        context.fillStyle = bankColor(bank); context.fillRect(x + 1, y, Math.max(1, barWidth - 2), barHeight);
        if (load > 1) {
          context.strokeStyle = '#8d221e'; context.lineWidth = 1.5;
          context.strokeRect(x + 1.5, y + 0.5, Math.max(0, barWidth - 3), Math.max(0, barHeight - 1));
        }
        const tickEvery = config.numBanks <= 16 ? 1 : config.numBanks <= 32 ? 4 : 8;
        if (bank % tickEvery === 0) {
          context.fillStyle = '#65717d'; context.fillText(String(bank), x + barWidth / 2, height - bottom + 7);
        }
      }
      context.fillStyle = '#65717d'; context.textAlign = 'right'; context.textBaseline = 'middle';
      context.fillText(String(scaleMax), left - 5, top); context.fillText('0', left - 5, top + chartHeight);
      $('pressureSubtitle').textContent = (state.mode === 'swizzle' ? 'NPU Swizzle' : '线性映射') + '：当前拍 Bank 请求';
      $('pressureStat').textContent = stats.activeBanks + '/' + config.numBanks + ' 个 Bank 被访问';
    }
    function updateMetrics(config, transactions, linearStats, swizzleStats) {
      $('linearPeak').textContent = linearStats.peak + 'x'; $('swizzlePeak').textContent = swizzleStats.peak + 'x';
      const delta = linearStats.excess - swizzleStats.excess;
      $('conflictDelta').textContent = (delta > 0 ? '-' : delta < 0 ? '+' : '') + Math.abs(delta);
      $('conflictDelta').className = 'metric-value ' + (delta > 0 ? 'good' : delta < 0 ? 'bad' : '');
      const balance = fourBlockBalance(config), coverage = $('fourBlockCoverage');
      coverage.textContent = balance.windows
        ? balance.minimum + '/' + balance.target + ' ' + (balance.pass ? '通过' : '未通过')
        : '不适用';
      coverage.className = 'metric-value ' + (balance.pass ? 'good' : 'bad');
      coverage.title = '已检查 ' + balance.windows + ' 个完整的连续四块窗口';
      $('transactionCount').textContent = String(transactions.length);
      const mask = effectiveMask(config).toString(16).toUpperCase().padStart(3, '0');
      const expression = $('expression');
      expression.replaceChildren();
      for (const rule of effectiveAddressBitRules(config)) {
        const item = document.createElement('span');
        item.className = 'bit-rule' + (rule.changed ? ' changed' : '');
        item.textContent = rule.label;
        expression.appendChild(item);
      }
      expression.title = 'outByte = (addr & ~0x7FE0) | ((local ^ ((local >> ' + config.s +
        ') & 0x' + mask + ')) << 5), where local = (addr >> 5) & 0x3FF';
      $('mapSubtitle').title = '元素位宽：' + (config.elementBits / 8) + 'B；起始地址：' + hex(config.base);
    }
    function renderAll() {
      const config = state.config; if (!config) return;
      const transactions = cycleTransactions(config, config.rowCycle, config.colCycle);
      const linearStats = conflictStats(transactions, 'linear', config.numBanks);
      const swizzleStats = conflictStats(transactions, 'swizzle', config.numBanks);
      const selectedStats = state.mode === 'swizzle' ? swizzleStats : linearStats;
      syncDerivedControls(config); updateMetrics(config, transactions, linearStats, swizzleStats);
      drawMap(config); drawConflict(config); drawCycle(config, transactions, selectedStats); drawPressure(config, selectedStats);
    }
    function applyControls() {
      try {
        const config = parseConfig();
        if (state.searchResult && state.searchResult.fingerprint !== configFingerprint(config)) hideSearchResult();
        state.config = config;
        dom.errorBanner.classList.remove('visible');
        renderAll();
      }
      catch (error) { setError(error.message || String(error)); }
    }
    function selectCycle(rowCycle, colCycle) {
      if (!state.config) return;
      state.config.rowCycle = clamp(rowCycle, 0, state.config.rowGroups - 1);
      state.config.colCycle = clamp(colCycle, 0, state.config.colGroups - 1);
      $('rowCycle').value = String(state.config.rowCycle); $('colCycle').value = String(state.config.colCycle); renderAll();
    }
    function resetControls() {
      state.mode = DEFAULTS.mode; state.config = { elementBits: DEFAULTS.elementBits };
      $('rows').value = DEFAULTS.rows; $('rowBytes').value = DEFAULTS.rowBytes;
      $('strideBytes').value = DEFAULTS.strideBytes; $('baseAddress').value = DEFAULTS.baseAddress;
      $('numBanks').value = DEFAULTS.numBanks; $('shiftS').value = DEFAULTS.s; $('widthB').value = DEFAULTS.b;
      $('maskM').value = DEFAULTS.m;
      $('rowCycle').value = DEFAULTS.rowCycle; $('colCycle').value = DEFAULTS.colCycle;
      $('preset').value = '3,1,7'; hideSearchResult(); applyControls();
    }
    function moveTooltip(event, content) {
      if (!content) { dom.tooltip.style.display = 'none'; return; }
      dom.tooltip.textContent = content; dom.tooltip.style.display = 'block';
      const width = dom.tooltip.offsetWidth, height = dom.tooltip.offsetHeight;
      dom.tooltip.style.left = clamp(event.clientX + 13, 6, window.innerWidth - width - 6) + 'px';
      dom.tooltip.style.top = clamp(event.clientY + 13, 6, window.innerHeight - height - 6) + 'px';
    }
    function mapHit(event) {
      const geometry = dom.mapCanvas._geometry; if (!geometry || !state.config) return null;
      const rect = dom.mapCanvas.getBoundingClientRect();
      const column = Math.floor((event.clientX - rect.left - geometry.left) / geometry.cellWidth);
      const row = Math.floor((event.clientY - rect.top - geometry.top) / geometry.cellHeight);
      if (row < 0 || row >= state.config.rows || column < 0 || column >= geometry.sectors) return null;
      return state.mapCells[row * geometry.sectors + column];
    }
    function conflictHit(event) {
      const geometry = dom.conflictCanvas._geometry; if (!geometry || !state.config) return null;
      const rect = dom.conflictCanvas.getBoundingClientRect();
      const column = Math.floor((event.clientX - rect.left - geometry.left) / geometry.cellWidth);
      const row = Math.floor((event.clientY - rect.top - geometry.top) / geometry.cellHeight);
      if (row < 0 || row >= state.config.rowGroups || column < 0 || column >= state.config.colGroups) return null;
      return state.conflictCells.find(cell => cell.rowCycle === row && cell.colCycle === column) || null;
    }
    function bindEvents() {
      $('controlsForm').addEventListener('submit', event => { event.preventDefault(); applyControls(); });
      $('applyButton').addEventListener('click', applyControls);
      $('resetButton').addEventListener('click', resetControls); $('sideResetButton').addEventListener('click', resetControls);
      $('searchButton').addEventListener('click', searchOptimalMbs);
      $('useSearchResult').addEventListener('click', () => {
        const result = state.searchResult;
        if (!result) return;
        $('shiftS').value = result.s; $('widthB').value = result.b; $('maskM').value = result.m;
        const presetValue = result.s + ',' + result.b + ',' + result.m;
        $('preset').value = [...$('preset').options].some(option => option.value === presetValue)
          ? presetValue : 'custom';
        applyControls();
        renderSearchResult(result);
      });
      for (const button of document.querySelectorAll('#modeSwitch button')) {
        button.addEventListener('click', () => {
          state.mode = button.dataset.mode; if (state.config) state.config.mode = state.mode; renderAll();
        });
      }
      for (const button of document.querySelectorAll('#widthSwitch button')) {
        button.addEventListener('click', () => {
          if (!state.config) state.config = {}; state.config.elementBits = Number(button.dataset.bits); applyControls();
        });
      }
      $('preset').addEventListener('change', event => {
        if (event.target.value !== 'custom') {
          const [s, b, m] = event.target.value.split(',');
          $('shiftS').value = s; $('widthB').value = b; $('maskM').value = m; applyControls();
        }
      });
      for (const id of ['shiftS', 'widthB', 'maskM']) $(id).addEventListener('input', () => { $('preset').value = 'custom'; });
      for (const button of document.querySelectorAll('[data-step]')) {
        button.addEventListener('click', () => {
          if (!state.config) return;
          const [axis, rawDelta] = button.dataset.step.split(','), delta = Number(rawDelta);
          if (axis === 'row') selectCycle(state.config.rowCycle + delta, state.config.colCycle);
          else selectCycle(state.config.rowCycle, state.config.colCycle + delta);
        });
      }
      $('rowCycle').addEventListener('change', () => selectCycle(intValue('rowCycle'), state.config.colCycle));
      $('colCycle').addEventListener('change', () => selectCycle(state.config.rowCycle, intValue('colCycle')));
      for (const metric of document.querySelectorAll('.metric[data-help]')) {
        metric.addEventListener('mousemove', event => moveTooltip(event, metric.dataset.help));
        metric.addEventListener('mouseleave', event => moveTooltip(event, null));
        metric.addEventListener('focus', () => {
          const rect = metric.getBoundingClientRect();
          moveTooltip({ clientX: rect.left + rect.width / 2, clientY: rect.bottom }, metric.dataset.help);
        });
        metric.addEventListener('blur', event => moveTooltip(event, null));
      }
      dom.mapCanvas.addEventListener('mousemove', event => {
        const cell = mapHit(event); if (!cell) return moveTooltip(event, null);
        const transaction = cell.transaction;
        moveTooltip(event, '第 ' + cell.row + ' 行，字节 +' + (cell.column * 32) + ':' + (cell.column * 32 + 31) + '\n' +
          '元素 e' + transaction.elementStart + ':e' + transaction.elementEnd + '\n' +
          hex(transaction.address) + ' -> ' + hex(transaction.mapped) + '\n' +
          'Bank：线性 ' + String(transaction.linearBank).padStart(2, '0') + ' -> Swizzle ' + String(transaction.swizzledBank).padStart(2, '0'));
      });
      dom.mapCanvas.addEventListener('mouseleave', event => moveTooltip(event, null));
      dom.mapCanvas.addEventListener('click', event => {
        const cell = mapHit(event); if (!cell) return;
        selectCycle(Math.floor(cell.row / ROWS_PER_CYCLE), Math.floor(cell.column * BANK_WORD_BYTES / state.config.accessBytes));
      });
      dom.conflictCanvas.addEventListener('mousemove', event => {
        const cell = conflictHit(event); if (!cell) return moveTooltip(event, null);
        moveTooltip(event, '拍 R' + cell.rowCycle + ' / C' + cell.colCycle + '\n峰值 ' + cell.stats.peak + 'x\n' +
          cell.stats.collidingBanks + ' 个冲突 Bank\n' + cell.stats.excess + ' 条额外请求');
      });
      dom.conflictCanvas.addEventListener('mouseleave', event => moveTooltip(event, null));
      dom.conflictCanvas.addEventListener('click', event => {
        const cell = conflictHit(event); if (cell) selectCycle(cell.rowCycle, cell.colCycle);
      });
      window.addEventListener('resize', () => {
        window.clearTimeout(state.resizeTimer); state.resizeTimer = window.setTimeout(renderAll, 100);
      });
    }
    function selfCheck() {
      const config = { s: 3, b: 1, m: 7 };
      console.assert(swizzleAddress(0x400n, config) === 0x480n, 'swizzle reference mismatch');
      console.assert((swizzleAddress(0x11Fn, config) & 0x1Fn) === 0x1Fn, 'low address bits changed');
      console.assert((swizzleAddress(0x8100n, config) >> 15n) === 1n, 'region bits changed');
    }
    function initialize() {
      for (const id of ['mapScroll', 'mapCanvas', 'conflictScroll', 'conflictCanvas', 'cycleGrid',
                        'pressureWrap', 'pressureCanvas', 'errorBanner', 'tooltip']) dom[id] = $(id);
      bindEvents(); selfCheck(); resetControls();
      window.VisualSwizzle = {
        swizzleAddress, effectiveMask, cycleTransactions, conflictStats, fourBlockBalance,
        findOptimalMbs, getState: () => state
      };
    }
    document.addEventListener('DOMContentLoaded', initialize);
  </script>
</body>
</html>
'''


def swizzle_address(addr: int, s: int, b: int, m: int) -> int:
    """Reference implementation of the Chisel primitive using byte addresses."""
    if addr < 0:
        raise ValueError("addr must be non-negative")
    s4, b4, m4 = s & 0xF, b & 0xF, m & 0xF
    local_addr = (addr >> 5) & 0x3FF
    shifted_addr = local_addr >> s4
    low_mask = ((1 << b4) - 1) & 0xFF
    mask = (low_mask << ((m4 - 5) & 0xF)) & 0x3FF
    swizzled_local = local_addr ^ (shifted_addr & mask)
    return (addr & ~(0x3FF << 5)) | (swizzled_local << 5)


def conflict_peak(addresses: Tuple[int, ...], banks: int, swizzled: bool = True) -> int:
    loads: Dict[int, int] = {}
    for address in addresses:
        mapped = swizzle_address(address, 3, 1, 7) if swizzled else address
        bank = (mapped >> 5) % banks
        loads[bank] = loads.get(bank, 0) + 1
    return max(loads.values(), default=0)


def run_self_test() -> None:
    assert swizzle_address(0x400, 3, 1, 7) == 0x480
    assert swizzle_address(0x41F, 3, 1, 7) == 0x49F
    assert swizzle_address(0x8400, 3, 1, 7) == 0x8480
    for address in (0, 1, 31, 32, 0x7FFF, 0x8000, 0x12345):
        mapped = swizzle_address(address, 3, 1, 7)
        assert (mapped & 0x1F) == (address & 0x1F)
        assert (mapped >> 15) == (address >> 15)
    # B is truncated to four bits and lowMask is always truncated to eight bits.
    for address in (0x100, 0x7FE0, 0x12340):
        assert swizzle_address(address, 1, 8, 7) == swizzle_address(address, 1, 15, 7)
    # M-5 is a four-bit UInt subtraction. M < 5 wraps to shifts 11..15,
    # which become zero after the shifted mask is truncated to ten bits.
    for m in range(5):
        assert swizzle_address(0x12340, 1, 8, m) == 0x12340
    row_words = tuple(row * 256 for row in range(8))
    assert conflict_peak(row_words, 32, swizzled=False) == 2
    assert conflict_peak(row_words, 32, swizzled=True) == 1
    for first_row in range(0, 32, 8):
        for first_column in range(5):
            four_blocks = tuple(
                row * 256 + column * 32
                for row in range(first_row, first_row + 8)
                for column in range(first_column, first_column + 4)
            )
            mapped_banks = {(swizzle_address(address, 3, 1, 7) >> 5) % 32 for address in four_blocks}
            assert mapped_banks == set(range(32))
    int4_words = tuple(row * 256 + word * 32 for row in range(8) for word in range(2))
    assert conflict_peak(int4_words, 32, swizzled=True) == 1
    print("VisualSwizzle 自检：通过")


class VisualSwizzleHandler(BaseHTTPRequestHandler):
    server_version = "VisualSwizzle/2"

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/":
            payload = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        if path == "/health":
            payload = json.dumps({"status": "ok", "service": "VisualSwizzle"}).encode("ascii")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if path == "/favicon.ico":
            self.send_response(204); self.end_headers(); return
        self.send_error(404)

    def do_HEAD(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/":
            payload_length = len(HTML.encode("utf-8"))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(payload_length))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        if path == "/health":
            payload_length = len(json.dumps({"status": "ok", "service": "VisualSwizzle"}).encode("ascii"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(payload_length))
            self.end_headers()
            return
        self.send_error(404)

    def log_message(self, format_string: str, *args: object) -> None:
        print("[%s] %s" % (self.log_date_time_string(), format_string % args))


def make_server(host: str, preferred_port: int) -> ThreadingHTTPServer:
    last_error: OSError | None = None
    for port in range(preferred_port, preferred_port + 21):
        try:
            return ThreadingHTTPServer((host, port), VisualSwizzleHandler)
        except OSError as error:
            last_error = error
    assert last_error is not None
    raise last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="NPU SRAM Swizzle 可视化工具", add_help=False)
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP 监听地址")
    parser.add_argument("--port", type=int, default=8765, help="首选 HTTP 端口")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    parser.add_argument("--self-test", action="store_true", help="运行模型自检后退出")
    args = parser.parse_args()
    if args.self_test:
        run_self_test(); return
    server = make_server(args.host, args.port)
    host, port = server.server_address[:2]
    browser_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = "http://%s:%d" % (browser_host, port)
    print("VisualSwizzle 已启动：%s" % url)
    print("按 Ctrl+C 停止服务。")
    if not args.no_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止 VisualSwizzle。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
