(() => {
  const $ = (id) => document.getElementById(id);
  let token = localStorage.getItem('aster_token') || '';
  let cachedUsername = localStorage.getItem('aster_username') || '';
  let user = null;
  let authMode = 'login';

  // Device-bound guest token management for unauthenticated / anonymous users
  let guestToken = localStorage.getItem('astra_guest_token') || '';
  let deviceId = localStorage.getItem('astra_device_id') || '';
  if (!deviceId) {
    deviceId = 'dev_' + (window.crypto && window.crypto.randomUUID ? window.crypto.randomUUID() : (Math.random().toString(36).substring(2) + Date.now().toString(36)));
    localStorage.setItem('astra_device_id', deviceId);
  }

  // Auto-reset API base if on localhost so requests always go to local server
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    localStorage.removeItem('astra_api_base_url');
  }

  // Configurable base URL for remote cloud deployment & Android mobile client
  const getApiBase = () => {
    return (window.ASTRA_API_BASE_URL || localStorage.getItem('astra_api_base_url') || '').replace(/\/+$/, '');
  };

  const api = async (path, options = {}) => {
    if (typeof navigator !== 'undefined' && navigator.onLine === false) {
      toast('Unable to connect to Astra. Please check your internet connection.', true);
      throw new Error('Unable to connect to Astra. Please check your internet connection.');
    }
    const headers = { ...(options.headers || {}) };
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    } else if (guestToken) {
      headers.Authorization = `Bearer ${guestToken}`;
      headers['X-Guest-Token'] = guestToken;
    }
    headers['X-Device-Id'] = deviceId;

    const fullUrl = path.startsWith('http') ? path : `${getApiBase()}${path}`;
    let response;
    try {
      response = await fetch(fullUrl, { credentials: 'include', ...options, headers });
    } catch (err) {
      const msg = (typeof navigator !== 'undefined' && navigator.onLine === false)
        ? 'Unable to connect to Astra. Please check your internet connection.'
        : 'Astra is temporarily unavailable. Please try again.';
      toast(msg, true);
      throw new Error(msg);
    }

    const serverGuestToken = response.headers && response.headers.get('X-Guest-Token');
    if (serverGuestToken && !token) {
      guestToken = serverGuestToken;
      localStorage.setItem('astra_guest_token', guestToken);
    }
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (response.status === 502 || response.status === 503) {
        toast('Astra is temporarily unavailable. Please try again.', true);
        throw new Error('Astra is temporarily unavailable. Please try again.');
      }
      throw new Error(body.detail || body.error || `Request failed (${response.status})`);
    }
    return body;
  };

  const stripEmojis = (str) => {
    return String(str || '')
      .replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE00}-\u{FE0F}\u{1F000}-\u{1F02F}\u{1F0A0}-\u{1F0FF}]/gu, '')
      .replace(/\s{2,}/g, ' ')
      .trim();
  };

  const toast = (text, isError = false) => {
    const node = $('toast');
    node.textContent = stripEmojis(text);
    node.style.background = isError ? '#e11d48' : '#ec4899';
    node.classList.remove('hidden');
    setTimeout(() => node.classList.add('hidden'), 4200);
  };

  const escapeHtml = (text) =>
    String(text || '').replace(/[&<>"']/g, (x) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    }[x]));

  // Time-sensitive predefined welcome taglines (each <= 4 words)
  const TIME_TAGLINES = {
    // Night: 21:00 - 04:59
    night: [
      'QUIET HOURS DEEP THOUGHTS',
      'MIDNIGHT INQUIRY AWAITS YOU',
      'NIGHT WISDOM BEGINS HERE',
      'THINK CLEAR UNDER STARS',
      'LATE HOURS SHARP MINDS',
      'NIGHTTIME FOCUS UNLOCKED TODAY',
      'PEACEFUL THOUGHTS DEEP ANSWERS',
      'MOONLIT INSIGHTS READY NOW'
    ],
    // Morning: 05:00 - 11:59
    morning: [
      'GOOD MORNING DIVE IN',
      'START FRESH THINK DEEP',
      'MORNING CLARITY BEGINS TODAY',
      'BRIGHT IDEAS AWAIT YOU',
      'RISE AND DISCOVER TRUTH',
      'EARLY FOCUS SHARP RESULTS',
      'NEW DAY FRESH INSIGHTS',
      'MORNING INSPIRATION STARTS HERE'
    ],
    // Afternoon: 12:00 - 16:59
    afternoon: [
      'AFTERNOON MOMENTUM STARTS NOW',
      'THINK FAST MOVE FORWARD',
      'POWER THROUGH YOUR DAY',
      'SHARP FOCUS ACTIVE MINDS',
      'DISCOVER ANSWERS WITH PRECISION',
      'KEEP EXPLORING YOUR IDEAS',
      'SPEED UP YOUR SEARCH',
      'DAYLIGHT BRINGS CLEAR THOUGHTS'
    ],
    // Evening: 17:00 - 20:59
    evening: [
      'GOOD EVENING EXPLORE FREELY',
      'SUNSET THOUGHTS CLEAR MINDS',
      'UNWIND WITH DEEP INSIGHTS',
      'EVENING CLARITY AT HAND',
      'REFLECT LEARN GROW TODAY',
      'TWILIGHT KNOWLEDGE UNLOCKED NOW',
      'FINISH YOUR DAY STRONG',
      'DISCOVER PEACE IN ANSWERS'
    ]
  };

  function getTimePeriod() {
    const hour = new Date().getHours();
    if (hour >= 5 && hour < 12) return 'morning';
    if (hour >= 12 && hour < 17) return 'afternoon';
    if (hour >= 17 && hour < 21) return 'evening';
    return 'night';
  }

  function shuffleWelcomeTagline() {
    const el = $('welcomeShuffledTagline');
    if (!el) return;
    const period = getTimePeriod();
    const pool = TIME_TAGLINES[period] || TIME_TAGLINES.morning;
    const picked = pool[Math.floor(Math.random() * pool.length)];
    el.style.opacity = '0';
    setTimeout(() => {
      el.textContent = picked;
      el.style.opacity = '1';
    }, 180);
  }

  function updateStageMode() {
    const stage = $('centerStage') || document.querySelector('.astra-center-stage');
    const welcome = $('welcome');
    const messageInput = $('message');
    if (!stage || !welcome) return;
    const isWelcome = !welcome.classList.contains('hidden');
    stage.classList.toggle('welcome-mode', isWelcome);
    stage.classList.toggle('chat-active', !isWelcome);

    if (messageInput) {
      if (isWelcome) {
        messageInput.placeholder = 'Begin searching';
      } else {
        messageInput.placeholder = 'Type here';
      }
    }
  }
  const renderMathContent = (text) => {
    if (!text) return '';

    // Function to render a single math chunk
    const formatLatexChunk = (tex, isBlock) => {
      const trimmed = tex.trim();
      if (!trimmed) return '';
      if (window.katex && typeof window.katex.renderToString === 'function') {
        try {
          return window.katex.renderToString(trimmed, {
            displayMode: isBlock,
            throwOnError: false,
          });
        } catch (e) {
          // Fallback below
        }
      }
      // Readable clean text fallback when KaTeX isn't loaded:
      let clean = trimmed
        .replace(/\\int\s*/g, '∫ ')
        .replace(/\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}/g, '($1 / $2)')
        .replace(/\\neq\s*/g, ' ≠ ')
        .replace(/\\leq\s*/g, ' ≤ ')
        .replace(/\\geq\s*/g, ' ≥ ')
        .replace(/\\times\s*/g, ' × ')
        .replace(/\\cdot\s*/g, ' · ')
        .replace(/\\pm\s*/g, ' ± ')
        .replace(/\\infty\s*/g, '∞')
        .replace(/\\sqrt\s*\{([^}]+)\}/g, '√($1)')
        .replace(/\\\s*/g, ' ')
        .replace(/\^\{([^}]+)\}/g, '^$1')
        .replace(/_\{([^}]+)\}/g, '_$1');
      return `<span style="font-family: 'DM Mono', monospace; font-weight: 600; color: #38bdf8; background: rgba(56, 189, 248, 0.12); padding: 2px 6px; border-radius: 6px;">${escapeHtml(clean)}</span>`;
    };

    // 1. Block math: $$ ... $$ or \[ ... \]
    let processed = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, math) => {
      return `<div style="margin: 12px 0; text-align: center; overflow-x: auto;">${formatLatexChunk(math, true)}</div>`;
    });
    processed = processed.replace(/\\\[([\s\S]+?)\\\]/g, (_, math) => {
      return `<div style="margin: 12px 0; text-align: center; overflow-x: auto;">${formatLatexChunk(math, true)}</div>`;
    });

    // 2. Inline math: $ ... $ or \( ... \) (carefully ignoring currency like $100 or 100$)
    processed = processed.replace(/\\\(([\s\S]+?)\\\)/g, (_, math) => {
      return formatLatexChunk(math, false);
    });

    // Match $math$ where inside contains math symbols (\, ^, _, =, etc.) or starts/ends without space
    processed = processed.replace(/\$([^\n$]+?)\$/g, (match, math) => {
      // If it looks like pure currency, e.g. "$ 50" or "$100", leave it
      if (/^\s*\d+(\.\d+)?\s*$/.test(math)) {
        return `$${escapeHtml(math)}`;
      }
      return formatLatexChunk(math, false);
    });

    return processed;
  };

  // Client-side Chemistry Problem & Solution Detector
  const isChemistryContent = (query = '', answer = '') => {
    const combined = `${query} ${answer}`.toLowerCase();
    const chemKeywords = [
      'chemistry', 'chemisrty', 'chemestry', 'chemical', 'chemist', 'chem',
      'reaction', 'reactant', 'reagent', 'molecule', 'molecular', 'atom', 'atomic',
      'element', 'elements', 'periodic table', 'compound', 'compounds', 'isotope',
      'ion', 'ions', 'anion', 'cation', 'acid', 'acids', 'acidic', 'base', 'bases',
      'alkali', 'alkaline', 'titration', 'titrate', 'buffer', 'neutralization',
      'molarity', 'molality', 'stoichiometry', 'stoichiometric', 'solute', 'solvent',
      'solubility', 'precipitate', 'avogadro', 'enthalpy', 'entropy', 'endothermic',
      'exothermic', 'catalyst', 'catalysis', 'electronegativity', 'hybridization',
      'organic chemistry', 'inorganic chemistry', 'hydrocarbon', 'alkane', 'alkene',
      'alkyne', 'polymer', 'redox', 'oxidation state', 'oxidation number',
      'half-reaction', 'molar mass', 'atomic mass', 'atomic number'
    ];
    for (const kw of chemKeywords) {
      if (new RegExp(`\\b${kw}\\b`, 'i').test(combined)) return true;
    }
    if (/\bpH\b/.test(`${query} ${answer}`)) return true;
    if (/\b\d+(?:\.\d+)?\s*(?:mol|moles?)\b/i.test(combined)) return true;
    const formulas = [
      'h2o', 'h2so4', 'hcl', 'hno3', 'nacl', 'naoh', 'koh', 'co2', 'ch4', 'nh3',
      'caco3', 'c6h12o6', 'o2', 'n2', 'h2', 'cl2', 'br2', 'fe2o3', 'cuso4', 'kmno4',
      'agno3', 'bacl2', 'mgcl2', 'al2o3', 'no2', 'so2', 'so3', 'ch3cooh', 'c2h5oh'
    ];
    for (const f of formulas) {
      if (new RegExp(`\\b${f}\\b`, 'i').test(combined)) return true;
    }
    if (/(?:->|→|⇌)\s*[A-Z][a-z]?/.test(`${query} ${answer}`)) return true;
    return false;
  };

  let nextWidgetId = 1;

  // Formats text: handles 3D models, interactive charts, mermaid diagrams, code blocks, images, math, tables
  const formatMessageText = (text) => {
    if (!text) return '';

    // Strip any internal thinking process dumps
    text = text.replace(/<think>[\s\S]*?<\/think>/g, '');
    text = text.replace(/(?:^|\n)(?:Here(?:'s| is) a thinking process:?|Thinking Process:?|1\. Analyze User Input:?)[\s\S]*?(?=(?:\n\n[A-Z]|\n\n\*\*|\n\n#|\n\n-|\n\n•|$))/gi, '').trim();
    if (!text) return '';

    // 1. Normalize code fences if LLM used '''lang or """lang
    let normalized = text.replace(/^([ \t]*)'{3}([a-zA-Z0-9_#+.-]*)/gm, '$1```$2');
    normalized = normalized.replace(/^([ \t]*)"{3}([a-zA-Z0-9_#+.-]*)/gm, '$1```$2');

    // Auto-close unclosed code block if odd number of ```
    const fenceCount = (normalized.match(/```/g) || []).length;
    if (fenceCount % 2 !== 0) {
      normalized += '\n```';
    }

    // 2. Extract and protect all fenced code blocks (```lang ... ```) FIRST so internal text is never altered
    const codeWidgets = [];
    let processed = normalized.replace(/```[ \t]*([a-zA-Z0-9_#+.-]*)[^\n]*\r?\n([\s\S]*?)```/g, (match, lang, code) => {
      const widgetId = `widget_${Date.now()}_${nextWidgetId++}`;
      const cleanLang = (lang || '').toLowerCase().trim();
      codeWidgets.push({ id: widgetId, lang: cleanLang, code: code.trim(), raw: match });
      return `%%CODEWIDGET_${widgetId}%%`;
    });

    // Strip citation tags: 【W1】, 【W2】, [W1], [S1], (W1), etc.
    processed = processed.replace(/【[^】]*】/g, '');
    processed = processed.replace(/\[[WwSs]\d+\]/g, '');
    processed = processed.replace(/\([WwSs]\d+\)/g, '');
    processed = processed.replace(/[ \t]{2,}/g, ' ');

    // Handle markdown images: ![alt](url)
    processed = processed.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (match, alt, src) => {
      const cleanSrc = src.trim();
      const cleanAlt = escapeHtml(alt.trim() || 'AI Generated Artwork');
      return `
        <div class="chat-image-card">
          <div class="chat-image-wrapper" onclick="window.open('${cleanSrc}', '_blank')">
            <img src="${cleanSrc}" alt="${cleanAlt}" class="chat-image-preview" loading="lazy" />
            <div class="chat-image-overlay">
              <span class="chat-image-overlay-text">🔍 Click to expand full resolution</span>
            </div>
          </div>
          <div class="chat-image-footer">
            <div class="chat-image-badge">
              <span>✨</span>
              <span>1024×1024 Flux.1</span>
            </div>
            <div class="chat-image-actions">
              <a href="${cleanSrc}" download="aster_artwork.jpg" target="_blank" class="chat-image-action-btn" onclick="event.stopPropagation()">
                ⬇️ Download
              </a>
              <a href="${cleanSrc}" target="_blank" class="chat-image-action-btn" onclick="event.stopPropagation()">
                🔍 Open Full
              </a>
            </div>
          </div>
        </div>
      `;
    });

    // Format markdown tables into sleek data tables
    processed = processed.replace(/(?:(?:^|\n)\|[^\n]+\|\r?\n\|[-: |]+\|\r?\n(?:\|[^\n]+\|\r?\n?)+)/g, (tableBlock) => {
      const rows = tableBlock.trim().split(/\r?\n/).filter((r) => r.trim().startsWith('|'));
      if (rows.length < 2) return tableBlock;
      const headerCells = rows[0].split('|').slice(1, -1).map((c) => c.trim());
      const dataRows = rows.slice(2);
      const thHtml = headerCells.map((h) => `<th>${h}</th>`).join('');
      const trHtml = dataRows.map((r) => {
        const cells = r.split('|').slice(1, -1).map((c) => `<td>${c.trim()}</td>`).join('');
        return `<tr>${cells}</tr>`;
      }).join('');
      return `<div class="deep-research-table-wrapper"><table class="deep-research-table"><thead><tr>${thHtml}</tr></thead><tbody>${trHtml}</tbody></table></div>`;
    });

    // Extract and render math blocks/inlines safely
    let mathHandled = renderMathContent(processed);

    // Format markdown headers cleanly without hashtags (#)
    // Only apply to lines that are NOT placeholder lines
    mathHandled = mathHandled.replace(/^(#{1,6})\s*(.+)$/gm, (m, hashes, title) => {
      if (title.includes('%%CODEWIDGET')) return m; // skip placeholder lines
      return `<h4 class="chat-subheading">${title}</h4>`;
    });

    // Format bullet points cleanly without asterisks (*)
    mathHandled = mathHandled.replace(/^[\s]*\*\s+(.+)$/gm, (m, content) => {
      if (content.includes('%%CODEWIDGET')) return m;
      return `• ${content}`;
    });

    // Format bold markdown (**text** or __text__) cleanly as bold and red important words
    mathHandled = mathHandled.replace(/\*\*\*([^*]+?)\*\*\*/g, '<strong class="agent-important-word"><em>$1</em></strong>');
    mathHandled = mathHandled.replace(/\*\*([^*]+?)\*\*/g, '<strong class="agent-important-word">$1</strong>');
    mathHandled = mathHandled.replace(/__([^_]+?)__/g, '<strong class="agent-important-word">$1</strong>');
    mathHandled = mathHandled.replace(/\*([^*\n]+?)\*/g, '<em>$1</em>');

    // Replace multiplication like 2 * 3 with 2 × 3
    mathHandled = mathHandled.replace(/(\d+)\s*\*\s*(\d+)/g, '$1 × $2');

    // Format inline code `code` cleanly
    mathHandled = mathHandled.replace(/`([^`\n]+?)`/g, '<code class="chat-inline-code">$1</code>');

    // Remove stray comment-like slash banners in prose outside code (e.g. / ---- ... ---- / or / --- /)
    mathHandled = mathHandled.replace(/^[ \t]*\/+[ \t]*[-=~_]+.*?[-=~_]+[ \t]*\/+[ \t]*$/gm, '');
    mathHandled = mathHandled.replace(/(?<![a-zA-Z0-9_])\/[ \t]*[-=~_]{3,}[ \t]*\/(?![a-zA-Z0-9_])/g, '');

    // Remove any remaining stray asterisks, hashtags, and citation brackets from the agent response outside code
    // Only remove them on lines that don't contain placeholders
    mathHandled = mathHandled.split('\n').map(line => {
      if (line.includes('%%CODEWIDGET')) return line;
      return line.replace(/[*#]/g, '');
    }).join('\n');
    mathHandled = mathHandled.replace(/【[^】]*】/g, '').replace(/\[[WwSs]\d+\]/g, '').replace(/\([WwSs]\d+\)/g, '');

    // Format emojis with tight, clean, evenly shaped borders following their natural shape
    mathHandled = mathHandled.replace(/(<[^>]+>)|(\p{Extended_Pictographic}(?:\u200D\p{Extended_Pictographic}|\uFE0F|\uFE0E|[\uD83C\uDFFB-\uD83C\uDFFF])*)/gu, (match, tag, emoji) => {
      if (tag) return tag;
      if (!emoji) return match;
      return `<span class="chat-emoji">${emoji}</span>`;
    });

    // 3. Restore and render all code widgets (3D Models, Charts, Diagrams, Code Blocks)
    for (const w of codeWidgets) {
      const placeholder = `%%CODEWIDGET_${w.id}%%`;
      let widgetHtml = '';

      if (w.lang === '3d' || w.lang === 'threejs' || w.lang === 'webgl') {
        let spec = null;
        try {
          spec = JSON.parse(w.code);
        } catch (_) {
          spec = {
            type: 'custom',
            title: 'Procedural 3D Model',
            description: 'Interactive real-time 3D simulation with 360° mouse controls',
            code: w.code,
          };
        }

        const title = escapeHtml(spec.title || (spec.type ? `${spec.type.toUpperCase()} 3D Model` : 'Interactive 3D Model'));
        const desc = spec.description ? `<div class="interactive-3d-desc">${escapeHtml(spec.description)}</div>` : '';
        const rawJson = escapeHtml(JSON.stringify(spec));

        widgetHtml = `
          <div class="interactive-3d-card" id="${w.id}" data-spec="${rawJson}">
            <div class="interactive-3d-header">
              <div class="interactive-3d-meta">
                <span class="interactive-3d-badge">✦ 3D WebGL</span>
                <span class="interactive-3d-title">${title}</span>
              </div>
              <div class="interactive-3d-controls">
                <button type="button" class="ctrl-btn-3d btn-rotate" title="Toggle Auto-Rotation">🔄 Rotate</button>
                <button type="button" class="ctrl-btn-3d btn-wireframe" title="Toggle Wireframe">🌐 Wireframe</button>
                <button type="button" class="ctrl-btn-3d btn-reset" title="Reset View">↺ Reset</button>
                <button type="button" class="ctrl-btn-3d btn-fullscreen" title="Toggle Fullscreen">⛶ Full</button>
              </div>
            </div>
            <div class="interactive-3d-viewport">
              <canvas class="interactive-3d-canvas"></canvas>
              <div class="interactive-3d-hint">🖱️ Left Drag: Orbit · Right Drag: Pan · Scroll: Zoom</div>
            </div>
            ${desc}
          </div>
        `;
      } else if (w.lang === 'chart' || w.lang === 'chartjs') {
        let chartSpec = null;
        try {
          chartSpec = JSON.parse(w.code);
        } catch (_) {
          chartSpec = { type: 'bar', data: { labels: [], datasets: [] }, title: 'Data Visualization' };
        }
        const title = escapeHtml(chartSpec.title || 'Interactive Data Chart');
        const rawJson = escapeHtml(JSON.stringify(chartSpec));

        widgetHtml = `
          <div class="interactive-chart-card" id="${w.id}" data-spec="${rawJson}">
            <div class="interactive-chart-header">
              <div class="interactive-chart-meta">
                <span class="interactive-chart-badge">📊 Chart</span>
                <span class="interactive-chart-title">${title}</span>
              </div>
              <button type="button" class="chart-download-btn" title="Download Chart as PNG">⬇️ PNG</button>
            </div>
            <div class="interactive-chart-body">
              <canvas class="interactive-chart-canvas"></canvas>
            </div>
          </div>
        `;
      } else if (w.lang === 'mermaid') {
        widgetHtml = `
          <div class="interactive-mermaid-card" id="${w.id}">
            <div class="interactive-mermaid-header">
              <span class="interactive-mermaid-badge">🔷 Architecture & Logic Diagram</span>
            </div>
            <div class="mermaid">${escapeHtml(w.code)}</div>
          </div>
        `;
      } else {
        const rawLang = (w.lang || '').toLowerCase().trim();
        let langDisplay = 'Code';
        if (rawLang === 'python' || rawLang === 'py') langDisplay = 'Python';
        else if (rawLang === 'javascript' || rawLang === 'js') langDisplay = 'JavaScript';
        else if (rawLang === 'typescript' || rawLang === 'ts') langDisplay = 'TypeScript';
        else if (rawLang === 'html') langDisplay = 'HTML';
        else if (rawLang === 'css') langDisplay = 'CSS';
        else if (rawLang === 'cpp' || rawLang === 'c++') langDisplay = 'C++';
        else if (rawLang === 'c') langDisplay = 'C';
        else if (rawLang === 'csharp' || rawLang === 'cs') langDisplay = 'C#';
        else if (rawLang === 'java') langDisplay = 'Java';
        else if (rawLang === 'bash' || rawLang === 'sh' || rawLang === 'shell') langDisplay = 'Bash';
        else if (rawLang === 'sql') langDisplay = 'SQL';
        else if (rawLang === 'json') langDisplay = 'JSON';
        else if (rawLang === 'rust' || rawLang === 'rs') langDisplay = 'Rust';
        else if (rawLang === 'go' || rawLang === 'golang') langDisplay = 'Go';
        else if (rawLang) langDisplay = rawLang.charAt(0).toUpperCase() + rawLang.slice(1);

        widgetHtml = `
          <div class="code-editor-card chatgpt-style" id="${w.id}" data-language="${rawLang}">
            <div class="code-editor-header">
              <div class="code-editor-meta">
                <span class="code-lang-icon">&lt;/&gt;</span>
                <span class="code-lang-title">${escapeHtml(langDisplay)}</span>
              </div>
              <div class="code-editor-actions">
                <button type="button" class="copy-code-btn" title="Copy code">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                  </svg>
                  <span class="copy-btn-label">Copy code</span>
                </button>
                <button type="button" class="download-code-btn" title="Download code file" data-lang="${escapeHtml(rawLang)}">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7 10 12 15 17 10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                  </svg>
                  <span class="download-btn-label">Download</span>
                </button>
                <button type="button" class="live-demo-btn" title="Live Demo">Live Demo</button>
              </div>
            </div>
            <div class="code-editor-viewport">
              <pre class="code-editor-pre"><code class="code-editor-code language-${escapeHtml(rawLang)}">${escapeHtml(w.code)}</code></pre>
            </div>
            <div class="code-run-drawer hidden">
              <div class="code-run-drawer-header">
                <span>Console Output</span>
                <button type="button" class="close-run-btn" title="Close Console">✕</button>
              </div>
              <pre class="code-run-terminal"></pre>
            </div>
          </div>
        `;
      }

      mathHandled = mathHandled.replace(placeholder, () => widgetHtml);
    }

    return mathHandled;
  };

  // Duration formatting matching mockup: "Responded in 0.8 seconds"
  const formatDuration = (ms) => {
    const val = Number(ms);
    if (!val || val <= 0) return '0.8 seconds';
    return `${(val / 1000).toFixed(1)} seconds`;
  };

  // Sidebar visibility & collapsed glass rail management
  function toggleSidebar(forceOpen) {
    const sidebar = $('sidebar');
    if (!sidebar) return;
    let isClosed;
    if (typeof forceOpen === 'boolean') {
      isClosed = !forceOpen;
      sidebar.classList.toggle('closed', isClosed);
    } else {
      isClosed = sidebar.classList.toggle('closed');
    }
    localStorage.setItem('aster_sidebar_closed', isClosed);
    const rail = $('sidebarRail');
    if (rail) {
      if (isClosed) {
        rail.classList.remove('hidden');
      } else {
        rail.classList.add('hidden');
      }
    }
    const backdrop = $('sidebarBackdrop');
    if (backdrop) {
      const isMobile = window.innerWidth <= 768;
      if (isMobile && !isClosed) {
        backdrop.classList.remove('hidden');
      } else {
        backdrop.classList.add('hidden');
      }
    }
  }

  function initSidebar() {
    let saved = localStorage.getItem('aster_sidebar_closed');
    const isMobile = window.innerWidth <= 768;
    if (window.location.search.includes('sidebar=closed')) saved = 'true';
    if (window.location.search.includes('sidebar=open')) saved = 'false';
    // On small mobile screens, default to closed if not explicitly saved
    const isClosed = saved !== null ? saved === 'true' : isMobile;
    const rail = $('sidebarRail');
    if (isClosed) {
      $('sidebar').classList.add('closed');
      if (rail) rail.classList.remove('hidden');
    } else {
      $('sidebar').classList.remove('closed');
      if (rail) rail.classList.add('hidden');
    }

    const backdrop = $('sidebarBackdrop');
    if (backdrop) {
      backdrop.onclick = () => toggleSidebar(false);
    }

    window.addEventListener('resize', () => {
      const isNowMobile = window.innerWidth <= 768;
      const sidebar = $('sidebar');
      const backdrop = $('sidebarBackdrop');
      if (!isNowMobile && backdrop) {
        backdrop.classList.add('hidden');
      } else if (isNowMobile && sidebar && !sidebar.classList.contains('closed') && backdrop) {
        backdrop.classList.remove('hidden');
      }
    });

    if ($('sidebarToggleBtn')) $('sidebarToggleBtn').onclick = () => toggleSidebar();
    if ($('closeSidebar')) $('closeSidebar').onclick = () => toggleSidebar(false);
    if ($('railExpandBtn')) $('railExpandBtn').onclick = () => toggleSidebar(true);
    if ($('mobileMenuToggleBtn')) $('mobileMenuToggleBtn').onclick = () => toggleSidebar();
    if ($('mobileNewChatBtn')) $('mobileNewChatBtn').onclick = () => {
      $('newChat').click();
      if (window.innerWidth <= 860) toggleSidebar(false);
    };
    if ($('mobileUserAuthBtn')) $('mobileUserAuthBtn').onclick = () => {
      $('userAuthBtn').click();
      if (window.innerWidth <= 860) toggleSidebar(false);
    };
    if ($('railNewChatBtn')) $('railNewChatBtn').onclick = () => {
      $('newChat').click();
      if (window.innerWidth <= 860) toggleSidebar(false);
    };
    if ($('railUserBtn')) $('railUserBtn').onclick = () => $('userAuthBtn').click();
  }

  let currentConversationId = localStorage.getItem('aster_active_conv_id') || '';
  let activeChatHistory = [];

  // Detailed Answer Mode (Astracore 3.1 toggle)
  let isDetailedMode = localStorage.getItem('astra_detailed_mode') === 'true';

  // Engine Modes: 'general' (Quick/Extended) | 'code'
  let activeEngineMode = 'general';

  // Enhanced configurable code-request detector
  const codeRequestConfig = {
    verbs: ['write','create','build','implement','generate','make','code','program','develop','script','design','show','give','provide'],
    langs: ['python','javascript','typescript','java','c++','cpp','c#','csharp','rust','go','golang','php','ruby','kotlin','swift','sql','bash','shell','html','css','react','node','django','flask','express','vue','angular','ts','js','py'],
    nouns: ['function','class','method','algorithm','snippet','program','script','code','api','endpoint','component','module','library','implementation','binary search','linked list','factorial','fibonacci','sort','recursion','loop','array','stack','queue','tree','graph','example']
  };
  const escapeRegExp = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const codeVerbRegex = new RegExp('\\b(' + codeRequestConfig.verbs.map(escapeRegExp).join('|') + ')\\b','i');
  const codeLangRegex = new RegExp('\\b(' + codeRequestConfig.langs.map(escapeRegExp).join('|') + ')\\b','i');
  const codeNounRegex = new RegExp('\\b(' + codeRequestConfig.nouns.map(escapeRegExp).join('|') + ')\\b','i');

  function _isCodeRequest(text) {
    const q = text.toLowerCase();
    const hasVerb = codeVerbRegex.test(q);
    const hasLang = codeLangRegex.test(q);
    const hasNoun = codeNounRegex.test(q);
    if (hasVerb && (hasLang || hasNoun)) return true;
    if (hasLang && hasNoun) return true;
    if (/\\b(give me|show me|write|provide)\\s+(a\\s+)?(code|program|script|implementation|function|class|example)\\b/i.test(q)) return true;
    if (/\\b(debug|fix|refactor|explain this code|optimize this code)\\b/i.test(q)) return true;
    return false;
  }

  // Debounce helper
  function debounce(fn, delay) {
    let timer;
    return function(...args) {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), delay);
    };
  }

  const evaluateCodeRequest = debounce(function(message) {
    if (_isCodeRequest(message)) {
      setEngineMode('code');
    } else {
      setEngineMode('general');
    }
  }, 300);

  function setEngineMode(mode) {
    activeEngineMode = mode;
    updateEngineModeUI();
    const label = mode === 'code' ? '💻 Code Mode' : '✨ General Mode';
    toast(`✓ ${label} Active`);
  }

  function updateEngineModeUI() {
    const codeCheck = $('popoverCodeCheck');
    const codeRow = $('popoverModeCode');
    if (codeCheck) codeCheck.classList.toggle('hidden', activeEngineMode !== 'code');
    if (codeRow) codeRow.classList.toggle('active', activeEngineMode === 'code');
    const label = $('engineLabel');
    if (label) {
      label.textContent = activeEngineMode === 'code' ? 'Astracore Code' : 'Astracore 3.1';
    }
  }

  function updateEngineLabelUI() {
    updateEngineModeUI();
  }

  function setDetailedMode(detailed) {
    isDetailedMode = !!detailed;
    localStorage.setItem('astra_detailed_mode', isDetailedMode);
    
    // Update checkmarks in popover
    const quickCheck = $('popoverQuickCheck');
    const extCheck = $('popoverExtendedCheck');
    const quickRow = $('popoverModeQuick');
    const extRow = $('popoverModeExtended');

    if (quickCheck) quickCheck.classList.toggle('hidden', isDetailedMode);
    if (extCheck) extCheck.classList.toggle('hidden', !isDetailedMode);
    if (quickRow) quickRow.classList.toggle('active', !isDetailedMode);
    if (extRow) extRow.classList.toggle('active', isDetailedMode);

    toast(isDetailedMode ? '⚡ Extended Mode Active: Detailed and deep answers.' : '✨ Quick Mode Active: Concise and fast answers.');
  }

  function toggleDetailedMode() {
    setDetailedMode(!isDetailedMode);
  }

  // ─── Incognito / Temporary Chat Mode State Management ──────────────────────
  // Architecture:
  //  • On ACTIVATION  – snapshot (convId, activeChatHistory, DOM nodes) and show banner.
  //  • During mode    – messages are tagged data-temporary="true" by addMessage().
  //                     activeTempHistory accumulates the temp exchange so the LLM keeps context
  //                     within the temp session, but normal activeChatHistory is NOT modified.
  //                     Backend receives incognito:true so it labels messages "temporary" and
  //                     skips QueryEvent logging.
  //  • On DEACTIVATION – remove all [data-temporary] DOM nodes, restore snapshot, hide banner.
  // ────────────────────────────────────────────────────────────────────────────
  let isIncognito = false;
  let savedNormalConvId = '';
  let savedNormalChatHistory = [];   // deep-copy of activeChatHistory at activation moment
  let savedNormalWelcomeHidden = false;
  let savedNormalMessagesNodes = []; // preserved real DOM message nodes of public chat
  let activeTempHistory = [];        // in-session history for temp LLM context (not persisted)

  function _ensureTempBanner() {
    let banner = document.getElementById('tempChatBanner');
    if (!banner) {
      banner = document.createElement('div');
      banner.id = 'tempChatBanner';
      banner.setAttribute('aria-live', 'polite');
      // Only layout/position properties here — colour is handled by CSS (#tempChatBanner rules)
      banner.style.cssText = [
        'display:none',
        'position:absolute',
        'top:8px',
        'left:12px',
        'z-index:20',
        'border-radius:8px',
        'padding:4px 10px',
        'font-size:11px',
        'font-weight:500',
        'letter-spacing:0.4px',
        'pointer-events:none',
        'white-space:nowrap',
      ].join(';');
      banner.textContent = '🕶️ Temporary mode activated';
      // Attach inside #centerStage so it floats inside the chat area
      const stage = $('centerStage');
      if (stage) {
        stage.style.position = 'relative'; // ensure positioning context
        stage.appendChild(banner);
      }
    }
    return banner;
  }

  function setIncognitoMode(active) {
    if (isIncognito === active) return;
    isIncognito = active;

    // ── Sync all toggle controls ────────────────────────────────────────────
    const toggle = $('tempChatToggle');
    if (toggle && toggle.checked !== isIncognito) toggle.checked = isIncognito;
    if ($('sidebarDisappearingBtn')) $('sidebarDisappearingBtn').classList.toggle('active', isIncognito);

    const banner = _ensureTempBanner();
    const msgContainer = $('messages');
    const welcomeEl = $('welcome');

    if (isIncognito) {
      // ── ACTIVATING TEMPORARY CHAT ─────────────────────────────────────────
      // 1. Snapshot current persistent state so we can restore it on deactivation
      savedNormalConvId = currentConversationId;
      savedNormalChatHistory = activeChatHistory.slice(); // shallow copy of plain objects
      savedNormalWelcomeHidden = welcomeEl ? welcomeEl.classList.contains('hidden') : false;
      activeTempHistory = [];

      // 2. Preserve public chat DOM nodes safely in an array and clear container for temporary chat
      savedNormalMessagesNodes = [];
      if (msgContainer) {
        while (msgContainer.firstChild) {
          savedNormalMessagesNodes.push(msgContainer.firstChild);
          msgContainer.removeChild(msgContainer.firstChild);
        }
      }

      // 3. Show clean temporary chat greeting interface
      if (welcomeEl) {
        welcomeEl.classList.remove('hidden');
      }
      updateStageMode();

      // 4. Apply CSS mode
      document.body.classList.add('temporary-chat-mode', 'incognito-mode');

      // 5. Show banner
      banner.style.display = 'block';

      toast('🕶️ Temporary Chat active — messages will not be saved.');
    } else {
      // ── DEACTIVATING TEMPORARY CHAT (RESTORE PUBLIC CHAT) ──────────────────
      // 1. Remove all temporary messages from DOM
      if (msgContainer) {
        msgContainer.innerHTML = '';
        // Restore all previous public chat DOM nodes exactly as they were
        savedNormalMessagesNodes.forEach((node) => {
          msgContainer.appendChild(node);
        });
      }
      savedNormalMessagesNodes = [];

      // 2. Restore pre-temporary persistent state
      currentConversationId = savedNormalConvId;
      activeChatHistory = savedNormalChatHistory.slice();
      activeTempHistory = [];

      // 3. Restore welcome screen state based on public chat snapshot
      if (welcomeEl) {
        if (savedNormalWelcomeHidden && msgContainer && msgContainer.children.length > 0) {
          welcomeEl.classList.add('hidden');
        } else {
          welcomeEl.classList.remove('hidden');
        }
        updateStageMode();
      }

      // 4. Remove CSS mode
      document.body.classList.remove('temporary-chat-mode', 'incognito-mode');

      // 5. Hide banner
      banner.style.display = 'none';

      // 6. Scroll to latest restored message if messages exist
      if (msgContainer && msgContainer.children.length > 0) {
        msgContainer.scrollTop = msgContainer.scrollHeight;
      }

      toast('Temporary Chat turned off. Previous conversation restored.');
    }
  }

  function toggleIncognitoMode() {
    setIncognitoMode(!isIncognito);
  }

  function updateAuthUI() {
    const displayName = token ? ((user && user.username) || cachedUsername || (user && user.email ? user.email.split('@')[0] : '')) : '';
    if (displayName) {
      $('userAccountLabel').textContent = displayName.toUpperCase();
      if ($('userAuthBtn')) $('userAuthBtn').setAttribute('title', `Logged in as ${displayName} (Click for Account / Sign Out)`);
      if ($('mobileUserAuthBtn')) {
        $('mobileUserAuthBtn').setAttribute('title', `Logged in as ${displayName} (Click for Account / Sign Out)`);
        $('mobileUserAuthBtn').classList.add('logged-in');
      }
      if ($('loginForm')) $('loginForm').classList.add('hidden');
      if ($('signupForm')) $('signupForm').classList.add('hidden');
      $('authLogoutSection').classList.remove('hidden');
      $('loggedInUserEmail').textContent = `${displayName}${user && user.email ? ` (${user.email})` : ''}`;
    } else {
      $('userAccountLabel').textContent = 'LOGIN';
      if ($('userAuthBtn')) $('userAuthBtn').setAttribute('title', 'Sign In / Register');
      if ($('mobileUserAuthBtn')) {
        $('mobileUserAuthBtn').setAttribute('title', 'Sign In / Register');
        $('mobileUserAuthBtn').classList.remove('logged-in');
      }
      $('authLogoutSection').classList.add('hidden');
      switchAuthMode(authMode);
    }
  }

  async function checkHealth() {
    try {
      const data = await api('/api/health');
      updateEngineLabelUI();
      if ($('statusText')) $('statusText').textContent = 'Ready';
      if ($('serviceStatus')) $('serviceStatus').classList.add('ok');
    } catch (_) {
      if ($('statusText')) $('statusText').textContent = 'Connecting…';
      if ($('serviceStatus')) $('serviceStatus').classList.remove('ok');
    }
  }

  // Function to update the small document tag on top of the composer textbox
  function updateComposerDocTag(docId, docName) {
    const tag = $('activeDocTag');
    const nameEl = $('activeDocName');
    if (!tag || !nameEl) return;
    if (docId && docName) {
      nameEl.textContent = docName;
      tag.classList.remove('hidden');
    } else {
      tag.classList.add('hidden');
      nameEl.textContent = '';
    }
  }

  async function refreshDocuments() {
    try {
      const data = await api('/api/documents');
      $('docCount').textContent = data.total_documents;
      
      let optionsHtml = '<option value="">All indexed documents</option>';
      data.documents.forEach((d) => {
        optionsHtml += `<option value="${d.id}">${escapeHtml(d.name)}</option>`;
      });
      $('documentSelect').innerHTML = optionsHtml;

      if (!data.documents.length) {
        $('documentList').innerHTML = '<p class="muted-box-msg">No documents uploaded yet</p>';
        updateComposerDocTag(null, null);
        return;
      }

      let pendingDeleteId = null;

      $('documentList').innerHTML = data.documents
        .map(
          (d) => `
        <div class="document-item" data-doc-id="${d.id}">
          <div style="flex: 1; min-width: 0;">
            <strong>${escapeHtml(d.name)}</strong>
            <small>${d.chunks_count} chunks · ${escapeHtml(d.status)}</small>
          </div>
          <button data-delete="${d.id}" data-doc-name="${escapeHtml(d.name)}" title="Delete document">×</button>
        </div>`
        )
        .join('');

      // Restore active highlight if document currently selected
      const currentSelectedId = $('documentSelect')?.value;
      if (currentSelectedId) {
        const found = data.documents.find((d) => String(d.id) === String(currentSelectedId));
        if (found) {
          updateComposerDocTag(found.id, found.name);
          document.querySelectorAll('.document-item').forEach((el) => {
            el.classList.toggle('active', el.dataset.docId === String(found.id));
          });
        }
      }

      // Handle (X) remove document chip on top of textbox
      if ($('removeActiveDocBtn')) {
        $('removeActiveDocBtn').onclick = () => {
          if ($('documentSelect')) $('documentSelect').value = '';
          document.querySelectorAll('.document-item').forEach((el) => el.classList.remove('active'));
          updateComposerDocTag(null, null);
          toast('Document focus removed. Switched to General Search Mode.');
        };
      }

      // Click on document row to switch active document query context
      document.querySelectorAll('.document-item').forEach((item) => {
        item.onclick = (e) => {
          if (e.target.closest('[data-delete]')) return;
          const docId = item.dataset.docId;
          const select = $('documentSelect');
          const docName = item.querySelector('strong')?.textContent || 'Document';
          if (select.value === docId) {
            select.value = '';
            item.classList.remove('active');
            updateComposerDocTag(null, null);
            toast('Switched to General Search Mode (All documents & Internet)');
          } else {
            select.value = docId;
            document.querySelectorAll('.document-item').forEach((el) => el.classList.remove('active'));
            item.classList.add('active');
            updateComposerDocTag(docId, docName);
            toast(`Switched query focus to: ${docName}`);
          }
        };
      });

      // Update active highlight & composer chip when dropdown changes
      $('documentSelect').onchange = () => {
        const val = $('documentSelect').value;
        let selectedDocName = '';
        document.querySelectorAll('.document-item').forEach((el) => {
          const isMatch = el.dataset.docId === val;
          el.classList.toggle('active', isMatch);
          if (isMatch) selectedDocName = el.querySelector('strong')?.textContent || 'Document';
        });
        updateComposerDocTag(val || null, selectedDocName || null);
      };

      // Custom Delete Confirmation Modal (no browser alert!)
      document.querySelectorAll('[data-delete]').forEach((btn) => {
        btn.onclick = (e) => {
          e.stopPropagation();
          pendingDeleteId = btn.dataset.delete;
          const docName = btn.dataset.docName || 'this document';
          if ($('deleteDocNameText')) {
            $('deleteDocNameText').textContent = `Are you sure you want to remove "${docName}" from your knowledge repository?`;
          }
          if ($('deleteConfirmModal')) $('deleteConfirmModal').classList.remove('hidden');
        };
      });

      if ($('cancelDeleteBtn')) {
        $('cancelDeleteBtn').onclick = () => {
          pendingDeleteId = null;
          if ($('deleteConfirmModal')) $('deleteConfirmModal').classList.add('hidden');
        };
      }

      if ($('confirmDeleteBtn')) {
        $('confirmDeleteBtn').onclick = async () => {
          if (!pendingDeleteId) return;
          const toDelete = pendingDeleteId;
          pendingDeleteId = null;
          if ($('deleteConfirmModal')) $('deleteConfirmModal').classList.add('hidden');
          try {
            await api(`/api/documents/${toDelete}`, { method: 'DELETE' });
            toast('Document removed from knowledge repository');
            refreshDocuments();
          } catch (err) {
            toast(err.message, true);
          }
        };
      }
    } catch (err) {
      console.warn('Failed to load documents:', err);
    }
  }

  // Initialize dynamic interactive widgets (Three.js 3D models, Chart.js charts, Mermaid diagrams, Code copy buttons)
  function initInteractiveWidgets(scope) {
    if (!scope) return;

    // 1. Copy Code Block Buttons (Pure raw code, no fences/markdown artifacts)
    scope.querySelectorAll('.copy-code-btn').forEach((btn) => {
      btn.onclick = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        const card = btn.closest('.code-editor-card, .code-block-card');
        const codeElem = card ? card.querySelector('.code-editor-code, code, pre') : null;
        let textToCopy = codeElem ? (codeElem.innerText || codeElem.textContent || '') : '';
        // Pure code: strip surrounding markdown code fences or trailing artifacts
        textToCopy = textToCopy.replace(/^```[a-zA-Z0-9_-]*\n?/, '').replace(/\n?```$/, '');
        if (!textToCopy) return;

        const label = btn.querySelector('.copy-btn-label') || btn;
        const origText = label.textContent;

        const setCopied = () => {
          btn.classList.add('copied');
          label.textContent = 'Copied!';
          toast('✓ Code copied to clipboard');
          setTimeout(() => {
            btn.classList.remove('copied');
            label.textContent = origText;
          }, 2000);
        };

        if (navigator.clipboard && navigator.clipboard.writeText) {
          try {
            await navigator.clipboard.writeText(textToCopy);
            setCopied();
            return;
          } catch (_) {}
        }

        try {
          const ta = document.createElement('textarea');
          ta.value = textToCopy;
          ta.style.position = 'fixed';
          ta.style.opacity = '0';
          document.body.appendChild(ta);
          ta.select();
          document.execCommand('copy');
          document.body.removeChild(ta);
          setCopied();
        } catch (err) {
          toast('Could not copy code: ' + err.message, true);
        }
      };
    });

    // 1a. Download Code File Buttons
    scope.querySelectorAll('.download-code-btn').forEach((btn) => {
      btn.onclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        const card = btn.closest('.code-editor-card, .code-block-card');
        const codeElem = card ? card.querySelector('.code-editor-code, code, pre') : null;
        let codeText = codeElem ? (codeElem.innerText || codeElem.textContent || '') : '';
        codeText = codeText.replace(/^```[a-zA-Z0-9_-]*\n?/, '').replace(/\n?```$/, '');
        if (!codeText) return;

        const rawLang = (btn.dataset.lang || (card && card.dataset.language) || '').toLowerCase();
        const langMap = {
          html: 'index.html',
          htm: 'index.html',
          css: 'style.css',
          javascript: 'script.js',
          js: 'script.js',
          typescript: 'app.ts',
          ts: 'app.ts',
          jsx: 'App.jsx',
          tsx: 'App.tsx',
          python: 'main.py',
          py: 'main.py',
          json: 'data.json',
          sql: 'query.sql',
          bash: 'script.sh',
          sh: 'script.sh',
          c: 'main.c',
          cpp: 'main.cpp',
          'c++': 'main.cpp',
          java: 'Main.java',
          rust: 'main.rs',
          rs: 'main.rs',
          go: 'main.go',
          markdown: 'README.md',
          md: 'README.md',
          xml: 'data.xml',
          yaml: 'config.yaml',
          yml: 'config.yaml',
        };

        const filename = langMap[rawLang] || (rawLang ? `code.${rawLang}` : 'snippet.txt');
        const blob = new Blob([codeText], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        toast(`Downloaded ${filename}`);
      };
    });

    // 1b. Syntax Highlighting via highlight.js
    if (window.hljs) {
      scope.querySelectorAll('.code-editor-card pre code').forEach((codeBlock) => {
        try {
          if (!codeBlock.dataset.highlighted) {
            window.hljs.highlightElement(codeBlock);
            codeBlock.dataset.highlighted = 'yes';
          }
        } catch (_) {}
      });
    }

    // 1c. Live Demo Button Handler
    scope.querySelectorAll('.live-demo-btn').forEach((btn) => {
      btn.onclick = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        // Find the parent message container to collect all code blocks
        const card = btn.closest('.code-editor-card');
        if (!card) return;
        const messageEl = card.closest('.message-bubble') || card.closest('[class*="message"]') || card.parentElement;
        // Collect HTML / CSS / JS from all code cards in same message
        const allCards = messageEl ? messageEl.querySelectorAll('.code-editor-card') : [card];
        let html = '', css = '', js = '';
        allCards.forEach((c) => {
          const lang = (c.dataset.language || '').toLowerCase();
          const codeEl = c.querySelector('.code-editor-code');
          const code = codeEl ? (codeEl.innerText || codeEl.textContent) : '';
          if (lang === 'html') html = code;
          else if (lang === 'css') css = code;
          else if (lang === 'javascript' || lang === 'js') js = code;
        });
        // If the only block is HTML, use it directly
        if (!html && !css && !js) {
          const codeEl = card.querySelector('.code-editor-code');
          html = codeEl ? (codeEl.innerText || codeEl.textContent) : '';
        }
        const origText = btn.textContent;
        btn.textContent = 'Launching…';
        btn.disabled = true;
        try {
          const resp = await fetch('/api/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ html, css, js })
          });
          const data = await resp.json();
          if (data && data.url) {
            window.open(data.url, '_blank');
          }
        } catch (err) {
          console.warn('Live demo error:', err);
          toast('Could not open Live Demo: ' + err.message, true);
        } finally {
          btn.textContent = origText;
          btn.disabled = false;
        }
      };
    });

    // 1d. Close Run Drawer Button
    scope.querySelectorAll('.close-run-btn').forEach((btn) => {
      btn.onclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        const drawer = btn.closest('.code-run-drawer');
        if (drawer) drawer.classList.add('hidden');
      };
    });

    // 1e. Multi-file Code Tabs: when a message has multiple code blocks, add tab switcher
    const containers = scope.querySelectorAll('[class*="message"]');
    containers.forEach((container) => {
      const codeCards = container.querySelectorAll('.code-editor-card');
      if (codeCards.length > 1 && !container.querySelector('.code-editor-tabs-bar')) {
        const tabsBar = document.createElement('div');
        tabsBar.className = 'code-editor-tabs-bar';
        codeCards.forEach((c, idx) => {
          const lang = (c.dataset.language || 'code').toUpperCase();
          const tab = document.createElement('div');
          tab.className = 'code-editor-tab' + (idx === 0 ? ' active' : '');
          tab.textContent = lang;
          tab.onclick = () => {
            codeCards.forEach((card) => card.style.display = 'none');
            c.style.display = '';
            tabsBar.querySelectorAll('.code-editor-tab').forEach((t) => t.classList.remove('active'));
            tab.classList.add('active');
          };
          if (idx !== 0) c.style.display = 'none';
          tabsBar.appendChild(tab);
        });
        codeCards[0].parentElement.insertBefore(tabsBar, codeCards[0]);
      }
    });


    // 2. Mermaid Diagrams
    if (window.mermaid) {
      try {
        const nodes = scope.querySelectorAll('.interactive-mermaid-card .mermaid');
        if (nodes.length) {
          window.mermaid.initialize({ startOnLoad: false, theme: 'neutral', securityLevel: 'loose' });
          window.mermaid.run({ nodes }).catch((e) => console.warn('Mermaid rendering:', e));
        }
      } catch (e) {
        console.warn('Mermaid init:', e);
      }
    }

    // 3. Interactive Charts (Chart.js)
    scope.querySelectorAll('.interactive-chart-card').forEach((card) => {
      if (card.dataset.initialized) return;
      card.dataset.initialized = 'true';
      let spec = {};
      try {
        spec = JSON.parse(card.dataset.spec || '{}');
      } catch (e) {
        console.warn('Invalid chart spec', e);
      }

      const canvas = card.querySelector('.interactive-chart-canvas');
      if (!canvas || !window.Chart) return;

      try {
        const isRadial = ['pie', 'doughnut', 'radar', 'polarArea'].includes(spec.type);
        const chart = new Chart(canvas, {
          type: spec.type || 'bar',
          data: spec.data || { labels: [], datasets: [] },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                position: 'top',
                labels: { color: '#334155', font: { family: 'Inter', size: 12, weight: '500' } }
              },
              tooltip: {
                backgroundColor: 'rgba(15, 23, 42, 0.92)',
                titleFont: { family: 'Inter', weight: '600' },
                bodyFont: { family: 'Inter' },
                padding: 10,
                cornerRadius: 8
              }
            },
            scales: isRadial ? {} : {
              x: {
                grid: { color: 'rgba(0, 0, 0, 0.04)' },
                ticks: { color: '#64748b', font: { family: 'Inter', size: 11 } }
              },
              y: {
                grid: { color: 'rgba(0, 0, 0, 0.04)' },
                ticks: { color: '#64748b', font: { family: 'Inter', size: 11 } }
              }
            },
            ...(spec.options || {})
          }
        });

        const dlBtn = card.querySelector('.chart-download-btn');
        if (dlBtn) {
          dlBtn.onclick = () => {
            const link = document.createElement('a');
            link.href = chart.toBase64Image();
            link.download = `${(spec.title || 'chart').toLowerCase().replace(/[^a-z0-9]+/g, '_')}.png`;
            link.click();
          };
        }
      } catch (err) {
        console.warn('Failed to build chart:', err);
      }
    });

    // 4. Interactive 3D WebGL Models (Three.js + OrbitControls)
    scope.querySelectorAll('.interactive-3d-card').forEach((card) => {
      if (card.dataset.initialized) return;
      card.dataset.initialized = 'true';
      let spec = {};
      try {
        spec = JSON.parse(card.dataset.spec || '{}');
      } catch (e) {
        console.warn('Invalid 3D spec', e);
      }

      const canvas = card.querySelector('.interactive-3d-canvas');
      const viewport = card.querySelector('.interactive-3d-viewport');
      if (!canvas || !window.THREE) return;

      try {
        const width = viewport.clientWidth || 640;
        const height = 360;

        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.15;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0f1d);

        // Ambient star dust particles in background
        const starGeom = new THREE.BufferGeometry();
        const starCount = 280;
        const starCoords = new Float32Array(starCount * 3);
        for (let i = 0; i < starCount * 3; i += 3) {
          starCoords[i] = (Math.random() - 0.5) * 75;
          starCoords[i + 1] = (Math.random() - 0.5) * 75;
          starCoords[i + 2] = (Math.random() - 0.5) * 75;
        }
        starGeom.setAttribute('position', new THREE.BufferAttribute(starCoords, 3));
        const starMat = new THREE.PointsMaterial({ color: 0x94a3b8, size: 0.22, transparent: true, opacity: 0.55 });
        scene.add(new THREE.Points(starGeom, starMat));

        // Lights
        const ambLight = new THREE.AmbientLight(0xffffff, 0.7);
        scene.add(ambLight);

        const dir1 = new THREE.DirectionalLight(0x38bdf8, 1.4);
        dir1.position.set(20, 25, 20);
        scene.add(dir1);

        const dir2 = new THREE.DirectionalLight(0xec4899, 1.0);
        dir2.position.set(-20, -15, -20);
        scene.add(dir2);

        const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        camera.position.set(0, 8, 26);

        let controls = null;
        if (window.THREE.OrbitControls) {
          controls = new THREE.OrbitControls(camera, canvas);
          controls.enableDamping = true;
          controls.dampingFactor = 0.06;
          controls.autoRotate = true;
          controls.autoRotateSpeed = 1.8;
          controls.minDistance = 4;
          controls.maxDistance = 120;
        }

        const type = (spec.type || 'dna').toLowerCase();
        const params = spec.params || {};
        let customUpdate = null;

        const createCylinder = (p1, p2, radius, material) => {
          const dir = new THREE.Vector3().subVectors(p2, p1);
          const len = dir.length();
          const geom = new THREE.CylinderGeometry(radius, radius, len, 12);
          const mesh = new THREE.Mesh(geom, material);
          mesh.position.copy(p1).addScaledVector(dir, 0.5);
          mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
          return mesh;
        };

        // 3D PRESET BUILDERS
        if (type.includes('dna') || type.includes('helix')) {
          const dnaGroup = new THREE.Group();
          const pairs = params.pairs || 22;
          const radius = 4.2;
          const heightStep = 1.0;
          const twistStep = 0.36;

          const sphereGeom = new THREE.SphereGeometry(0.38, 16, 16);
          const matA = new THREE.MeshStandardMaterial({ color: 0x38bdf8, metalness: 0.2, roughness: 0.3, emissive: 0x0284c7, emissiveIntensity: 0.25 });
          const matB = new THREE.MeshStandardMaterial({ color: 0xec4899, metalness: 0.2, roughness: 0.3, emissive: 0xdb2777, emissiveIntensity: 0.25 });
          const matAT = new THREE.MeshStandardMaterial({ color: 0x10b981, roughness: 0.4 });
          const matGC = new THREE.MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.4 });

          const startY = -(pairs * heightStep) / 2;
          for (let i = 0; i < pairs; i++) {
            const y = startY + i * heightStep;
            const ang = i * twistStep;
            const x1 = Math.cos(ang) * radius;
            const z1 = Math.sin(ang) * radius;
            const x2 = Math.cos(ang + Math.PI) * radius;
            const z2 = Math.sin(ang + Math.PI) * radius;

            const s1 = new THREE.Mesh(sphereGeom, matA);
            s1.position.set(x1, y, z1);
            dnaGroup.add(s1);

            const s2 = new THREE.Mesh(sphereGeom, matB);
            s2.position.set(x2, y, z2);
            dnaGroup.add(s2);

            if (i > 0) {
              const prevAng = (i - 1) * twistStep;
              const px1 = Math.cos(prevAng) * radius;
              const py = startY + (i - 1) * heightStep;
              const pz1 = Math.sin(prevAng) * radius;
              dnaGroup.add(createCylinder(new THREE.Vector3(px1, py, pz1), new THREE.Vector3(x1, y, z1), 0.12, matA));

              const px2 = Math.cos(prevAng + Math.PI) * radius;
              const pz2 = Math.sin(prevAng + Math.PI) * radius;
              dnaGroup.add(createCylinder(new THREE.Vector3(px2, py, pz2), new THREE.Vector3(x2, y, z2), 0.12, matB));
            }

            const rungMat = (i % 2 === 0) ? matAT : matGC;
            dnaGroup.add(createCylinder(new THREE.Vector3(x1, y, z1), new THREE.Vector3(x2, y, z2), 0.15, rungMat));
          }
          scene.add(dnaGroup);

        } else if (type.includes('molecule') || type.includes('chemical') || type.includes('benzene') || type.includes('water')) {
          const molGroup = new THREE.Group();
          const molName = (params.molecule || spec.title || '').toLowerCase();

          const atomMats = {
            H: new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.3 }),
            C: new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.1, roughness: 0.4 }),
            O: new THREE.MeshStandardMaterial({ color: 0xef4444, roughness: 0.3 }),
            N: new THREE.MeshStandardMaterial({ color: 0x3b82f6, roughness: 0.3 }),
            S: new THREE.MeshStandardMaterial({ color: 0xeab308, roughness: 0.3 }),
            Cl: new THREE.MeshStandardMaterial({ color: 0x22c55e, roughness: 0.3 }),
          };
          const bondMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.6, roughness: 0.3 });

          if (molName.includes('water') || molName.includes('h2o')) {
            const oMesh = new THREE.Mesh(new THREE.SphereGeometry(1.2, 24, 24), atomMats.O);
            oMesh.position.set(0, 0.4, 0);
            molGroup.add(oMesh);

            const h1 = new THREE.Mesh(new THREE.SphereGeometry(0.7, 20, 20), atomMats.H);
            h1.position.set(-1.8, -0.8, 0);
            molGroup.add(h1);

            const h2 = new THREE.Mesh(new THREE.SphereGeometry(0.7, 20, 20), atomMats.H);
            h2.position.set(1.8, -0.8, 0);
            molGroup.add(h2);

            molGroup.add(createCylinder(oMesh.position, h1.position, 0.16, bondMat));
            molGroup.add(createCylinder(oMesh.position, h2.position, 0.16, bondMat));

          } else if (molName.includes('methane') || molName.includes('ch4')) {
            const cMesh = new THREE.Mesh(new THREE.SphereGeometry(1.2, 24, 24), atomMats.C);
            molGroup.add(cMesh);
            const hCoords = [[1.5, 1.5, 1.5], [-1.5, -1.5, 1.5], [-1.5, 1.5, -1.5], [1.5, -1.5, -1.5]];
            hCoords.forEach(([x, y, z]) => {
              const h = new THREE.Mesh(new THREE.SphereGeometry(0.7, 20, 20), atomMats.H);
              h.position.set(x, y, z);
              molGroup.add(h);
              molGroup.add(createCylinder(new THREE.Vector3(0,0,0), h.position, 0.16, bondMat));
            });

          } else {
            // Benzene C6H6 aromatic ring
            const rC = 3.6;
            const rH = 5.6;
            const cPositions = [];
            for (let i = 0; i < 6; i++) {
              const a = (i * Math.PI) / 3;
              const cx = Math.cos(a) * rC;
              const cy = Math.sin(a) * rC;
              const cMesh = new THREE.Mesh(new THREE.SphereGeometry(0.9, 20, 20), atomMats.C);
              cMesh.position.set(cx, cy, 0);
              molGroup.add(cMesh);
              cPositions.push(cMesh.position);

              const hx = Math.cos(a) * rH;
              const hy = Math.sin(a) * rH;
              const hMesh = new THREE.Mesh(new THREE.SphereGeometry(0.55, 16, 16), atomMats.H);
              hMesh.position.set(hx, hy, 0);
              molGroup.add(hMesh);
              molGroup.add(createCylinder(cMesh.position, hMesh.position, 0.12, bondMat));
            }
            for (let i = 0; i < 6; i++) {
              molGroup.add(createCylinder(cPositions[i], cPositions[(i + 1) % 6], 0.18, bondMat));
            }
          }
          scene.add(molGroup);

        } else if (type.includes('solar') || type.includes('planet')) {
          const solarGroup = new THREE.Group();
          const sun = new THREE.Mesh(new THREE.SphereGeometry(2.8, 32, 32), new THREE.MeshBasicMaterial({ color: 0xfbbf24 }));
          solarGroup.add(sun);
          solarGroup.add(new THREE.PointLight(0xffedd5, 2.5, 80));

          const planetData = [
            { r: 5.5, size: 0.5, color: 0x94a3b8, speed: 0.04 },
            { r: 8.0, size: 0.7, color: 0xf59e0b, speed: 0.025 },
            { r: 11.5, size: 0.85, color: 0x38bdf8, speed: 0.018 },
            { r: 15.0, size: 0.65, color: 0xef4444, speed: 0.014 },
            { r: 20.0, size: 1.6, color: 0xd97706, speed: 0.008 },
          ];

          const planetMeshes = [];
          planetData.forEach((p) => {
            const ringGeom = new THREE.RingGeometry(p.r - 0.06, p.r + 0.06, 64);
            const orbitLine = new THREE.Mesh(ringGeom, new THREE.MeshBasicMaterial({ color: 0x334155, side: THREE.DoubleSide }));
            orbitLine.rotation.x = Math.PI / 2;
            solarGroup.add(orbitLine);

            const pMesh = new THREE.Mesh(new THREE.SphereGeometry(p.size, 20, 20), new THREE.MeshStandardMaterial({ color: p.color, roughness: 0.4 }));
            solarGroup.add(pMesh);
            planetMeshes.push({ mesh: pMesh, r: p.r, speed: p.speed, angle: Math.random() * Math.PI * 2 });
          });

          customUpdate = () => {
            planetMeshes.forEach((item) => {
              item.angle += item.speed;
              item.mesh.position.set(Math.cos(item.angle) * item.r, 0, Math.sin(item.angle) * item.r);
            });
          };
          scene.add(solarGroup);

        } else if (type.includes('atom') || type.includes('orbital')) {
          const atomGroup = new THREE.Group();
          const nucGroup = new THREE.Group();
          const pMat = new THREE.MeshStandardMaterial({ color: 0xef4444, roughness: 0.3 });
          const nMat = new THREE.MeshStandardMaterial({ color: 0x3b82f6, roughness: 0.3 });
          for (let i = 0; i < 14; i++) {
            const mat = (i % 2 === 0) ? pMat : nMat;
            const sp = new THREE.Mesh(new THREE.SphereGeometry(0.5, 16, 16), mat);
            sp.position.set((Math.random() - 0.5) * 1.6, (Math.random() - 0.5) * 1.6, (Math.random() - 0.5) * 1.6);
            nucGroup.add(sp);
          }
          atomGroup.add(nucGroup);

          const orbits = [
            { rx: 6, ry: 4, rotX: 0.4, rotY: 0.8, speed: 0.05 },
            { rx: 6.5, ry: 4.2, rotX: -0.6, rotY: 0.4, speed: -0.04 },
            { rx: 7, ry: 4.5, rotX: 1.2, rotY: -0.5, speed: 0.06 },
          ];
          const electronMeshes = [];
          orbits.forEach((orb) => {
            const curve = new THREE.EllipseCurve(0, 0, orb.rx, orb.ry, 0, 2 * Math.PI, false, 0);
            const pts = curve.getPoints(64);
            const geom = new THREE.BufferGeometry().setFromPoints(pts.map(p => new THREE.Vector3(p.x, p.y, 0)));
            const ring = new THREE.Line(geom, new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.5 }));
            ring.rotation.x = orb.rotX;
            ring.rotation.y = orb.rotY;
            atomGroup.add(ring);

            const elMesh = new THREE.Mesh(new THREE.SphereGeometry(0.32, 16, 16), new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x38bdf8, emissiveIntensity: 0.8 }));
            atomGroup.add(elMesh);
            electronMeshes.push({ mesh: elMesh, orb, angle: Math.random() * Math.PI * 2 });
          });

          customUpdate = () => {
            electronMeshes.forEach((item) => {
              item.angle += item.orb.speed;
              const x = Math.cos(item.angle) * item.orb.rx;
              const y = Math.sin(item.angle) * item.orb.ry;
              const v = new THREE.Vector3(x, y, 0);
              v.applyAxisAngle(new THREE.Vector3(1, 0, 0), item.orb.rotX);
              v.applyAxisAngle(new THREE.Vector3(0, 1, 0), item.orb.rotY);
              item.mesh.position.copy(v);
            });
          };
          scene.add(atomGroup);

        } else if (type.includes('neural') || type.includes('network')) {
          const netGroup = new THREE.Group();
          const layers = params.layers || [3, 5, 5, 2];
          const layerSpacing = 5.0;
          const nodeSpacing = 2.0;
          const allNodes = [];

          const nodeGeom = new THREE.SphereGeometry(0.42, 16, 16);
          const nodeMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x0284c7, emissiveIntensity: 0.6, roughness: 0.2 });

          const startX = -((layers.length - 1) * layerSpacing) / 2;
          layers.forEach((count, lIdx) => {
            const layerNodes = [];
            const x = startX + lIdx * layerSpacing;
            const startY = -((count - 1) * nodeSpacing) / 2;
            for (let nIdx = 0; nIdx < count; nIdx++) {
              const y = startY + nIdx * nodeSpacing;
              const z = (Math.random() - 0.5) * 0.8;
              const node = new THREE.Mesh(nodeGeom, nodeMat);
              node.position.set(x, y, z);
              netGroup.add(node);
              layerNodes.push(node.position);
            }
            allNodes.push(layerNodes);
          });

          const lineMat = new THREE.LineBasicMaterial({ color: 0x6366f1, transparent: true, opacity: 0.35 });
          for (let l = 0; l < allNodes.length - 1; l++) {
            for (const p1 of allNodes[l]) {
              for (const p2 of allNodes[l + 1]) {
                const geom = new THREE.BufferGeometry().setFromPoints([p1, p2]);
                netGroup.add(new THREE.Line(geom, lineMat));
              }
            }
          }
          scene.add(netGroup);

        } else if (type.includes('crystal') || type.includes('lattice')) {
          const cryGroup = new THREE.Group();
          const size = 3;
          const sp = 2.5;
          const atomGeom = new THREE.SphereGeometry(0.4, 16, 16);
          const atomMat = new THREE.MeshStandardMaterial({ color: 0x10b981, metalness: 0.4, roughness: 0.2 });
          const barMat = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.7, roughness: 0.3 });

          const offset = ((size - 1) * sp) / 2;
          for (let x = 0; x < size; x++) {
            for (let y = 0; y < size; y++) {
              for (let z = 0; z < size; z++) {
                const pos = new THREE.Vector3(x * sp - offset, y * sp - offset, z * sp - offset);
                const m = new THREE.Mesh(atomGeom, atomMat);
                m.position.copy(pos);
                cryGroup.add(m);

                if (x < size - 1) cryGroup.add(createCylinder(pos, new THREE.Vector3((x + 1) * sp - offset, y * sp - offset, z * sp - offset), 0.08, barMat));
                if (y < size - 1) cryGroup.add(createCylinder(pos, new THREE.Vector3(x * sp - offset, (y + 1) * sp - offset, z * sp - offset), 0.08, barMat));
                if (z < size - 1) cryGroup.add(createCylinder(pos, new THREE.Vector3(x * sp - offset, y * sp - offset, (z + 1) * sp - offset), 0.08, barMat));
              }
            }
          }
          scene.add(cryGroup);

        } else if (type.includes('galaxy')) {
          const count = 2200;
          const gGeom = new THREE.BufferGeometry();
          const posArray = new Float32Array(count * 3);
          const colArray = new Float32Array(count * 3);
          const arms = 3;

          for (let i = 0; i < count; i++) {
            const r = Math.pow(Math.random(), 1.5) * 12;
            const armAngle = ((i % arms) * (2 * Math.PI)) / arms;
            const spiralAngle = r * 0.7 + armAngle;
            const x = Math.cos(spiralAngle) * r + (Math.random() - 0.5) * (r * 0.3);
            const z = Math.sin(spiralAngle) * r + (Math.random() - 0.5) * (r * 0.3);
            const y = (Math.random() - 0.5) * (2.5 / (r + 1));

            posArray[i * 3] = x;
            posArray[i * 3 + 1] = y;
            posArray[i * 3 + 2] = z;

            const mix = r / 12;
            colArray[i * 3] = 0.2 + 0.8 * mix;
            colArray[i * 3 + 1] = 0.6 + 0.3 * (1 - mix);
            colArray[i * 3 + 2] = 1.0;
          }
          gGeom.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
          gGeom.setAttribute('color', new THREE.BufferAttribute(colArray, 3));
          const gMat = new THREE.PointsMaterial({ size: 0.18, vertexColors: true, transparent: true, opacity: 0.85 });
          const gal = new THREE.Points(gGeom, gMat);
          scene.add(gal);
          customUpdate = () => { gal.rotation.y += 0.003; };

        } else if (type.includes('torus') || type.includes('knot') || type.includes('geometry')) {
          const knotGeom = new THREE.TorusKnotGeometry(5.5, 1.8, 128, 32, params.p || 2, params.q || 3);
          const knotMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, metalness: 0.7, roughness: 0.2, emissive: 0x075985, emissiveIntensity: 0.2 });
          scene.add(new THREE.Mesh(knotGeom, knotMat));

        } else if (spec.code) {
          try {
            const fn = new Function('scene', 'camera', 'renderer', 'THREE', 'canvas', 'controls', spec.code);
            fn(scene, camera, renderer, THREE, canvas, controls);
          } catch (errCode) {
            console.warn('Custom Three.js script error:', errCode);
          }
        } else {
          const isoGeom = new THREE.IcosahedronGeometry(6, 2);
          const isoMat = new THREE.MeshStandardMaterial({ color: 0x6366f1, metalness: 0.5, roughness: 0.25 });
          scene.add(new THREE.Mesh(isoGeom, isoMat));
        }

        // Controls Buttons
        const btnRotate = card.querySelector('.btn-rotate');
        if (btnRotate && controls) {
          btnRotate.onclick = () => {
            controls.autoRotate = !controls.autoRotate;
            btnRotate.classList.toggle('active', controls.autoRotate);
          };
          if (controls.autoRotate) btnRotate.classList.add('active');
        }

        const btnWireframe = card.querySelector('.btn-wireframe');
        if (btnWireframe) {
          let isWire = false;
          btnWireframe.onclick = () => {
            isWire = !isWire;
            btnWireframe.classList.toggle('active', isWire);
            scene.traverse((child) => {
              if (child.isMesh && child.material) {
                if (Array.isArray(child.material)) {
                  child.material.forEach((m) => (m.wireframe = isWire));
                } else {
                  child.material.wireframe = isWire;
                }
              }
            });
          };
        }

        const btnReset = card.querySelector('.btn-reset');
        if (btnReset && controls) {
          btnReset.onclick = () => {
            camera.position.set(0, 8, 26);
            controls.target.set(0, 0, 0);
            controls.update();
          };
        }

        const btnFull = card.querySelector('.btn-fullscreen');
        if (btnFull) {
          btnFull.onclick = () => {
            const isFull = card.classList.toggle('fullscreen-3d');
            btnFull.textContent = isFull ? '✕ Exit' : '⛶ Full';
            setTimeout(handleResize, 100);
          };
        }

        function animate() {
          requestAnimationFrame(animate);
          if (controls) controls.update();
          if (customUpdate) customUpdate();
          renderer.render(scene, camera);
        }
        animate();

        function handleResize() {
          const newW = viewport.clientWidth || 640;
          const newH = card.classList.contains('fullscreen-3d') ? window.innerHeight - 100 : 360;
          camera.aspect = newW / newH;
          camera.updateProjectionMatrix();
          renderer.setSize(newW, newH);
        }

        window.addEventListener('resize', handleResize);

      } catch (err3d) {
        console.warn('Failed to initialize 3D scene:', err3d);
      }
    });
  }

  // Add message: Formats user message as lavender bubble and assistant message with A* avatar & Cormorant font
  function addMessage(role, text, meta = '', sources = [], stayAtTop = false, isChemistry = false, researchTrace = null, isTemporary = false) {
    const welcome = $('welcome');
    if (welcome) {
      welcome.classList.add('hidden');
      welcome.style.display = 'none';
    }
    const stage = $('centerStage') || document.querySelector('.astra-center-stage');
    if (stage) {
      stage.classList.remove('welcome-mode');
      stage.classList.add('chat-active');
    }
    const container = $('messages');
    if (container) {
      container.style.display = 'flex';
    }
    updateStageMode();
    const row = document.createElement('article');
    const isTempMsg = isTemporary || (role === 'user' && isIncognito);
    const tempClass = isTempMsg ? ' temporary-bubble' : '';
    row.className = `message-card ${role}${tempClass}`;
    if (isTempMsg) {
      row.setAttribute('data-temporary', 'true');
    }

    if (role === 'user') {
      row.innerHTML = `<div class="user-bubble">${escapeHtml(text)}</div>`;
    } else {
      const formattedText = formatMessageText(text);
      const latencyHtml = meta ? `<span class="meta-responded">Responded in ${escapeHtml(meta)}</span>` : '';
      const chemBadgeHtml = '';

      const sourcesHtml = (sources && sources.length) ? `
        <div class="message-sources" style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px;">
          ${sources.map((s) => `<span class="source-chip" style="font-size: 11px; background: rgba(0,0,0,0.06); padding: 3px 8px; border-radius: 8px;">📄 ${escapeHtml(s.title || s.label || s.name || 'Source')}</span>`).join('')}
        </div>
      ` : '';

      row.innerHTML = `
        <div class="assistant-row">
          <div class="astra-avatar-logo">
            <img src="static/astra_logo.svg" alt="Astra Logo" class="astra-logo-img" onerror="this.src='/static/astra_logo.svg'">
          </div>
          <div class="assistant-body">
            <div class="assistant-content">${formattedText}</div>
            <div class="assistant-meta-row">
              ${latencyHtml}
              <button type="button" class="action-icon-btn copy-msg-btn" title="Copy response">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
              </button>
              <button type="button" class="action-icon-btn share-msg-btn" title="Share response">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"></path><polyline points="16 6 12 2 8 6"></polyline><line x1="12" y1="2" x2="12" y2="15"></line></svg>
              </button>
            </div>
            ${sourcesHtml}
          </div>
        </div>
      `;

      const copyBtn = row.querySelector('.copy-msg-btn');
      if (copyBtn) {
        copyBtn.onclick = () => {
          navigator.clipboard.writeText(text).then(() => toast('Copied response to clipboard'));
        };
      }

      const shareBtn = row.querySelector('.share-msg-btn');
      if (shareBtn) {
        shareBtn.onclick = () => {
          if (navigator.share) {
            navigator.share({ title: 'Astra Response', text }).catch(() => {});
          } else {
            navigator.clipboard.writeText(text).then(() => toast('Response copied to clipboard for sharing'));
          }
        };
      }
    }

    if (container) container.append(row);

    // Initialize 3D models, charts, and diagrams within this message
    initInteractiveWidgets(row);

    // Update outside-the-box code layout matching ChatGPT aesthetic
    if (stage) {
      const hasCode = Boolean(stage.querySelector('.code-editor-card') || row.querySelector('.code-editor-card') || activeEngineMode === 'code');
      stage.classList.toggle('has-code-view', hasCode);
    }

    if (stayAtTop) {
      const raf = (typeof window !== 'undefined' && window.requestAnimationFrame) ? window.requestAnimationFrame : ((cb) => setTimeout(cb, 16));
      raf(() => {
        setTimeout(() => {
          if (!container) return;
          const topPos = row.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop;
          container.scrollTo({
            top: Math.max(0, topPos - 12),
            behavior: 'smooth',
          });
        }, 20);
      });
    } else {
      if (container) container.scrollTop = container.scrollHeight;
    }
    return row;
  }
  window.addMessage = addMessage;

  // Conversations Management
  async function refreshConversations() {
    try {
      const data = await api('/api/conversations');
      const list = $('conversationList');
      if (!list) return;

      if (!data.conversations || !data.conversations.length) {
        list.innerHTML = '<p class="muted-box-msg">No conversations yet</p>';
        return;
      }

      list.innerHTML = data.conversations
        .map((c) => `
          <div class="conv-item ${c.id === currentConversationId ? 'active' : ''}" data-conv-id="${c.id}">
            <span title="${escapeHtml(c.title)}">${escapeHtml(c.title)}</span>
            <button class="conv-delete-btn" data-delete-conv="${c.id}" title="Delete conversation">×</button>
          </div>
        `)
        .join('');

      // Wire conversation item click
      list.querySelectorAll('.conv-item').forEach((item) => {
        item.onclick = async (e) => {
          if (e.target.closest('[data-delete-conv]')) return;
          const convId = item.dataset.convId;
          loadConversation(convId);
        };
      });

      // Wire conversation delete button
      list.querySelectorAll('[data-delete-conv]').forEach((btn) => {
        btn.onclick = async (e) => {
          e.stopPropagation();
          const convId = btn.dataset.deleteConv;
          try {
            await api(`/api/conversations/${convId}`, { method: 'DELETE' });
            if (currentConversationId === convId) {
              currentConversationId = '';
              localStorage.removeItem('aster_active_conv_id');
              $('messages').innerHTML = '';
              $('welcome').classList.remove('hidden');
            }
            refreshConversations();
            toast('Conversation removed');
          } catch (err) {
            toast(err.message, true);
          }
        };
      });
    } catch (err) {
      console.warn('Failed to load conversations:', err);
    }
  }

  async function loadConversation(convId) {
    try {
      currentConversationId = convId;
      localStorage.setItem('aster_active_conv_id', convId);
      document.querySelectorAll('.conv-item').forEach((el) => {
        el.classList.toggle('active', el.dataset.convId === convId);
      });

      const conv = await api(`/api/conversations/${convId}`);
      $('messages').innerHTML = '';
      $('welcome').classList.add('hidden');
      updateStageMode();

      if (window.innerWidth <= 768) {
        toggleSidebar(false);
      }

      if (conv.messages && conv.messages.length) {
        // Rebuild persistent history without temp messages so LLM context stays clean
        activeChatHistory = conv.messages
          .filter((m) => !m.temporary)
          .map((m) => ({ role: m.role, content: m.text }));
        for (const m of conv.messages) {
          const isChem = false;
          addMessage(m.role, m.text, m.latency || '', m.sources || [], false, isChem, m.research_trace, m.temporary);
        }
      } else {
        activeChatHistory = [];
        $('welcome').classList.remove('hidden');
        updateStageMode();
      }

      const stage = $('centerStage');
      if (stage) {
        const hasCode = Boolean(stage.querySelector('.code-editor-card') || activeEngineMode === 'code');
        stage.classList.toggle('has-code-view', hasCode);
      }
    } catch (err) {
      toast(`Could not load conversation: ${err.message}`, true);
    }
  }

  let isSending = false;

  async function sendMessage(event) {
    if (event) event.preventDefault();
    if (isSending) return;
    stopVoiceRecognition();

    // Automatically close sidebar smoothly when user sends query
    const sidebar = $('sidebar');
    if (sidebar && !sidebar.classList.contains('closed')) {
      toggleSidebar();
    }

    const box = $('message');
    const message = box ? box.value.trim() : '';
    if (!message) return;

    isSending = true;
    const sendBtn = $('sendButton');
    const homeSendBtn = $('composerHomeSendBtn');
    if (sendBtn) sendBtn.disabled = true;
    if (homeSendBtn) homeSendBtn.disabled = true;

    if (box) {
      box.value = '';
      box.style.height = 'auto';
    }

    const welcome = $('welcome');
    if (welcome) {
      welcome.classList.add('hidden');
      welcome.style.display = 'none';
    }
    const stage = $('centerStage') || document.querySelector('.astra-center-stage');
    if (stage) {
      stage.classList.remove('welcome-mode');
      stage.classList.add('chat-active');
    }
    const msgStream = $('messages');
    if (msgStream) {
      msgStream.style.display = 'flex';
    }

    addMessage('user', message, '', [], false, false, null, isIncognito);

    const pending = addMessage('assistant', 'Astra is thinking…', '', [], false, false, null, isIncognito);
    const bubble = pending.querySelector('.assistant-content') || pending;
    bubble.classList.add('thinking');

    const clientStartTime = Date.now();

    try {
      // In temporary mode, use only the in-session temp history as context
      // and send the savedNormalConvId so the backend associates with the
      // existing persistent conversation (not creating a brand-new one).
      const historyToSend = isIncognito
        ? activeTempHistory.slice(-8)
        : activeChatHistory.slice(-8);

      const convIdToSend = isIncognito
        ? (savedNormalConvId || currentConversationId || null)
        : (currentConversationId || null);

      const data = await api('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          selected_doc_id: ($('documentSelect') && $('documentSelect').value) || null,
          conversation_id: convIdToSend,
          history: historyToSend,
          incognito: isIncognito,
          detailed: isDetailedMode,
          mode: activeEngineMode,
        }),
      });

      pending.remove();

      if (isIncognito) {
        // ── Temporary mode: NEVER update persistent state ────────────────────
        // Only grow the in-session temp history for multi-turn context.
        activeTempHistory.push({ role: 'user', content: message });
        activeTempHistory.push({ role: 'assistant', content: data.answer });
        // Do NOT touch currentConversationId, activeChatHistory, or localStorage.
        // Do NOT refresh the sidebar conversation list.
      } else {
        // ── Normal mode: update persistent state as usual ────────────────────
        if (data.conversation_id && data.conversation_id !== currentConversationId) {
          currentConversationId = data.conversation_id;
          localStorage.setItem('aster_active_conv_id', currentConversationId);
        }
        activeChatHistory.push({ role: 'user', content: message });
        activeChatHistory.push({ role: 'assistant', content: data.answer });
        refreshConversations();
      }

      // Ensure latency calculation is strictly positive
      const elapsedMs = data.latency_ms && data.latency_ms > 0 ? data.latency_ms : (Date.now() - clientStartTime);
      const latencyStr = formatDuration(elapsedMs);

      // Render answer + small ChatGPT-style source chips with clean response time
      const isChem = false;
      addMessage(
        'assistant',
        data.answer,
        latencyStr,
        data.sources || [],
        true,
        isChem,
        data.research_trace,
        isIncognito
      );

    } catch (err) {
      console.error('sendMessage error:', err);
      if (pending) pending.remove();
      addMessage('assistant', `Error: ${err.message}`, '', [], true, false, null, isIncognito);
      toast(err.message, true);
    } finally {
      isSending = false;
      if (sendBtn) sendBtn.disabled = false;
      if (homeSendBtn) homeSendBtn.disabled = false;
      if (box) box.focus();
    }
  }

  async function uploadFiles(files) {
    if (!files || !files.length) return;
    const MAX_UPLOAD_BYTES = 400 * 1024 * 1024; // 400 MB
    let lastUploadedDoc = null;
    for (const file of files) {
      if (file.size > MAX_UPLOAD_BYTES) {
        toast('File size exceeds the 400 MB limit. Please upload a file smaller than or equal to 400 MB.', true);
        continue;
      }
      const form = new FormData();
      form.append('file', file);
      try {
        toast(`Uploading & indexing ${file.name}…`);
        const res = await api('/api/upload', { method: 'POST', body: form });
        const count = res.document?.chunks_count || 'multiple';
        toast(`✓ ${file.name} indexed successfully (${count} chunks)`);
        if (res.document) {
          lastUploadedDoc = res.document;
        }
      } catch (err) {
        toast(`${file.name}: ${err.message}`, true);
      }
    }
    if ($('fileInput')) $('fileInput').value = '';
    await refreshDocuments();
    if (lastUploadedDoc && lastUploadedDoc.id) {
      if ($('documentSelect')) $('documentSelect').value = lastUploadedDoc.id;
      updateComposerDocTag(lastUploadedDoc.id, lastUploadedDoc.name);
      document.querySelectorAll('.document-item').forEach((el) => {
        el.classList.toggle('active', el.dataset.docId === String(lastUploadedDoc.id));
      });
      toast(`Document focus set to: ${lastUploadedDoc.name}`);
    }
  }

  function clearAllAuthFields() {
    if ($('loginForm')) $('loginForm').reset();
    if ($('signupForm')) $('signupForm').reset();
    if ($('forgotPasswordForm')) $('forgotPasswordForm').reset();
    if ($('resetPasswordForm')) $('resetPasswordForm').reset();

    const fieldIds = [
      'loginIdentifier',
      'loginPassword',
      'loginMathAnswer',
      'signupUsername',
      'signupEmail',
      'signupPhone',
      'signupPassword',
      'signupMathAnswer',
      'forgotIdentifier',
      'forgotMathAnswer',
      'newPasswordInput',
      'confirmPasswordInput',
      'resetTokenHidden',
    ];
    fieldIds.forEach((id) => {
      const el = $(id);
      if (el) {
        el.value = '';
        el.setAttribute('value', '');
      }
    });

    if ($('resetErrorMsg')) {
      $('resetErrorMsg').textContent = '';
      $('resetErrorMsg').classList.add('hidden');
    }
  }

  function ensureLoginFormEmpty() {
    clearAllAuthFields();
    setTimeout(() => {
      if (!token) {
        if ($('loginIdentifier')) $('loginIdentifier').value = '';
        if ($('loginPassword')) $('loginPassword').value = '';
        if ($('loginMathAnswer')) $('loginMathAnswer').value = '';
      }
    }, 40);
    setTimeout(() => {
      if (!token) {
        if ($('loginIdentifier')) $('loginIdentifier').value = '';
        if ($('loginPassword')) $('loginPassword').value = '';
        if ($('loginMathAnswer')) $('loginMathAnswer').value = '';
      }
    }, 150);
  }

  function switchAuthMode(mode) {
    authMode = mode;
    ensureLoginFormEmpty();
    document.querySelectorAll('.modal-tab').forEach((t) => t.classList.toggle('active', t.dataset.mode === mode));
    $('authHeading').textContent = mode === 'login' ? 'Quick Login' : 'Create Account';
    if ($('loginForm')) $('loginForm').classList.toggle('hidden', mode !== 'login');
    if ($('signupForm')) $('signupForm').classList.toggle('hidden', mode !== 'signup');
    loadMathChallenge(mode);
  }

  // ==========================================
  // 2-Digit Math Calculation Challenge System
  // ==========================================
  let currentLoginChallenge = null;
  let currentSignupChallenge = null;
  let currentForgotChallenge = null;

  async function loadMathChallenge(mode = 'login') {
    try {
      const challenge = await api('/api/auth/challenge');
      if (mode === 'login' || mode === 'all') {
        currentLoginChallenge = challenge;
        if ($('loginMathQuestion')) {
          $('loginMathQuestion').innerHTML = `<span style="color:#f472b6;">${challenge.num1}</span> <span style="color:#f97316;">${challenge.operation}</span> <span style="color:#f472b6;">${challenge.num2}</span> = ?`;
        }
        if ($('loginMathAnswer')) {
          $('loginMathAnswer').value = ''; // NEVER pre-fill! Manual user calculation required.
        }
      }
      if (mode === 'signup' || mode === 'all') {
        currentSignupChallenge = challenge;
        if ($('signupMathQuestion')) {
          $('signupMathQuestion').innerHTML = `<span style="color:#f472b6;">${challenge.num1}</span> <span style="color:#f97316;">${challenge.operation}</span> <span style="color:#f472b6;">${challenge.num2}</span> = ?`;
        }
        if ($('signupMathAnswer')) $('signupMathAnswer').value = '';
      }
    } catch (err) {
      toast('Failed to load calculation challenge: ' + err.message, true);
    }
  }

  async function loadForgotMathChallenge() {
    try {
      const challenge = await api('/api/auth/challenge');
      currentForgotChallenge = challenge;
      if ($('forgotMathQuestion')) {
        $('forgotMathQuestion').innerHTML = `<span style="color:#f472b6;">${challenge.num1}</span> <span style="color:#f97316;">${challenge.operation}</span> <span style="color:#f472b6;">${challenge.num2}</span> = ?`;
      }
      if ($('forgotMathAnswer')) $('forgotMathAnswer').value = '';
    } catch (err) {
      toast('Failed to load calculation challenge: ' + err.message, true);
    }
  }

  if ($('loginRefreshChallengeBtn')) {
    $('loginRefreshChallengeBtn').onclick = () => loadMathChallenge('login');
  }

  if ($('signupRefreshChallengeBtn')) {
    $('signupRefreshChallengeBtn').onclick = () => loadMathChallenge('signup');
  }

  if ($('forgotRefreshChallengeBtn')) {
    $('forgotRefreshChallengeBtn').onclick = () => loadForgotMathChallenge();
  }

  // Password visibility toggle
  if ($('toggleLoginPasswordBtn')) {
    $('toggleLoginPasswordBtn').onclick = () => {
      const inp = $('loginPassword');
      if (inp) {
        inp.type = inp.type === 'password' ? 'text' : 'password';
      }
    };
  }

  // Forgot Password Modal Triggers
  if ($('forgotPasswordLinkBtn')) {
    $('forgotPasswordLinkBtn').onclick = () => {
      if ($('authModal')) $('authModal').classList.add('hidden');
      if ($('forgotPasswordModal')) {
        $('forgotPasswordModal').classList.remove('hidden');
        if ($('forgotPasswordForm')) $('forgotPasswordForm').classList.remove('hidden');
        if ($('resetPasswordForm')) $('resetPasswordForm').classList.add('hidden');
        loadForgotMathChallenge();
      }
    };
  }

  if ($('closeForgotModal')) {
    $('closeForgotModal').onclick = () => {
      if ($('forgotPasswordModal')) $('forgotPasswordModal').classList.add('hidden');
    };
  }

  if ($('backToLoginFromForgotBtn')) {
    $('backToLoginFromForgotBtn').onclick = () => {
      if ($('forgotPasswordModal')) $('forgotPasswordModal').classList.add('hidden');
      if ($('authModal')) {
        $('authModal').classList.remove('hidden');
        switchAuthMode('login');
      }
    };
  }

  // Handle Forgot Password Step 1 (Request reset token with math check)
  if ($('forgotPasswordForm')) {
    $('forgotPasswordForm').onsubmit = async (e) => {
      e.preventDefault();
      const identifier = $('forgotIdentifier') ? $('forgotIdentifier').value.trim() : '';
      const ansVal = $('forgotMathAnswer') ? $('forgotMathAnswer').value.trim() : '';
      const calculation_result = parseInt(ansVal, 10);

      if (!identifier) {
        toast('Please enter your account identifier', true);
        return;
      }
      if (isNaN(calculation_result)) {
        toast('Please calculate and enter the answer to the math problem', true);
        return;
      }
      if (!currentForgotChallenge) {
        toast('Challenge expired, refreshing…', true);
        await loadForgotMathChallenge();
        return;
      }

      try {
        toast('Verifying identity & math calculation…');
        const res = await api('/api/auth/forgot-password', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            identifier,
            challenge_token: currentForgotChallenge.challenge_token,
            calculation_result,
          }),
        });

        if (res.reset_token) {
          $('resetTokenHidden').value = res.reset_token;
          $('forgotPasswordForm').classList.add('hidden');
          $('resetPasswordForm').classList.remove('hidden');
          if ($('resetErrorMsg')) $('resetErrorMsg').classList.add('hidden');
          toast('Identity verified! Please enter your new password.');
        } else {
          toast(res.message || 'Verification complete.');
        }
      } catch (err) {
        toast(err.message, true);
        loadForgotMathChallenge();
      }
    };
  }

  // Handle Forgot Password Step 2 (Reset Password)
  if ($('resetPasswordForm')) {
    $('resetPasswordForm').onsubmit = async (e) => {
      e.preventDefault();
      const reset_token = $('resetTokenHidden').value;
      const new_password = $('newPasswordInput').value;
      const confirm_password = $('confirmPasswordInput').value;

      if (!new_password) {
        toast('Please enter a new password', true);
        return;
      }
      if (new_password !== confirm_password) {
        toast('Passwords do not match', true);
        return;
      }

      try {
        toast('Updating password…');
        const res = await api('/api/auth/reset-password', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            reset_token,
            new_password,
            new_password_confirm: confirm_password,
          }),
        });

        toast('✓ ' + (res.message || 'Password reset successfully!'));
        if ($('forgotPasswordModal')) $('forgotPasswordModal').classList.add('hidden');
        if ($('authModal')) {
          $('authModal').classList.remove('hidden');
          switchAuthMode('login');
        }
      } catch (err) {
        if ($('resetErrorMsg')) {
          $('resetErrorMsg').textContent = err.message;
          $('resetErrorMsg').classList.remove('hidden');
        }
        toast(err.message, true);
      }
    };
  }

  // Handle Math Calculation Login (Requires BOTH math challenge and password)
  if ($('loginForm')) {
    $('loginForm').onsubmit = async (e) => {
      e.preventDefault();
      const identifier = $('loginIdentifier') ? $('loginIdentifier').value.trim() : '';
      const password = $('loginPassword') ? $('loginPassword').value : '';
      const ansVal = $('loginMathAnswer') ? $('loginMathAnswer').value.trim() : '';
      const calculation_result = parseInt(ansVal, 10);

      if (!identifier) {
        toast('Please enter your phone number, email, or username', true);
        return;
      }
      if (!password) {
        toast('Please enter your password', true);
        return;
      }
      if (isNaN(calculation_result)) {
        toast('Please calculate and enter the answer to the math problem', true);
        return;
      }
      if (!currentLoginChallenge) {
        toast('Calculation expired, refreshing…', true);
        await loadMathChallenge('login');
        return;
      }

      try {
        toast('Verifying calculation & logging in…');
        const payload = {
          identifier,
          password,
          challenge_token: currentLoginChallenge.challenge_token,
          calculation_result,
        };

        const result = await api('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        token = result.access_token;
        user = result.user;
        cachedUsername = (user && user.username) || identifier;
        guestToken = '';
        localStorage.removeItem('astra_guest_token');
        localStorage.setItem('aster_token', token);
        localStorage.setItem('aster_username', cachedUsername);
        if ($('authModal')) $('authModal').classList.add('hidden');
        toast(`Welcome back, ${cachedUsername}!`);
        updateAuthUI();
        refreshDocuments();
        refreshConversations();
      } catch (err) {
        toast(err.message, true);
        loadMathChallenge('login');
      }
    };
  }


  // Handle Math Calculation Registration
  if ($('signupForm')) {
    $('signupForm').onsubmit = async (e) => {
      e.preventDefault();
      const username = $('signupUsername').value.trim();
      const email = $('signupEmail').value.trim();
      const phone = $('signupPhone') ? $('signupPhone').value.trim() : '';
      const password = $('signupPassword').value;
      const calculation_result = parseInt($('signupMathAnswer').value.trim(), 10);

      if (!username || !email || !password) {
        toast('Please fill in username, email, and password', true);
        return;
      }
      if (isNaN(calculation_result)) {
        toast('Please calculate and enter the answer to the math problem', true);
        return;
      }
      if (!currentSignupChallenge) {
        toast('Calculation expired, refreshing…', true);
        await loadMathChallenge('signup');
        return;
      }

      try {
        toast('Verifying calculation & creating account…');
        const result = await api('/api/auth/register', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            username,
            email,
            phone,
            password,
            challenge_token: currentSignupChallenge.challenge_token,
            calculation_result,
          }),
        });
        token = result.access_token;
        user = result.user;
        cachedUsername = (user && user.username) || username;
        guestToken = '';
        localStorage.removeItem('astra_guest_token');
        localStorage.setItem('aster_token', token);
        localStorage.setItem('aster_username', cachedUsername);
        $('authModal').classList.add('hidden');
        toast(`Account created! Welcome, ${cachedUsername}!`);
        updateAuthUI();
        refreshDocuments();
        refreshConversations();
      } catch (err) {
        toast(err.message, true);
        loadMathChallenge('signup');
      }
    };
  }

  // Event Listeners
  const submitHandler = (e) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    sendMessage(e);
    return false;
  };

  if ($('chatForm')) {
    $('chatForm').onsubmit = submitHandler;
  }
  if ($('sendButton')) {
    $('sendButton').onclick = submitHandler;
  }
  if ($('composerHomeSendBtn')) {
    $('composerHomeSendBtn').onclick = submitHandler;
  }
  if ($('message')) {
    $('message').addEventListener('keydown', (e) => {
      if ((e.key === 'Enter' || e.keyCode === 13) && !e.shiftKey) {
        e.preventDefault();
        e.stopPropagation();
        sendMessage(e);
      }
    });
    $('message').addEventListener('input', (e) => {
      e.target.style.height = 'auto';
      e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
    });
  }

  // ==========================================
  // Voice-to-Text & Attachment Actions Menu
  // ==========================================
  let speechRecognition = null;
  let isListening = false;
  let shouldRestartVoice = false;
  let voiceInitialText = '';
  let voiceSafetyTimer = null;

  function setVoiceActive(active) {
    const pill = $('voiceSearchPill');
    const wave = $('voiceWaveform');
    const homePill = $('composerHomeMicPill');
    const homeWave = $('composerHomeVoiceWaveform');
    const micBtn = $('composerMicBtn');
    const homeMicBtn = $('composerHomeMicBtn');
    const msgInput = $('message');

    if (pill) pill.classList.toggle('active', active);
    if (wave) wave.classList.toggle('hidden', !active);
    if (homePill) homePill.classList.toggle('active', active);
    if (homeWave) homeWave.classList.toggle('hidden', !active);
    if (micBtn) micBtn.classList.toggle('recording', active);
    if (homeMicBtn) homeMicBtn.classList.toggle('recording', active);

    if (msgInput) {
      if (active) {
        msgInput.setAttribute('data-prev-placeholder', msgInput.placeholder || 'Type here');
        msgInput.placeholder = 'Listening… Speak now';
      } else {
        const prev = msgInput.getAttribute('data-prev-placeholder');
        if (prev) msgInput.placeholder = prev;
      }
    }
  }

  function stopVoiceRecognition() {
    isListening = false;
    shouldRestartVoice = false;
    if (voiceSafetyTimer) {
      clearTimeout(voiceSafetyTimer);
      voiceSafetyTimer = null;
    }
    setVoiceActive(false);

    if (speechRecognition) {
      try {
        speechRecognition.abort();
      } catch (_) {
        try { speechRecognition.stop(); } catch (__) {}
      }
      speechRecognition = null;
    }

    if ($('voiceListeningBadge')) $('voiceListeningBadge').classList.add('hidden');
    if ($('composerAttachBtn')) $('composerAttachBtn').classList.remove('active');
  }

  function startVoiceRecognition() {
    if (isListening) {
      stopVoiceRecognition();
      return;
    }

    isListening = true;
    shouldRestartVoice = true;
    setVoiceActive(true);

    // Auto-stop safety timeout after 60s of inactivity so it doesn't run forever
    if (voiceSafetyTimer) clearTimeout(voiceSafetyTimer);
    voiceSafetyTimer = setTimeout(() => {
      if (isListening) stopVoiceRecognition();
    }, 60000);

    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      return;
    }

    try {
      if (speechRecognition) {
        try { speechRecognition.abort(); } catch (_) {}
      }

      speechRecognition = new SpeechRec();
      speechRecognition.continuous = true;
      speechRecognition.interimResults = true;
      speechRecognition.lang = navigator.language || 'en-US';

      voiceInitialText = $('message') ? $('message').value : '';

      speechRecognition.onstart = () => {
        isListening = true;
        setVoiceActive(true);
      };

      speechRecognition.onresult = (event) => {
        let sessionTranscript = '';
        for (let i = 0; i < event.results.length; i++) {
          sessionTranscript += event.results[i][0].transcript;
        }
        if ($('message')) {
          const separator = voiceInitialText && !voiceInitialText.endsWith(' ') ? ' ' : '';
          $('message').value = voiceInitialText ? `${voiceInitialText}${separator}${sessionTranscript}` : sessionTranscript;
          $('message').dispatchEvent(new Event('input', { bubbles: true }));
        }
      };

      speechRecognition.onerror = (event) => {
        console.warn('Speech recognition status:', event.error);
        if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
          toast('Microphone permission required. Please allow microphone in browser.', true);
        }
      };

      speechRecognition.onend = () => {
        // Keep the animation alive! If still in listening mode, restart gracefully
        if (isListening && shouldRestartVoice) {
          try {
            speechRecognition.start();
          } catch (_) {
            setTimeout(() => {
              if (isListening && shouldRestartVoice) {
                try { speechRecognition.start(); } catch (__) {}
              }
            }, 250);
          }
        }
      };

      speechRecognition.start();
    } catch (err) {
      console.warn('Speech recognition note:', err);
    }
  }

  function handleMicClick(e) {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
      if (e.currentTarget && e.currentTarget.blur) e.currentTarget.blur();
    }
    if (isListening) {
      stopVoiceRecognition();
    } else {
      startVoiceRecognition();
    }
  }

  if ($('composerMicBtn')) {
    $('composerMicBtn').onclick = handleMicClick;
  }

  if ($('composerHomeMicBtn')) {
    $('composerHomeMicBtn').onclick = handleMicClick;
  }

  if ($('voiceSearchPill')) {
    $('voiceSearchPill').onclick = (e) => {
      // Clicking the red capsule directly stops voice
      if (isListening && e.target === $('voiceSearchPill')) {
        handleMicClick(e);
      }
    };
  }

  if ($('composerHomeMicPill')) {
    $('composerHomeMicPill').onclick = (e) => {
      if (isListening && e.target === $('composerHomeMicPill')) {
        handleMicClick(e);
      }
    };
  }

  if ($('cancelRecordingBtn')) {
    $('cancelRecordingBtn').onclick = (e) => {
      if (e) e.preventDefault();
      stopVoiceRecognition();
    };
  }

  // Toggle attachment action dropdown
  if ($('composerAttachBtn') && $('attachDropdown')) {
    $('composerAttachBtn').onclick = (e) => {
      e.stopPropagation();
      const isHidden = $('attachDropdown').classList.toggle('hidden');
      $('composerAttachBtn').classList.toggle('active', !isHidden);
    };

    if ($('composerHomeAttachBtn')) {
      $('composerHomeAttachBtn').onclick = (e) => {
        e.stopPropagation();
        const wrapper = $('composerHomeAttachBtn').closest('.composer-attach-wrapper');
        if (wrapper && $('attachDropdown')) {
          wrapper.appendChild($('attachDropdown'));
        }
        const isHidden = $('attachDropdown').classList.toggle('hidden');
        $('composerHomeAttachBtn').classList.toggle('active', !isHidden);
      };
    }

    // Close dropdown when clicking anywhere outside or pressing Escape
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.composer-attach-wrapper')) {
        $('attachDropdown').classList.add('hidden');
        $('composerAttachBtn').classList.remove('active');
      }
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && $('attachDropdown') && !$('attachDropdown').classList.contains('hidden')) {
        $('attachDropdown').classList.add('hidden');
        $('composerAttachBtn').classList.remove('active');
      }
    });

    // Dropdown Option 1: Upload Document
    if ($('actionUploadDoc')) {
      $('actionUploadDoc').onclick = () => {
        $('attachDropdown').classList.add('hidden');
        $('composerAttachBtn').classList.remove('active');
        if ($('fileInput')) $('fileInput').click();
      };
    }

    // Dropdown Option 2: Voice to Text Search
    if ($('actionVoiceSearch')) {
      $('actionVoiceSearch').onclick = () => {
        $('attachDropdown').classList.add('hidden');
        $('composerAttachBtn').classList.remove('active');
        startVoiceRecognition();
      };
    }
  }

  if ($('cancelVoiceBtn')) {
    $('cancelVoiceBtn').onclick = stopVoiceRecognition;
  }

  // Suggestion buttons click handler
  document.querySelectorAll('.suggestions button').forEach((btn) => {
    btn.onclick = () => {
      const prompt = btn.dataset.prompt || btn.textContent.trim();
      $('message').value = prompt;
      sendMessage();
    };
  });

  if ($('fileInput')) $('fileInput').onchange = (e) => uploadFiles(e.target.files);
  if ($('uploadDrop')) {
    $('uploadDrop').ondragover = (e) => e.preventDefault();
    $('uploadDrop').ondrop = (e) => {
      e.preventDefault();
      uploadFiles(e.dataTransfer.files);
    };
  }
  window.ondragover = (e) => e.preventDefault();
  window.ondrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
      uploadFiles(e.dataTransfer.files);
    }
  };

  if ($('newChat')) {
    $('newChat').onclick = async () => {
    $('messages').innerHTML = '';
    $('welcome').classList.remove('hidden');
    const stage = $('centerStage');
    if (stage) stage.classList.remove('has-code-view');
    shuffleWelcomeTagline();
    updateStageMode();
    activeChatHistory = [];
    if ($('documentSelect')) $('documentSelect').value = '';
    document.querySelectorAll('.document-item').forEach((el) => el.classList.remove('active'));
    document.querySelectorAll('.conv-item').forEach((el) => el.classList.remove('active'));
    if ($('activeDocTag')) $('activeDocTag').classList.add('hidden');
    if ($('activeDocName')) $('activeDocName').textContent = '';
    if ($('message')) {
      $('message').value = '';
      $('message').style.height = 'auto';
    }

    try {
      // Create new clean conversation record in DB
      const newConv = await api('/api/conversations', { method: 'POST' });
      currentConversationId = newConv.id;
      localStorage.setItem('aster_active_conv_id', currentConversationId);
      refreshConversations();
    } catch (_) {
      currentConversationId = '';
      localStorage.removeItem('aster_active_conv_id');
    }
    toast('Started a new conversation session');
    };
  }

  // Wire Private Mode Toggle Buttons (Top right navbar button and Sidebar header icon)


  // Settings & Appearance Modal Controls
  function openSettingsModal() {
    if ($('settingsModal')) $('settingsModal').classList.remove('hidden');
  }

  function closeSettingsModal() {
    if ($('settingsModal')) $('settingsModal').classList.add('hidden');
  }

  if ($('closeSettingsModal')) {
    $('closeSettingsModal').onclick = () => closeSettingsModal();
  }

  if ($('settingsModal')) {
    $('settingsModal').onclick = (e) => {
      if (e.target === $('settingsModal')) closeSettingsModal();
    };
  }

  // 1. Glass Frost Slider (Blur level)
  const frostSlider = $('glassFrostSlider');
  const frostDisplay = $('frostValueDisplay');
  if (frostSlider) {
    frostSlider.oninput = (e) => {
      const val = e.target.value;
      document.documentElement.style.setProperty('--glass-blur', `${val}px`);
      if (frostDisplay) frostDisplay.textContent = `${val}px`;
      localStorage.setItem('astra_frost_blur', val);
    };
  }

  // Helper to detect perceived background image brightness & automatically adjust text contrast
  function detectBackgroundBrightness(dataUrl, callback) {
    if (!dataUrl) {
      if (callback) callback('light');
      return;
    }
    const img = new Image();
    img.crossOrigin = 'Anonymous';
    img.onload = () => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = 32;
        canvas.height = 32;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, 32, 32);
        const imgData = ctx.getImageData(0, 0, 32, 32).data;
        let totalLuminance = 0;
        let count = 0;
        for (let i = 0; i < imgData.length; i += 4) {
          const r = imgData[i];
          const g = imgData[i + 1];
          const b = imgData[i + 2];
          // ITU-R BT.709 perceived luminance formula
          const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
          totalLuminance += lum;
          count++;
        }
        const avgLuminance = count ? (totalLuminance / count) : 255;
        // If average luminance < 130, consider dark background
        const theme = avgLuminance < 130 ? 'dark' : 'light';
        if (callback) callback(theme);
      } catch (err) {
        console.warn('Luminance calculation failed, defaulting to light theme:', err);
        if (callback) callback('light');
      }
    };
    img.onerror = () => {
      if (callback) callback('light');
    };
    img.src = dataUrl;
  }

  // 2. Custom Background Wallpaper Upload
  const customBgInput = $('customBgFileInput');
  const uploadBgBtn = $('uploadBgBtn');
  const resetBgBtn = $('resetBgBtn');
  const customBgStatus = $('customBgStatus');
  const bgWallpaper = document.querySelector('.astra-background');

  if (uploadBgBtn && customBgInput) {
    uploadBgBtn.onclick = () => customBgInput.click();
  }

  if (customBgInput) {
    customBgInput.onchange = (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;

      if (file.size > 8 * 1024 * 1024) {
        toast('⚠️ Wallpaper image must be under 8MB', true);
        return;
      }

      const reader = new FileReader();
      reader.onload = (loadEvent) => {
        const dataUrl = loadEvent.target.result;
        if (bgWallpaper) {
          bgWallpaper.style.backgroundImage = `url("${dataUrl}")`;
        }
        try {
          localStorage.setItem('astra_custom_bg', dataUrl);
        } catch (_) {
          console.warn('LocalStorage quota exceeded for wallpaper DataURL');
        }
        detectBackgroundBrightness(dataUrl, (theme) => {
          if (theme === 'dark') {
            document.documentElement.setAttribute('data-bg-theme', 'dark');
          } else {
            document.documentElement.removeAttribute('data-bg-theme');
          }
        });
        if (customBgStatus) {
          customBgStatus.textContent = `Custom wallpaper: ${file.name}`;
        }
        toast('✨ Custom background applied successfully!');
      };
      reader.onerror = () => {
        toast('❌ Failed to read background image file', true);
      };
      reader.readAsDataURL(file);
    };
  }

  if (resetBgBtn) {
    resetBgBtn.onclick = () => {
      if (bgWallpaper) {
        bgWallpaper.style.backgroundImage = '';
      }
      document.documentElement.removeAttribute('data-bg-theme');
      localStorage.removeItem('astra_custom_bg');
      if (customBgInput) customBgInput.value = '';
      if (customBgStatus) {
        customBgStatus.textContent = 'Default fluid wallpaper active';
      }
      toast('🔄 Restored default fluid wallpaper');
    };
  }

  // Restore saved appearance preferences on startup
  const savedFrost = localStorage.getItem('astra_frost_blur');
  if (savedFrost) {
    document.documentElement.style.setProperty('--glass-blur', `${savedFrost}px`);
    if (frostSlider) frostSlider.value = savedFrost;
    if (frostDisplay) frostDisplay.textContent = `${savedFrost}px`;
  }

  const savedCustomBg = localStorage.getItem('astra_custom_bg');
  if (savedCustomBg && bgWallpaper) {
    bgWallpaper.style.backgroundImage = `url("${savedCustomBg}")`;
    detectBackgroundBrightness(savedCustomBg, (theme) => {
      if (theme === 'dark') {
        document.documentElement.setAttribute('data-bg-theme', 'dark');
      } else {
        document.documentElement.removeAttribute('data-bg-theme');
      }
    });
    if (customBgStatus) customBgStatus.textContent = 'Custom wallpaper active';
  }


  // User Auth Modal Controls & Gear Icon Routing
  if ($('userAuthBtn')) {
    $('userAuthBtn').onclick = (e) => {
      // If user specifically clicked on the gear icon, open Settings
      if (e.target.closest('.user-profile-right')) {
        e.stopPropagation();
        openSettingsModal();
        return;
      }
      if (!token) {
        ensureLoginFormEmpty();
        authMode = 'login';
        switchAuthMode('login');
      }
      $('authModal').classList.remove('hidden');
      loadMathChallenge(authMode);
    };
  }
  if ($('closeAuthModal')) {
    $('closeAuthModal').onclick = () => {
      $('authModal').classList.add('hidden');
      if (!token) {
        ensureLoginFormEmpty();
      }
    };
  }
  document.querySelectorAll('.modal-tab').forEach((t) => (t.onclick = () => switchAuthMode(t.dataset.mode)));

  if ($('logoutBtn')) {
    $('logoutBtn').onclick = async () => {
      // 1. Notify server to invalidate session cookie
      try {
        await api('/api/auth/logout', { method: 'POST' });
      } catch (_) {}

      // 2. Clear client credentials and local storage completely
      token = '';
      user = null;
      cachedUsername = '';
      currentConversationId = '';
      activeChatHistory = [];
      localStorage.clear();

      // Reset device ID and guest token to ensure complete isolation for the new session
      deviceId = 'dev_' + (window.crypto && window.crypto.randomUUID ? window.crypto.randomUUID() : (Math.random().toString(36).substring(2) + Date.now().toString(36)));
      localStorage.setItem('astra_device_id', deviceId);
      guestToken = '';

      // 3. Completely blank out every single login and registration form field
      ensureLoginFormEmpty();

      // 4. Hide auth modal
      if ($('authModal')) $('authModal').classList.add('hidden');

      // 5. Reset chat stream and show welcome card
      if ($('messages')) $('messages').innerHTML = '';
      if ($('welcome')) $('welcome').classList.remove('hidden');
      if ($('message')) {
        $('message').value = '';
        $('message').style.height = 'auto';
      }

      // 6. Reset composer document focus tag
      if ($('activeDocTag')) $('activeDocTag').classList.add('hidden');
      if ($('activeDocName')) $('activeDocName').textContent = '';

      // 7. Completely wipe previous user's documents & conversations from the view
      if ($('documentSelect')) {
        $('documentSelect').innerHTML = '<option value="">All indexed documents</option>';
      }
      if ($('documentList')) {
        $('documentList').innerHTML = '<p class="muted-box-msg">No documents uploaded yet</p>';
      }
      if ($('docCount')) $('docCount').textContent = '0';
      if ($('conversationList')) {
        $('conversationList').innerHTML = '<p class="muted-box-msg">No conversations yet</p>';
      }

      // 8. Reset auth UI badges and state to guest
      authMode = 'login';
      updateAuthUI();
      loadMathChallenge('login');
      toast('Signed out. Every field and session has been cleared.');
    };
  }

  // Initialize workspace immediately for everyone
  initSidebar();
  refreshDocuments();
  refreshConversations();
  checkHealth();
  if (!token) {
    ensureLoginFormEmpty();
  }
  shuffleWelcomeTagline();
  updateStageMode();

  // Popover Menu Controls & Temporary Chat Switch
  const enginePopover = $('modelEnginePopover');
  const engineBtn = $('engineToggleBtn');
  const tempToggle = $('tempChatToggle');

  if (engineBtn && enginePopover) {
    engineBtn.onclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      const isHidden = enginePopover.classList.toggle('hidden');
      engineBtn.setAttribute('aria-expanded', !isHidden);
    };

    // Close popover when clicking anywhere outside
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.engine-toggle-wrapper')) {
        if (!enginePopover.classList.contains('hidden')) {
          enginePopover.classList.add('hidden');
          engineBtn.setAttribute('aria-expanded', 'false');
        }
      }
    });

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !enginePopover.classList.contains('hidden')) {
        enginePopover.classList.add('hidden');
        engineBtn.setAttribute('aria-expanded', 'false');
      }
    });
  }

  // Row 1: Temporary Chat Toggle Switch
  if (tempToggle) {
    tempToggle.checked = isIncognito;
    tempToggle.onchange = (e) => {
      setIncognitoMode(e.target.checked);
    };
  }

  // Sidebar Header Disappearing Chat Button
  if ($('sidebarDisappearingBtn')) {
    $('sidebarDisappearingBtn').onclick = (e) => {
      e.preventDefault();
      toggleIncognitoMode();
    };
  }

  // Row 2: Astracore 3.1
  if ($('popoverModelAstracore')) {
    $('popoverModelAstracore').onclick = () => {
      setEngineMode('general');
    };
  }

  // Row 3: Astracore O (Coming Soon)
  if ($('popoverModelAstracoreO')) {
    $('popoverModelAstracoreO').onclick = () => {
      toast('🚀 Astracore O is coming soon! Stay tuned.');
    };
  }



  // Row: Quick Mode
  if ($('popoverModeQuick')) {
    $('popoverModeQuick').onclick = () => {
      activeEngineMode = 'general';
      setDetailedMode(false);
      updateEngineModeUI();
    };
  }

  // Row: Extended Mode
  if ($('popoverModeExtended')) {
    $('popoverModeExtended').onclick = () => {
      activeEngineMode = 'general';
      setDetailedMode(true);
      updateEngineModeUI();
    };
  }

  // Row: Code Mode
  if ($('popoverModeCode')) {
    $('popoverModeCode').onclick = () => {
      setEngineMode('code');
    };
  }

  // Sync initial detailed mode checkmark UI
  setDetailedMode(isDetailedMode);
  updateEngineLabelUI();

  if (token) {
    api('/api/auth/me')
      .then((data) => {
        user = data;
        if (user && user.username) {
          cachedUsername = user.username;
          localStorage.setItem('aster_username', cachedUsername);
        }
        updateAuthUI();
        refreshConversations();
      })
      .catch(() => {
        token = '';
        cachedUsername = '';
        localStorage.removeItem('aster_token');
        localStorage.removeItem('aster_username');
        updateAuthUI();
      });
  } else {
    updateAuthUI();
  }

  // Support automated test execution via URL parameter
  const autoQuery = new URLSearchParams(window.location.search).get('auto_query');
  if (autoQuery && $('message') && $('sendButton')) {
    setTimeout(() => {
      $('message').value = autoQuery;
      $('sendButton').click();
    }, 500);
  }

  // Security: Disable Right-Click Context Menu & Inspection Shortcuts
  window.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    return false;
  }, true);

  window.addEventListener('keydown', (e) => {
    // F12 key
    if (e.key === 'F12' || e.keyCode === 123) {
      e.preventDefault();
      e.stopPropagation();
      toast('🔒 Developer tools inspection is disabled.', true);
      return false;
    }

    const isCtrl = e.ctrlKey || e.metaKey;

    // Ctrl+Shift+I, Ctrl+Shift+J, Ctrl+Shift+C
    if (isCtrl && e.shiftKey && ['I', 'J', 'C', 'i', 'j', 'c'].includes(e.key)) {
      e.preventDefault();
      e.stopPropagation();
      toast('🔒 Inspect Element is disabled.', true);
      return false;
    }

    // Ctrl+U (View Page Source), Ctrl+S (Save Page)
    if (isCtrl && ['u', 'U', 's', 'S'].includes(e.key)) {
      e.preventDefault();
      e.stopPropagation();
      toast('🔒 Viewing page source is disabled.', true);
      return false;
    }
  }, true);

  // ── Mobile Virtual Keyboard Auto-Scroll & Viewport Handling ────────────────
  if (window.visualViewport) {
    const handleViewportChange = () => {
      const stage = $('centerStage');
      if (stage && !stage.classList.contains('welcome-mode')) {
        const stream = $('messages');
        if (stream) {
          stream.scrollTop = stream.scrollHeight;
        }
      }
    };
    window.visualViewport.addEventListener('resize', handleViewportChange);
    window.visualViewport.addEventListener('scroll', handleViewportChange);
  }

  const msgInput = $('message');
  if (msgInput) {
    msgInput.addEventListener('focus', () => {
      setTimeout(() => {
        const stream = $('messages');
        if (stream) stream.scrollTop = stream.scrollHeight;
        msgInput.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 300);
    });
  }
})();
