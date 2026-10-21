# generate_html_report.py
import os
import json

def build_html_report():
    md_path = "BAO_CAO_TOAN_DIEN_DU_AN_BANKING77.md"
    html_out_path = "BAO_CAO_TOAN_DIEN_DU_AN_BANKING77.html"
    
    if not os.path.exists(md_path):
        print(f"File not found: {md_path}")
        return
        
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
        
    # Serialize markdown text safely for embedding in JS
    md_json = json.dumps(md_content)
    
    html_template = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo Toàn Diện Đồ Án Data Mining — BANKING77</title>
    <!-- Marked.js for Markdown parsing -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- KaTeX for math rendering -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <!-- Mermaid for diagrams -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    
    <style>
        :root {{
            --bg-primary: #0F172A;
            --bg-secondary: #1E293B;
            --bg-card: #1E293B;
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
            --accent: #38BDF8;
            --accent-glow: rgba(56, 189, 248, 0.2);
            --border-color: #334155;
            --code-bg: #0B1120;
            --sidebar-width: 320px;
        }}
        
        [data-theme="light"] {{
            --bg-primary: #F8FAFC;
            --bg-secondary: #FFFFFF;
            --bg-card: #FFFFFF;
            --text-primary: #0F172A;
            --text-secondary: #475569;
            --accent: #0284C7;
            --accent-glow: rgba(2, 132, 199, 0.15);
            --border-color: #E2E8F0;
            --code-bg: #F1F5F9;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            display: flex;
            min-height: 100vh;
            line-height: 1.7;
            transition: background-color 0.2s, color 0.2s;
        }}
        
        /* Sidebar styling */
        #sidebar {{
            width: var(--sidebar-width);
            height: 100vh;
            position: fixed;
            top: 0;
            left: 0;
            background-color: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            z-index: 100;
            padding: 20px;
            overflow-y: auto;
        }}
        
        .sidebar-header {{
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 15px;
        }}
        
        .sidebar-title {{
            font-size: 16px;
            font-weight: 800;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .sidebar-subtitle {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 4px;
        }}
        
        .nav-links {{
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 4px;
            flex-grow: 1;
        }}
        
        .nav-links li a {{
            display: block;
            padding: 8px 12px;
            border-radius: 6px;
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 13.5px;
            line-height: 1.4;
            transition: all 0.2s;
        }}
        
        .nav-links li a:hover {{
            background-color: var(--accent-glow);
            color: var(--accent);
            transform: translateX(4px);
        }}
        
        .nav-links li.active a {{
            background-color: var(--accent-glow);
            color: var(--accent);
            font-weight: 700;
            border-left: 3px solid var(--accent);
        }}
        
        .sidebar-footer {{
            padding-top: 15px;
            border-top: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        
        .btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 9px 14px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            border: 1px solid var(--border-color);
            background-color: var(--bg-card);
            color: var(--text-primary);
            text-decoration: none;
            transition: all 0.2s;
        }}
        
        .btn:hover {{
            border-color: var(--accent);
            color: var(--accent);
        }}
        
        .btn-primary {{
            background: linear-gradient(135deg, #0284C7, #2563EB);
            color: #FFFFFF;
            border: none;
        }}
        
        .btn-primary:hover {{
            opacity: 0.95;
            color: #FFFFFF;
        }}
        
        /* Main content container */
        #main-content {{
            margin-left: var(--sidebar-width);
            flex-grow: 1;
            padding: 40px 60px;
            max-width: 1100px;
        }}
        
        /* Typography and Markdown Styling */
        h1 {{
            font-size: 28px;
            margin-bottom: 12px;
            color: var(--accent);
            line-height: 1.3;
        }}
        
        h2 {{
            font-size: 22px;
            margin-top: 40px;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 1.5px solid var(--border-color);
            color: var(--text-primary);
            scroll-margin-top: 30px;
        }}
        
        h3 {{
            font-size: 18px;
            margin-top: 28px;
            margin-bottom: 12px;
            color: #38BDF8;
            scroll-margin-top: 30px;
        }}
        
        h4 {{
            font-size: 15px;
            margin-top: 20px;
            margin-bottom: 8px;
            color: #FBBF24;
        }}
        
        p {{
            margin-bottom: 16px;
            font-size: 15px;
            color: var(--text-primary);
        }}
        
        ul, ol {{
            margin-bottom: 16px;
            padding-left: 24px;
            font-size: 15px;
        }}
        
        li {{
            margin-bottom: 6px;
        }}
        
        /* Table styling */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 24px 0;
            font-size: 13.5px;
            background-color: var(--bg-card);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }}
        
        th, td {{
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background-color: rgba(56, 189, 248, 0.1);
            color: var(--accent);
            font-weight: 700;
        }}
        
        tr:hover td {{
            background-color: rgba(255, 255, 255, 0.03);
        }}
        
        /* Code & Pre */
        pre {{
            background-color: var(--code-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            overflow-x: auto;
            margin: 20px 0;
            font-family: Consolas, Monaco, "Courier New", monospace;
            font-size: 13px;
        }}
        
        code {{
            font-family: Consolas, Monaco, "Courier New", monospace;
            background-color: var(--code-bg);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 13.5px;
            color: #38BDF8;
        }}
        
        pre code {{
            background-color: transparent;
            padding: 0;
            color: var(--text-primary);
        }}
        
        /* Blockquote / Alerts */
        blockquote {{
            border-left: 4px solid var(--accent);
            background-color: var(--accent-glow);
            padding: 14px 18px;
            border-radius: 0 8px 8px 0;
            margin: 20px 0;
            font-size: 14.5px;
        }}
        
        /* Mermaid diagrams container */
        .mermaid {{
            margin: 24px 0;
            padding: 20px;
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            text-align: center;
        }}
        
        hr {{
            border: none;
            height: 1px;
            background-color: var(--border-color);
            margin: 35px 0;
        }}
        
        /* Reading progress bar */
        #progress-bar {{
            position: fixed;
            top: 0;
            left: var(--sidebar-width);
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #38BDF8, #3B82F6, #8B5CF6);
            width: 0%;
            z-index: 999;
            transition: width 0.1s;
        }}
        
        @media (max-width: 900px) {{
            #sidebar {{
                display: none;
            }}
            #main-content {{
                margin-left: 0;
                padding: 20px;
            }}
            #progress-bar {{
                left: 0;
            }}
        }}
    </style>
</head>
<body data-theme="dark">

    <div id="progress-bar"></div>

    <!-- Sidebar Navigation -->
    <nav id="sidebar">
        <div class="sidebar-header">
            <div class="sidebar-title">
                <span>🏦</span> BANKING77
            </div>
            <div class="sidebar-subtitle">Báo cáo Khai phá Dữ liệu Toàn diện</div>
        </div>
        
        <ul class="nav-links" id="toc-nav">
            <!-- Generated dynamically by JS -->
        </ul>
        
        <div class="sidebar-footer">
            <button class="btn" onclick="toggleTheme()" id="theme-btn">
                <span>🌓</span> Đổi Giao diện Sáng/Tối
            </button>
            <a href="http://localhost:8501" target="_blank" class="btn btn-primary">
                <span>🚀</span> Mở Web Demo Tương Tác
            </a>
            <button class="btn" onclick="window.print()">
                <span>🖨️</span> In / Xuất PDF
            </button>
        </div>
    </nav>

    <!-- Main Content -->
    <main id="main-content">
        <div id="content-body">Đang tải và kết xuất báo cáo...</div>
    </main>

    <script>
        const rawMarkdown = {md_json};
        
        // Cấu hình marked.js
        marked.setOptions({{
            gfm: true,
            breaks: false,
            highlight: function(code, lang) {{
                return code;
            }}
        }});
        
        // Render Math và Markdown
        function renderDocument() {{
            // 1. Tách và giữ chỗ cho Mermaid
            const mermaidBlocks = [];
            let processedMd = rawMarkdown.replace(/```mermaid([\\s\\S]*?)```/g, (match, p1) => {{
                const id = 'mermaid-block-' + mermaidBlocks.length;
                mermaidBlocks.push({{ id, code: p1.trim() }});
                return `<div class="mermaid" id="${{id}}">${{p1.trim()}}</div>`;
            }});
            
            // 2. Render Markdown qua Marked
            const htmlContent = marked.parse(processedMd);
            const container = document.getElementById('content-body');
            container.innerHTML = htmlContent;
            
            // 3. Render KaTeX Math
            renderMathInElement(container);
            
            // 4. Khởi chạy Mermaid
            mermaid.initialize({{
                startOnLoad: false,
                theme: document.body.getAttribute('data-theme') === 'light' ? 'default' : 'dark',
                securityLevel: 'loose'
            }});
            mermaid.run();
            
            // 5. Tạo Table of Contents ở Sidebar
            buildTOC();
        }}
        
        // Render KaTeX cho inline ($...$) và display ($$...$$)
        function renderMathInElement(root) {{
            const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false);
            const textNodes = [];
            while (walker.nextNode()) textNodes.push(walker.currentNode);
            
            for (let node of textNodes) {{
                const text = node.nodeValue;
                if (!text || text.indexOf('$') === -1) continue;
                
                // Thay thế $$...$$ display math và $...$ inline math
                const regex = /\\$\\$([\\s\\S]+?)\\$\\$|\\$([^$]+?)\\$/g;
                let match;
                let lastIdx = 0;
                let fragments = [];
                let hasMath = false;
                
                while ((match = regex.exec(text)) !== null) {{
                    hasMath = true;
                    if (match.index > lastIdx) {{
                        fragments.push(document.createTextNode(text.substring(lastIdx, match.index)));
                    }}
                    const isDisplay = !!match[1];
                    const mathStr = match[1] || match[2];
                    const span = document.createElement('span');
                    try {{
                        katex.render(mathStr, span, {{ displayMode: isDisplay, throwOnError: false }});
                    }} catch (e) {{
                        span.textContent = match[0];
                    }}
                    fragments.push(span);
                    lastIdx = regex.lastIndex;
                }}
                
                if (hasMath) {{
                    if (lastIdx < text.length) {{
                        fragments.push(document.createTextNode(text.substring(lastIdx)));
                    }}
                    const parent = node.parentNode;
                    if (parent && !parent.closest('pre') && !parent.closest('code') && !parent.closest('.mermaid')) {{
                        for (let frag of fragments) {{
                            parent.insertBefore(frag, node);
                        }}
                        parent.removeChild(node);
                    }}
                }}
            }}
        }}
        
        // Xây dựng Mục lục động
        function buildTOC() {{
            const tocNav = document.getElementById('toc-nav');
            tocNav.innerHTML = '';
            
            const headings = document.querySelectorAll('#main-content h2');
            headings.forEach((h, idx) => {{
                // Tạo ID cho heading nếu chưa có
                if (!h.id) {{
                    h.id = 'heading-' + idx;
                }}
                const li = document.createElement('li');
                const a = document.createElement('a');
                a.href = '#' + h.id;
                // Rút gọn tiêu đề
                let title = h.innerText.replace(/^[0-9\\.\\s]+/, '');
                if (title.length > 38) title = title.substring(0, 38) + '...';
                a.innerText = title;
                li.appendChild(a);
                tocNav.appendChild(li);
            }});
        }}
        
        // Toggle Sáng / Tối
        function toggleTheme() {{
            const cur = document.body.getAttribute('data-theme');
            const next = cur === 'light' ? 'dark' : 'light';
            document.body.setAttribute('data-theme', next);
            // Re-render Mermaid diagrams with appropriate theme
            location.reload();
        }}
        
        // Scroll spy & Progress bar
        window.addEventListener('scroll', () => {{
            const winScroll = document.documentElement.scrollTop || document.body.scrollTop;
            const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
            const scrolled = (winScroll / height) * 100;
            document.getElementById('progress-bar').style.width = scrolled + '%';
            
            // Highlight current TOC section
            const headings = document.querySelectorAll('#main-content h2');
            let currentId = '';
            headings.forEach(h => {{
                const top = h.getBoundingClientRect().top;
                if (top <= 120) {{
                    currentId = h.id;
                }}
            }});
            
            const links = document.querySelectorAll('#toc-nav li');
            links.forEach(li => {{
                const a = li.querySelector('a');
                if (a.getAttribute('href') === '#' + currentId) {{
                    li.classList.add('active');
                }} else {{
                    li.classList.remove('active');
                }}
            }});
        }});
        
        document.addEventListener('DOMContentLoaded', renderDocument);
    </script>
</body>
</html>
"""
    
    with open(html_out_path, "w", encoding="utf-8") as f:
        f.write(html_template)
        
    print(f"Successfully generated HTML report at: {os.path.abspath(html_out_path)}")

if __name__ == "__main__":
    build_html_report()
