import matplotlib
matplotlib.use('Agg')  # Неинтерактивный бэкенд для PDF

import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import dash_daq as daq
import plotly.graph_objs as go
import numpy as np
from PIL import Image
import base64
import io
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Img(
            src='/assets/logo.png',
            style={
                "height": "70px",
                "position": "absolute",
                "right": "40px",
                "top": "20px",
                "zIndex": 999
            }
        )
    ], style={"position": "relative", "height": "70px"}),
    html.H2(
        "BotkinICG – Веб-анализ ICG изображений",
        style={
            "font-family": "Montserrat, sans-serif",
            "color": "#003366"
        }
    ),
    dcc.Upload(
        id='upload-image',
        children=html.Button('Загрузить изображение', style={
            "margin": "5px",
            "border": "2px solid #003366",
            "color": "#003366",
            "background": "white",
            "font-family": "Montserrat, sans-serif",
            "font-weight": "600"
        }),
        multiple=False
    ),
    html.Div([
        html.Button("Автоматически выделить ROI", id="auto-roi-btn", n_clicks=0, style={
            "margin": "5px",
            "border": "2px solid #003366",
            "color": "#003366",
            "background": "white",
            "font-family": "Montserrat, sans-serif",
            "font-weight": "600"
        }),
        html.Button("Показать тепловую карту", id="heatmap-btn", n_clicks=0, style={
            "margin": "5px",
            "border": "2px solid #003366",
            "color": "#003366",
            "background": "white",
            "font-family": "Montserrat, sans-serif",
            "font-weight": "600"
        }),
        html.Button("Очистить ROI", id="clear-roi-btn", n_clicks=0, style={
            "margin": "5px",
            "border": "2px solid #003366",
            "color": "#003366",
            "background": "white",
            "font-family": "Montserrat, sans-serif",
            "font-weight": "600"
        }),
        html.Button("Скачать CSV", id='download-csv-btn', n_clicks=0, style={
            "margin": "5px",
            "border": "2px solid #003366",
            "color": "#003366",
            "background": "white",
            "font-family": "Montserrat, sans-serif",
            "font-weight": "600"
        }),
        html.Button("Скачать PDF", id="download-pdf-btn", n_clicks=0, style={
            "margin": "5px",
            "border": "2px solid #003366",
            "color": "#003366",
            "background": "white",
            "font-family": "Montserrat, sans-serif",
            "font-weight": "600"
        }),
        dcc.Download(id="download-pdf"),
    ]),
    dcc.Graph(
        id='img-plot',
        config={'displayModeBar': True, 'modeBarButtonsToAdd': ['drawrect']},
        style={"height": "750px"}
    ),
    dcc.Download(id="download-csv"),
    dcc.Graph(id='roi-bar-chart'),
    dcc.Graph(id="heatmap-graph", style={"display": "none"}),
    html.Div(id='roi-result', style={"margin-top": "30px", "font-family": "Montserrat, sans-serif"}),
    html.Div(id="roi-analysis", style={"margin-top": "32px", "font-family": "Montserrat, sans-serif", "font-size": "16px", "color": "#003366"}),
    dcc.Graph(id="roi-analysis-plot", style={"height": "400px"}),
    dcc.Graph(id="roi-violin-plot", style={"height": "400px"}),
    html.Div("Violin plot по ROI", style={"font-family": "Montserrat, sans-serif", "font-size": "16px", "color": "#003366", "margin-top": "10px", "margin-bottom": "10px"}),
    dcc.Graph(id="img-hist-plot", style={"height": "400px"}),
    html.Div("Гистограмма по всему изображению", style={"font-family": "Montserrat, sans-serif", "font-size": "16px", "color": "#003366", "margin-top": "10px", "margin-bottom": "10px"}),
    # store info for stateful callbacks
    dcc.Store(id='img-np'),
    dcc.Store(id='roi-shapes'),
    html.Div(
        "Программа BotkinICG, визуальный интерфейс и аналитические алгоритмы являются объектами интеллектуальной собственности. Все права защищены ©Аладин Марк Николаевич. BotkinICG, 2025.",
        style={
            "margin": "40px 0 10px 0",
            "font-family": "Montserrat, sans-serif",
            "color": "#003366",
            "font-size": "13px",
            "text-align": "center"
        }
    ),
])

def parse_image(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded)).convert("RGB")
    # Ограничение размера изображения для экономии памяти
    MAX_SIZE = (1200, 1200)
    if image.size[0] > MAX_SIZE[0] or image.size[1] > MAX_SIZE[1]:
        image.thumbnail(MAX_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(image)
    return arr

def roi_stats(img_np, shapes):
    stats = []
    for idx, shape in enumerate(shapes, 1):
        if shape["type"] == "rect":
            x0, x1 = int(round(shape["x0"])), int(round(shape["x1"]))
            y0, y1 = int(round(shape["y0"])), int(round(shape["y1"]))
            x0, x1 = sorted([max(0, x0), max(0, x1)])
            y0, y1 = sorted([max(0, y0), max(0, y1)])
            roi = img_np[y0:y1, x0:x1, 1]  # green channel
            mean_int = float(np.mean(roi)) if roi.size > 0 else 0.0
            sd_int = float(np.std(roi, ddof=1)) if roi.size > 1 else 0.0
            stats.append({
                "ROI": f"ROI_{idx}",
                "x0": x0, "x1": x1, "y0": y0, "y1": y1,
                "Mean_intensity": mean_int,
                "SD_intensity": sd_int
            })
    return stats

def auto_rois(img_np, n=3, size_px=50):
    """Автоматически выбирает n самых ярких не пересекающихся квадратов (по зелёному каналу)."""
    gray = img_np[:, :, 1].astype(np.float32)
    k = size_px
    avg = cv2.boxFilter(gray, ddepth=-1, ksize=(k, k), normalize=True)
    pad = k // 2
    avg_crop = avg[pad:-pad or None, pad:-pad or None].copy()
    h_crop, w_crop = avg_crop.shape
    selected = []
    for _ in range(n):
        _, maxVal, _, maxLoc = cv2.minMaxLoc(avg_crop)
        if np.isneginf(maxVal):
            break
        cx, cy = maxLoc
        selected.append((cx, cy))
        x0 = max(cx - k, 0)
        y0 = max(cy - k, 0)
        x1 = min(cx + k, w_crop - 1)
        y1 = min(cy + k, h_crop - 1)
        avg_crop[y0:y1 + 1, x0:x1 + 1] = -np.inf
    shapes = []
    h0, w0 = gray.shape
    for cx, cy in selected:
        x0 = cx + pad
        y0 = cy + pad
        x1 = min(x0 + k, w0)
        y1 = min(y0 + k, h0)
        shapes.append({"type": "rect", "x0": x0, "x1": x1, "y0": y0, "y1": y1})
    return shapes

def get_heatmap_fig(img_np, alpha=0.5):
    gray = img_np[:, :, 1]
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    heat_bgr = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    blend = cv2.addWeighted(img_np, 1-alpha, heat_rgb, alpha, 0)
    fig = go.Figure(go.Image(z=blend))
    fig.update_layout(
        title="Тепловая карта (JET)",
        height=750,
        font_family="Montserrat, sans-serif",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.04)', gridwidth=0.7),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.04)', gridwidth=0.7)
    )
    return fig

@app.callback(
    Output('img-plot', 'figure'),
    Output('roi-result', 'children'),
    Output('roi-bar-chart', 'figure'),
    Output('download-csv', 'data'),
    Output('roi-shapes', 'data'),
    Output('img-np', 'data'),
    Output('heatmap-graph', 'figure'),
    Output('heatmap-graph', 'style'),
    Output('roi-analysis', 'children'),
    Output('roi-analysis-plot', 'figure'),
    Output('download-pdf', 'data'),
    Output('roi-violin-plot', 'figure'),
    Output('img-hist-plot', 'figure'),
    Input('upload-image', 'contents'),
    Input('img-plot', 'relayoutData'),
    Input('download-csv-btn', 'n_clicks'),
    Input('auto-roi-btn', 'n_clicks'),
    Input('heatmap-btn', 'n_clicks'),
    Input('clear-roi-btn', 'n_clicks'),
    Input('download-pdf-btn', 'n_clicks'),
    State('upload-image', 'filename'),
    State('roi-shapes', 'data'),
    State('img-np', 'data'),
    prevent_initial_call=True
)
def update_output(contents, relayoutData, n_csv, n_auto, n_heat, n_clear, n_pdf, filename, roi_shapes, img_np_data):
    trigger = ctx.triggered_id
    # Image upload
    if contents:
        img_np = parse_image(contents)
        img_np_list = img_np.tolist()  # for dcc.Store
    elif img_np_data is not None:
        img_np = np.array(img_np_data, dtype=np.uint8)
        img_np_list = img_np_data
    else:
        empty_fig = go.Figure()
        return empty_fig, "", empty_fig, None, [], None, empty_fig, {"display": "none"}, "", empty_fig, None, empty_fig, empty_fig
    # Shapes
    shapes = roi_shapes if roi_shapes else []
    # Auto ROI
    if trigger == "auto-roi-btn":
        shapes = auto_rois(img_np, n=3, size_px=50)
    # Очистить ROI — очищаем вне зависимости от relayoutData
    if ctx.triggered_id == "clear-roi-btn":
        shapes = []
    # Только если НЕ была нажата очистка, обрабатываем relayoutData
    elif relayoutData and "shapes" in relayoutData:
        shapes = relayoutData["shapes"]
    # ROI stats
    stats = roi_stats(img_np, shapes)
    intensities = [r["Mean_intensity"] for r in stats]
    colors = []
    for i, r in enumerate(stats):
        if not intensities: colors.append('yellow')
        elif intensities[i] == max(intensities): colors.append('green')
        elif intensities[i] == min(intensities): colors.append('red')
        else: colors.append('yellow')
    fig = go.Figure()
    fig.add_trace(go.Image(z=img_np))
    fig.update_layout(
        dragmode='drawrect',
        height=750,
        font_family="Montserrat, sans-serif",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.04)', gridwidth=0.7),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.04)', gridwidth=0.7)
    )
    for i, shape in enumerate(shapes):
        if shape["type"] == "rect":
            fig.add_shape(type="rect",
                x0=shape["x0"], x1=shape["x1"], y0=shape["y0"], y1=shape["y1"],
                line=dict(color=colors[i], width=2))
            mx = (shape["x0"] + shape["x1"]) / 2
            my = min(shape["y0"], shape["y1"])
            sd_val = stats[i]["SD_intensity"] if i < len(stats) else 0.0
            fig.add_annotation(
                x=mx, y=my-10, text=f"ROI_{i+1} (SD: {sd_val:.2f})",
                showarrow=False, font=dict(color=colors[i], size=13, family="Montserrat, sans-serif"), bgcolor="white"
            )
    # ROI bar chart
    bar_fig = go.Figure()
    if stats:
        bar_fig.add_bar(
            x=[r["ROI"] for r in stats],
            y=[r["Mean_intensity"] for r in stats],
            marker_color=colors
        )
        bar_fig.update_layout(
            title="Средняя интенсивность по ROI (зеленый канал)",
            yaxis_title="Mean intensity",
            xaxis_title="ROI",
            font_family="Montserrat, sans-serif",
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.04)', gridwidth=0.7),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.04)', gridwidth=0.7)
        )
    # Text summary
    summary = ""
    if stats:
        df = pd.DataFrame(stats)
        best = df.loc[df["Mean_intensity"].idxmax(), "ROI"]
        worst = df.loc[df["Mean_intensity"].idxmin(), "ROI"]
        summary = (
            f"Всего ROI: {len(stats)} | "
            f"Лучший участок: <b>{best}</b> | "
            f"Худший участок: <b>{worst}</b>"
        )
        summary += "<br>"
        summary += "<br>".join([
            f'{r["ROI"]}: [{r["x0"]}:{r["x1"]}, {r["y0"]}:{r["y1"]}] – {r["Mean_intensity"]:.1f} (SD: {r["SD_intensity"]:.2f})'
            for r in stats
        ])
    else:
        summary = "ROI не выделены"
    # Download CSV
    download = None
    if trigger == "download-csv-btn" and stats:
        df = pd.DataFrame(stats)
        csv_str = df.to_csv(index=False)
        b64 = base64.b64encode(csv_str.encode('utf-8')).decode()
        download = dict(
            content=b64,
            filename="icg_stats.csv",
            base64=True
        )
    # Heatmap
    heatmap_fig = go.Figure()
    heatmap_style = {"display": "none"}
    if trigger == "heatmap-btn":
        heatmap_fig = get_heatmap_fig(img_np, alpha=0.5)
        heatmap_style = {"display": "block"}

    # ROI анализ
    analysis_txt = ""
    analysis_fig = go.Figure()
    vals = [r["Mean_intensity"] for r in stats] if stats else []
    if stats:
        mean_val = np.mean(vals)
        sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        cv = (sd / mean_val * 100.0) if mean_val else 0.0
        snr = (mean_val / sd) if sd else float("inf")
        analysis_txt = f"""
       <b>Анализ ROI:</b><br>
       Среднее: {mean_val:.2f}<br>
       SD: {sd:.2f}<br>
       CV: {cv:.2f}%<br>
       SNR: {snr:.2f}
       """
        analysis_fig.add_trace(go.Box(y=vals, name='Boxplot', marker_color="#003366"))
        analysis_fig.update_layout(title="Boxplot по ROI", font_family="Montserrat, sans-serif")
    else:
        analysis_txt = "Нет ROI для анализа."

    # Violin plot по ROI
    violin_fig = go.Figure()
    roi_values = []
    roi_labels = []
    if stats:
        # Получаем массивы значений из каждого ROI
        for i, shape in enumerate(shapes):
            if shape["type"] == "rect":
                x0, x1 = int(round(shape["x0"])), int(round(shape["x1"]))
                y0, y1 = int(round(shape["y0"])), int(round(shape["y1"]))
                x0, x1 = sorted([max(0, x0), max(0, x1)])
                y0, y1 = sorted([max(0, y0), max(0, y1)])
                roi = img_np[y0:y1, x0:x1, 1].flatten()
                if roi.size > 0:
                    roi_values.append(roi)
                    roi_labels.append(f"ROI_{i+1}")
        for i, values in enumerate(roi_values):
            violin_fig.add_trace(go.Violin(y=values, name=roi_labels[i], box_visible=True, meanline_visible=True))
        violin_fig.update_layout(
            title="Violin plot по ROI (зелёный канал)",
            font_family="Montserrat, sans-serif",
            yaxis_title="Интенсивность",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
    else:
        violin_fig.update_layout(
            title="Violin plot по ROI (зелёный канал) - нет данных",
            font_family="Montserrat, sans-serif"
        )

    # Гистограмма по всему изображению (зелёный канал)
    hist_fig = go.Figure()
    green_vals = img_np[:, :, 1].flatten()
    hist_fig.add_trace(go.Histogram(x=green_vals, nbinsx=50, marker_color="#003366"))
    hist_fig.update_layout(
        title="Гистограмма интенсивности по всему изображению (зелёный канал)",
        font_family="Montserrat, sans-serif",
        xaxis_title="Интенсивность",
        yaxis_title="Частота",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    # --- PDF export ---
    download_pdf = None
    if ctx.triggered_id == "download-pdf-btn" and stats:
        pdf_bytes = io.BytesIO()
        with PdfPages(pdf_bytes) as pdf:
            # 1: Исходник с ROI (matplotlib только для PDF!)
            colors = []
            for i, r in enumerate(stats):
                if not intensities: colors.append('yellow')
                elif intensities[i] == max(intensities): colors.append('green')
                elif intensities[i] == min(intensities): colors.append('red')
                else: colors.append('yellow')
            fig_, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(img_np)
            for i, s in enumerate(shapes):
                rect = plt.Rectangle((s["x0"], s["y0"]), s["x1"]-s["x0"], s["y1"]-s["y0"], edgecolor=colors[i], facecolor='none', linewidth=2)
                ax.add_patch(rect)
                sd_val = stats[i]["SD_intensity"] if i < len(stats) else 0.0
                ax.text(s["x0"], s["y0"], f"ROI_{i+1} (SD: {sd_val:.2f})", color=colors[i], fontsize=8)
            plt.title("ICG с ROI")
            ax.axis("off")
            pdf.savefig(fig_)
            plt.close(fig_)
            import gc; gc.collect()
            # 2: Boxplot
            fig2_, ax2_ = plt.subplots(figsize=(6, 4))
            ax2_.boxplot(vals)
            ax2_.set_title("Boxplot по ROI")
            pdf.savefig(fig2_)
            plt.close(fig2_)
            import gc; gc.collect()
            # 3: Таблица с SD
            df = pd.DataFrame(stats).round(2)
            n_rows, n_cols = df.shape
            fig3_, ax3_ = plt.subplots(figsize=(n_cols*1.5, 0.6 + 0.45 * n_rows))
            ax3_.axis('off')
            table = ax3_.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.5)
            pdf.savefig(fig3_)
            plt.close(fig3_)
            import gc; gc.collect()
            # 4: Тепловая карта
            heatmap_np = cv2.applyColorMap(cv2.normalize(img_np[:, :, 1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_np, cv2.COLOR_BGR2RGB)
            fig4_, ax4_ = plt.subplots(figsize=(7, 7))
            ax4_.imshow(heatmap_rgb)
            ax4_.set_title("Тепловая карта (JET)")
            ax4_.axis("off")
            pdf.savefig(fig4_)
            plt.close(fig4_)
            import gc; gc.collect()
            # 5: Violin plot по ROI (зелёный канал)
            # --- Violin plot: логирование размера данных, ограничение размера массива, обработка ошибок, лимит MAX_POINTS ---
            fig5_, ax5_ = plt.subplots(figsize=(7, 5))
            # Логирование размера данных для violin plot
            print("len(roi_values):", [len(v.flatten()) for v in roi_values])
            # Назначение лимита MAX_POINTS для экономии памяти и предотвращения переполнения
            MAX_POINTS = 5000  # Более строгий лимит по точкам
            MAX_ROI = 8        # Максимальное число ROI для PDF-отчёта
            # Ограничение размера массива для экономии памяти:
            roi_data = [vals.flatten() for vals in roi_values]
            roi_data = [
                v if len(v) <= MAX_POINTS else np.random.choice(v, MAX_POINTS, replace=False)
                for v in roi_data
            ]
            # Проверка лимитов
            if any(len(v) > MAX_POINTS for v in roi_values) or len(roi_values) > MAX_ROI:
                plt.close('all')
                import gc; gc.collect()
                download_pdf = dict(
                    content=None,
                    filename="error.txt",
                    base64=False
                )
                return fig, summary, bar_fig, download, shapes, img_np_list, heatmap_fig, heatmap_style, analysis_txt, analysis_fig, download_pdf, violin_fig, hist_fig
            # Обработка ошибок визуализации: если данных слишком много или ошибка построения
            try:
                parts = ax5_.violinplot(roi_data, showmeans=True, showmedians=True)
            except Exception as e:
                print("Ошибка построения violinplot:", str(e))
                ax5_.text(0.5, 0.5, 'Ошибка визуализации\nслишком много данных',
                          horizontalalignment='center', verticalalignment='center',
                          transform=ax5_.transAxes)
            ax5_.set_title("Violin plot по ROI (зелёный канал)")
            ax5_.set_xticks(np.arange(1, len(roi_labels)+1))
            ax5_.set_xticklabels(roi_labels)
            pdf.savefig(fig5_)
            plt.close(fig5_)
            import gc; gc.collect()
            # 6: Гистограмма по всему изображению
            fig6_, ax6_ = plt.subplots(figsize=(7, 5))
            ax6_.hist(green_vals, bins=50, color="#003366")
            ax6_.set_title("Гистограмма интенсивности по всему изображению (зелёный канал)")
            pdf.savefig(fig6_)
            plt.close(fig6_)
            import gc; gc.collect()
        pdf_bytes.seek(0)
        b64pdf = base64.b64encode(pdf_bytes.read()).decode()
        download_pdf = dict(
            content=b64pdf,
            filename="icg_report.pdf",
            base64=True
        )

    return fig, summary, bar_fig, download, shapes, img_np_list, heatmap_fig, heatmap_style, analysis_txt, analysis_fig, download_pdf, violin_fig, hist_fig

server = app.server

if __name__ == '__main__':
    app.run(debug=True)