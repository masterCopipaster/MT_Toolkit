# -*- coding: utf-8 -*-
"""
T-Heatmap + GPS-KML Toolkit  (v2.1, 2025-05-13)
------------------------------------------------
• гибкая / дискретная теплокарта (чек-бокс «Дискретная палитра»)
• поиск аномалий по скачкам кривых
• генерация GPS-точек + экспорт KML (цвет подписи)
"""

import os, sys, tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Tuple

import traceback
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap, BoundaryNorm   # NEW
from pyproj import Geod

# -------------------------------------------------------------
#  paste helper (Ctrl+C / Ctrl+V)
# -------------------------------------------------------------
def enable_copy_paste(entry: tk.Entry):
    def _paste(event):
        try:
            raw = entry.clipboard_get()
            entry.delete(0, tk.END)
            entry.insert(0, raw.strip().replace(',', '.').replace(' ', ''))
        except Exception:
            pass
        return "break"
    entry.bind('<Control-v>', _paste)
    entry.bind('<Control-V>', _paste)           # оба варианта
    entry.bind('<Control-c>', lambda e: e.widget.event_generate('<<Copy>>'))
    entry.bind('<Control-C>', lambda e: e.widget.event_generate('<<Copy>>'))
    entry.bind('<Control-a>', lambda e: e.widget.event_generate('<<SelectAll>>'))
    entry.bind('<Control-A>', lambda e: e.widget.event_generate('<<SelectAll>>'))

# -------------------------------------------------------------
#  АНОМАЛИИ: скачки сопротивлений
# -------------------------------------------------------------
def detect_jumps(curves: np.ndarray, sensitivity: int = 5) -> np.ndarray:
    if curves.shape[0] < 2:
        return np.zeros_like(curves, bool)
    d_abs = np.abs(np.diff(curves, axis=0))
    thr = np.percentile(d_abs, 100 - sensitivity * 4.5)   # 1>99-й … 10>55-й
    mask_low = np.vstack([d_abs > thr, np.zeros((1, curves.shape[1]), bool)])
    mask_up  = np.vstack([np.zeros((1, curves.shape[1]), bool), d_abs > thr])
    return mask_low | mask_up

# -------------------------------------------------------------
#  ПОЛИНОМИАЛЬНАЯ НОРМИРОВКА ДАННЫХ
# -------------------------------------------------------------

def apply_polynomial_norm(curves: np.ndarray, order: int = 1) -> np.ndarray:
    #average data by lines
    print(curves.shape)
    print(curves)
    av = np.average(curves, 0)
    print(av)
    x = np.arange(av.shape[0])
    pol = np.polyval(np.polyfit(x, 1/av, order), x)
    print(pol)
    ret = curves * pol.reshape(1, -1)
    print(ret)
    return ret
    
def apply_polynomial_norm_df(df: pd.DataFrame, order: int = 1) -> pd.DataFrame:
    if 'N' not in df.columns:
        raise ValueError('CSV должен содержать колонку N')

    new_df = pd.DataFrame()
    new_df['N'] = df['N']
    freq_cols  = [c for c in df if c.startswith('freq')]
    data = df[freq_cols].to_numpy()
    output = apply_polynomial_norm(data, order)
    for f in range(1, output.shape[1] + 1):
        new_df[f"freq{f}"] = output[:, f - 1]
    return new_df
    
    
def apply_logarithmic_scale_df(df: pd.DataFrame) -> pd.DataFrame:
    if 'N' not in df.columns:
        raise ValueError('CSV должен содержать колонку N')

    new_df = pd.DataFrame()
    new_df['N'] = df['N']
    freq_cols  = [c for c in df if c.startswith('freq')]
    data = df[freq_cols].to_numpy()
    output = np.log(data)
    minn = np.min(output[np.isfinite(output)])
    output[np.isnan(output)] = minn
    output[np.isinf(output)] = minn
    for f in range(1, output.shape[1] + 1):
        new_df[f"freq{f}"] = output[:, f - 1]
    return new_df
 
 
def apply_diff_filter_df(df: pd.DataFrame, force: int = 1) -> pd.DataFrame:
    if 'N' not in df.columns:
        raise ValueError('CSV должен содержать колонку N')

    new_df = pd.DataFrame()
    new_df['N'] = df['N']
    
    fil = np.array(
    [
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]
    ])
    
    freq_cols  = [c for c in df if c.startswith('freq')]
    data = df[freq_cols].to_numpy()
    output = data
    for i in range(force):
        output = scipy.signal.convolve2d(output, fil, mode = 'same', boundary = 'symm')
    print(output.shape)
    for f in range(1, output.shape[1] + 1):
        new_df[f"freq{f}"] = output[:, f - 1]
    return new_df


# -------------------------------------------------------------
#  ПЛОТТЕР
# -------------------------------------------------------------
def build_heatmap_figure(
        df: pd.DataFrame,
        cmap: str,
        depth_rng: Tuple[int, int],
        n_rng: Tuple[int, int],
        highlight: bool,
        sensitivity: int,
        discrete: bool = False        # < новое
) -> plt.Figure:

    if 'N' not in df.columns:
        raise ValueError('CSV должен содержать колонку N')

    df = df.sort_values('N')
    df = df[(df['N'] >= n_rng[0]) & (df['N'] <= n_rng[1])]
    if df.empty:
        raise ValueError('Нет данных в выбранном диапазоне N')

    freq_cols  = [c for c in df if c.startswith('freq')]
    depths_all = np.arange(1, len(freq_cols)+1)
    mask_depth = (depths_all >= depth_rng[0]) & (depths_all <= depth_rng[1])
    sel_cols   = [c for i, c in enumerate(freq_cols) if mask_depth[i]]
    if not sel_cols:
        raise ValueError('Пустой диапазон глубин')

    heatmap = df[sel_cols].to_numpy().T          # depth ? N
    depths  = depths_all[mask_depth]
    n_vals  = df['N'].to_numpy()

    fig, (ax_top, ax_hm) = plt.subplots(
        2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [1.4, 2]}
    )

    # кривые
    cm     = plt.get_cmap(cmap)
    colors = cm(np.linspace(0, 1, len(depths)))
    for i, d in enumerate(depths):
        ax_top.plot(n_vals, heatmap[i], lw=1.4, color=colors[i], label=f'{d} м')
    ax_top.set(title='Кривые сопротивлений', xlabel='Пикет N', ylabel='Ом·м')
    ax_top.set_xticks(n_vals); ax_top.grid(True, ls=':', lw=0.6)
    step = max(1, len(depths)//12)
    ax_top.legend(
        handles=[plt.Line2D([0],[0], color=colors[i], lw=2,
                 label=f'{depths[i]} м') for i in range(0, len(depths), step)],
        bbox_to_anchor=(1.02,1), loc='upper left', fontsize='small')

    # палитра: плавная vs дискретная
    if discrete:
        N_BANDS = 8
        disc_cmap = ListedColormap(cm(np.linspace(0,1,N_BANDS)))
        bounds    = np.linspace(heatmap.min(), heatmap.max(), N_BANDS+1)
        norm      = BoundaryNorm(bounds, N_BANDS)
        cmap_used, norm_used = disc_cmap, norm
    else:
        cmap_used, norm_used = cm, None

    im = ax_hm.imshow(
        heatmap, aspect='auto',
        cmap=cmap_used, norm=norm_used,
        origin='upper',
        extent=[n_vals.min()-0.5, n_vals.max()+0.5,
                depths.max()+0.5, depths.min()-0.5]
    )
    fig.colorbar(im, ax=ax_hm, label='Ом·м', shrink=0.8)
    ax_hm.set(title='Тепловая карта', xlabel='Пикет N', ylabel='Глубина м')
    ax_hm.set_xticks(n_vals); ax_hm.set_yticks(depths)
    ax_hm.grid(True, ls=':', lw=0.6)

    if highlight:
        m = detect_jumps(heatmap, sensitivity)
        ys, xs = np.where(m)
        if xs.size:
            ax_hm.scatter(n_vals[xs], depths[ys],
                          s=60, facecolors='none', edgecolors='red', lw=1.4)
    fig.tight_layout()
    return fig

# -------------------------------------------------------------
#  ГЕОДЕЗИЯ + KML  (без изменений)
# -------------------------------------------------------------
def geo_points(lat1, lon1, lat2, lon2, step)->List[Tuple[float,float]]:
    g = Geod(ellps='WGS84'); _,_,d = g.inv(lon1,lat1,lon2,lat2)
    n = int(d//step)
    return ([(lat1,lon1)] +
            [(lat,lon) for lon,lat in g.npts(lon1,lat1,lon2,lat2,n-1)] if n>=1 else []
            +[(lat2,lon2)])

COLOR_ORDER = ['black','yellow','green','red']
LABEL_HEX = {'black':'ff000000','yellow':'ff00ffff','green':'ff00ff00','red':'ff0000ff'}
KML_HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>{}</name>"""
KML_FOOTER = "</Document></kml>\n"
def build_kml(name, pts, cols, prefix, start)->str:
    doc=[KML_HEADER.format(name)]
    doc+= [f"<Style id='{c}'><LabelStyle><color>{h}</color></LabelStyle></Style>"
           for c,h in LABEL_HEX.items()]
    for i,((lat,lon),c) in enumerate(zip(pts,cols)):
        nm=f"{prefix}{i+start}"
        doc.append(f"<Placemark><name>{nm}</name><styleUrl>#{c}</styleUrl>"
                   f"<Point><coordinates>{lon:.7f},{lat:.7f},0</coordinates></Point></Placemark>")
    doc.append(KML_FOOTER); return '\n'.join(doc)

# -------------------------------------------------------------
#  GUI
# -------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MT-Heatmap + GPS-KML Toolkit")
        self.geometry("980x760")
        self.protocol('WM_DELETE_WINDOW', self._quit)

        self._df=None; self._canvas=None
        self._pts=[]; self._cols=[]
        self._build_gui()

    # ---------- layout ----------
    def _build_gui(self):
        nb = ttk.Notebook(self); nb.pack(fill='both', expand=True)
        self.tab_hm = ttk.Frame(nb); nb.add(self.tab_hm, text='Heatmap')
        self.tab_gps= ttk.Frame(nb); nb.add(self.tab_gps,text='GPS / KML')
        self._hm_widgets(); self._gps_widgets()

    # ---------- Heatmap widgets ----------
    def _hm_widgets(self):
        pad=8; ctrl=ttk.Frame(self.tab_hm); ctrl.pack(side='left', fill='y', padx=pad, pady=pad)

        ttk.Label(ctrl,text='CSV-файл').pack(anchor='w')
        self.csv_var=tk.StringVar()
        fr=ttk.Frame(ctrl); fr.pack(fill='x')
        ttk.Entry(fr,textvariable=self.csv_var,width=34).pack(side='left',fill='x',expand=True)
        ttk.Button(fr,text='…',command=self._load_csv).pack(side='left', padx=4)

        self.zmin=tk.IntVar(value=1); self.zmax=tk.IntVar(value=33)
        self.nmin=tk.IntVar(value=1); self.nmax=tk.IntVar(value=1)
        for lbl,var in (('Мин. глубина',self.zmin),('Макс. глубина',self.zmax),
                        ('Мин. N',self.nmin),('Макс. N',self.nmax)):
            ttk.Label(ctrl,text=lbl).pack(anchor='w'); tk.Spinbox(ctrl,from_=1,to=999,width=7,
                                                                  textvariable=var).pack()

        ttk.Label(ctrl,text='Палитра').pack(anchor='w', pady=(6,0))
        self.cmap_var=tk.StringVar(value='jet')
        ttk.Combobox(ctrl,textvariable=self.cmap_var,state='readonly',width=12,
                     values=sorted(plt.colormaps())).pack()

        self.disc_var= tk.BooleanVar()  # NEW
        ttk.Checkbutton(ctrl,text='Дискретная палитра',
                        variable=self.disc_var).pack(anchor='w')
                        
        self.polynorm= tk.BooleanVar()  # NEW
        ttk.Checkbutton(ctrl,text='Полиномиальная нормировка',
                        variable=self.polynorm).pack(anchor='w')
                        
        self.polyorder= tk.IntVar()
        self.polyorder.set(1)
        ttk.Label(ctrl,text="Порядок полинома").pack(anchor='w'); tk.Spinbox(ctrl,from_=1,to=999,width=7,
                                                                  textvariable=self.polyorder).pack()
                                                                  
        self.do_log= tk.BooleanVar()  # NEW
        ttk.Checkbutton(ctrl,text='Логарифмическая шкала',
                        variable=self.do_log).pack(anchor='w')
        
                                
        self.do_diff_filter= tk.BooleanVar()  # NEW
        ttk.Checkbutton(ctrl,text='Фильтр увеличения резкости',
                        variable=self.do_diff_filter).pack(anchor='w')
                        
        ttk.Label(ctrl,text='Резкость').pack(anchor='w')
        self.diff_var=tk.IntVar(value=1)
        tk.Scale(ctrl,from_=0,to=9,orient='horizontal',length=120,
                 variable=self.diff_var,showvalue=True).pack()
                        
        
        self.hl_var  = tk.BooleanVar()
        ttk.Checkbutton(ctrl,text='Подсветить аномалии',
                        variable=self.hl_var).pack(anchor='w', pady=(6,0))

        ttk.Label(ctrl,text='Чувствительность').pack(anchor='w')
        self.sens_var=tk.IntVar(value=5)
        tk.Scale(ctrl,from_=1,to=10,orient='horizontal',length=120,
                 variable=self.sens_var,showvalue=True).pack()

        ttk.Separator(ctrl,orient='horizontal').pack(fill='x',pady=6)
        ttk.Button(ctrl,text='Построить',command=self._draw_plot).pack(fill='x')
        ttk.Button(ctrl,text='Сохранить PNG…',command=self._save_png).pack(fill='x',pady=(4,0))

        self.fig_frame=ttk.Frame(self.tab_hm); self.fig_frame.pack(fill='both',expand=True,padx=5,pady=5)

    # ---------- GPS widgets ----------
    def _gps_widgets(self):
        pad=8; left=ttk.Frame(self.tab_gps); left.pack(side='left',fill='y',padx=pad,pady=pad)
        for lbl,dflt,nm in (('Широта A',55.0,'lat1'),('Долгота A',37.0,'lon1'),
                            ('Широта B',55.0,'lat2'),('Долгота B',37.0,'lon2')):
            ttk.Label(left,text=lbl).pack(anchor='w')
            var=tk.DoubleVar(value=dflt); setattr(self,nm,var)
            e=ttk.Entry(left,textvariable=var,width=10); e.pack(); enable_copy_paste(e)

        ttk.Label(left,text='Шаг (м)').pack(anchor='w',pady=(6,0))
        self.step_var=tk.DoubleVar(value=1.0)
        es=ttk.Entry(left,textvariable=self.step_var,width=8); es.pack(); enable_copy_paste(es)

        ttk.Label(left,text='Префикс').pack(anchor='w',pady=(6,0))
        self.pref_var=tk.StringVar(value='P'); ttk.Entry(left,textvariable=self.pref_var,width=6).pack()

        ttk.Label(left,text='Нумерация с').pack(anchor='w')
        self.start_var=tk.IntVar(value=1)
        frn=ttk.Frame(left); frn.pack()
        ttk.Radiobutton(frn,text='0',variable=self.start_var,value=0).pack(side='left')
        ttk.Radiobutton(frn,text='1',variable=self.start_var,value=1).pack(side='left')

        ttk.Separator(left,orient='horizontal').pack(fill='x',pady=6)
        ttk.Button(left,text='Сгенерировать',command=self._gen_points).pack(fill='x')

        ttk.Label(left,text='Цвет выбранных').pack(anchor='w',pady=(10,0))
        self._make_color_buttons(left)

        ttk.Separator(left,orient='horizontal').pack(fill='x',pady=6)
        ttk.Button(left,text='Экспорт KML…',command=self._export_kml).pack(fill='x')

        self.lb=tk.Listbox(self.tab_gps,selectmode='extended')
        self.lb.pack(side='left',fill='both',expand=True,padx=5,pady=5)
        self.lb.bind('<Double-1>',self._cycle_color)

    # ---------- helpers ----------
    def _make_color_buttons(self,parent):
        st=ttk.Style(self)
        for c in COLOR_ORDER:
            st.configure(f"{c}.TButton",background=c,foreground='white' if c=='black' else 'black')
            st.map(f"{c}.TButton",background=[('active',c)])
        fr=ttk.Frame(parent); fr.pack()
        for c in COLOR_ORDER:
            ttk.Button(fr,text=c.capitalize(),style=f"{c}.TButton",
                       command=lambda col=c:self._set_color(col)).pack(side='left',padx=2)

    # ---------- actions ----------
    def _load_csv(self):
        p=filedialog.askopenfilename(filetypes=[('CSV','*.csv')]); 
        if not p: return
        self.csv_var.set(p)
        try:
            df=pd.read_csv(p); self._df=df
            freq=[c for c in df if c.startswith('freq')]
            self.zmax.set(len(freq)); self.nmin.set(int(df['N'].min())); self.nmax.set(int(df['N'].max()))
        except Exception as e: messagebox.showerror('Ошибка',str(e)); self._df=None

    def _draw_plot(self):
        if self._df is None:
            messagebox.showwarning('Нет данных','Сначала выберите CSV-файл'); return
        try:
            self._df_postproc = self._df
            if self.polynorm.get():
                self._df_postproc = apply_polynomial_norm_df(self._df_postproc, self.polyorder.get())
                
            if self.do_log.get():
                self._df_postproc = apply_logarithmic_scale_df(self._df_postproc)   
                
            if self.do_diff_filter.get():
                self._df_postproc = apply_diff_filter_df(self._df_postproc, force = self.diff_var.get())
                
            fig=build_heatmap_figure(self._df_postproc, self.cmap_var.get(),
                                     (self.zmin.get(),self.zmax.get()),
                                     (self.nmin.get(),self.nmax.get()),
                                     self.hl_var.get(), self.sens_var.get(),
                                     self.disc_var.get())
        except Exception as e: messagebox.showerror('Ошибка',traceback.format_exc()); return
        for w in self.fig_frame.winfo_children(): w.destroy()
        self._canvas=FigureCanvasTkAgg(fig,master=self.fig_frame)
        self._canvas.draw(); self._canvas.get_tk_widget().pack(fill='both',expand=True)

    def _save_png(self):
        if not self._canvas:
            messagebox.showwarning('Нет изображения','Сначала постройте график'); return
        p=filedialog.asksaveasfilename(defaultextension='.png',filetypes=[('PNG','*.png')])
        if p: self._canvas.figure.savefig(p,dpi=300); messagebox.showinfo('Сохранено',p)

    # ---------- GPS ----------
    def _gen_points(self):
        try:
            self._pts=geo_points(self.lat1.get(),self.lon1.get(),
                                 self.lat2.get(),self.lon2.get(),self.step_var.get())
        except Exception as e: messagebox.showerror('Ошибка',str(e)); return
        self._cols=['black']*len(self._pts); self.lb.delete(0,tk.END)
        for i in range(len(self._pts)):
            self.lb.insert(tk.END,f"{self.pref_var.get()}{i+self.start_var.get()}")
            self.lb.itemconfig(i,fg='black')

    def _set_color(self,color):
        for idx in self.lb.curselection():
            self._cols[idx]=color; self.lb.itemconfig(idx,fg=color)

    def _cycle_color(self,_):
        idx=self.lb.nearest(self.lb.winfo_pointery()-self.lb.winfo_rooty())
        cur=self._cols[idx]; nxt=COLOR_ORDER[(COLOR_ORDER.index(cur)+1)%len(COLOR_ORDER)]
        self._cols[idx]=nxt; self.lb.itemconfig(idx,fg=nxt)

    def _export_kml(self):
        if not self._pts:
            messagebox.showwarning('Нет точек','Сначала сгенерируйте точки'); return
        p=filedialog.asksaveasfilename(defaultextension='.kml',filetypes=[('KML','*.kml')])
        if not p: return
        try:
            with open(p,'w',encoding='utf-8') as f:
                f.write(build_kml(os.path.splitext(os.path.basename(p))[0],
                                  self._pts,self._cols,
                                  self.pref_var.get(),self.start_var.get()))
            messagebox.showinfo('KML сохранён',p)
        except Exception as e: messagebox.showerror('Ошибка',str(e))

    # ---------- quit ----------
    def _quit(self): self.destroy(); sys.exit(0)

# -------------------------------------------------------------
if __name__ == '__main__':
    App().mainloop()
