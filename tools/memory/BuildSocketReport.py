# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Build a Chinese, self-contained HTML report from retained measurements.

No external assets or inferred peak values are used. Figures show medians of
at least three synchronized, placement-verified, fault-free observations.
Scheduler-interrupted observations retain their full elapsed envelopes.
"""
import argparse
import base64
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':FONT.get_name(),'axes.unicode_minus':False,
                     'figure.dpi':150,'savefig.dpi':160,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
                     'axes.titlepad':14,'figure.facecolor':'white'})
# Register the installed CJK collection explicitly on systems without font cache.
matplotlib.font_manager.fontManager.addfont(FONT.get_file())
OPS={'read':'读','write':'写','copy':'拷贝','triad':'Triad'}
COLORS=['#147d92','#d48035','#6b63ad','#438b55','#bd5074','#60748c']
KERNELS=['scalar16','scalar32','scalar64','vec1','vec2','vec4']


def esc(v):return html.escape(str(v))
def number(v,d=2):return '—' if v is None else f'{v:,.{d}f}'
def mib(v):return v/1024**2


class Report:
    def __init__(self,root):
        self.root=root
        self.data=json.loads((root/'analysis.json').read_text())
        self.out=root/'report';self.out.mkdir(exist_ok=True)
        self.sections=[];self.images=[]
        self.raw_rows=self.data['summaries']
        self.rows=self.data.get('selected_summaries',self.raw_rows)
        self.main=[r for r in self.rows if r['scope'] in ('total','chase:g0','round-trip')]
        self.cases={(c['run'],c['id']):c for c in self.data['cases']}
        supplement=root/'interleave-20261030'/'analysis.json'
        self.interleave_data=json.loads(supplement.read_text()) if supplement.exists() else None

    def interleave_get(self,die,memory,pages,placement,op,access,n=37):
        if not self.interleave_data:return None
        name=f'interleave_die_{die}_{memory}_{pages}_{placement}'
        return next((r for r in self.interleave_data['selected_summaries']
                     if r['case_name']==name and r['scope']=='total' and r['operation']==op
                     and r['access']==access and r['requested_threads_per_group']==n),None)

    def rows_for(self,**conditions):
        rows=[r for r in self.main if all(r.get(k)==v for k,v in conditions.items())]
        # Reruns are specified before results are inspected; never choose by speed.
        chosen={}
        for r in rows:
            signature=tuple(str(r.get(k)) for k in ('case_id','scope'))
            rank=(r['run'].startswith('repair'),r['run']=='scaling',r['run']=='full',r['run'])
            if signature not in chosen or rank>chosen[signature][0]:chosen[signature]=(rank,r)
        return [v[1] for v in chosen.values()]

    def get(self,**kw):
        rows=self.rows_for(**kw)
        rows.sort(key=lambda r:(r['run'].startswith('repair'),r['run']=='scaling',r['run']=='full'),reverse=True)
        return rows[0] if rows else None

    def val(self,r):return r['median'] if r and r['samples_usable']>=3 else None

    def metric(self,r):
        if r is None:return '未执行'
        v=self.val(r)
        if v is None:return f'<span class="muted">样本不足 ({r["samples_usable"]}/{r["samples_recorded"]})</span>'
        policy=f'<br>默认绑定 {esc(r["default_memory_bind"])}' if r.get('default_memory_bind') else ''
        pooling=f'；合并 {len(r["pooled_runs"])} 轮' if len(r.get('pooled_runs',[]))>1 else ''
        return (f'<b>{number(v)}</b><small>{number(r["minimum"])}–{number(r["maximum"])}<br>'
                f'n={r["samples_usable"]}，严格={r["samples_strict"]}；{number(r.get("target_seconds",.15),2)} s{policy}{pooling}</small>')

    def table(self,headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{esc(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

    def section(self,title,content):self.sections.append((title,content))

    def figure(self,name,fig,caption):
        fig.tight_layout(pad=1.5)
        path=self.out/(name+'.png');fig.savefig(path,bbox_inches='tight');plt.close(fig)
        self.images.append(path)
        encoded=base64.b64encode(path.read_bytes()).decode()
        return f'<figure><img alt="{esc(caption)}" src="data:image/png;base64,{encoded}"><figcaption>{esc(caption)}</figcaption></figure>'

    def core_scaling(self):
        content='<p>计算 NUMA 12（die B），CPU 偏移按 4–37、0–3 的固定顺序取前 N 个，CPU 数是实际并行工作线程数。所有曲线每个线程持有私有数组；读/写一份、拷贝两份、Triad 三份。</p>'
        counts=[1,2,4,8,15,24,32,36,37,38]
        for access in ('temporal','stream'):
            fig,axes=plt.subplots(2,2,figsize=(12,8))
            for ax,(op,label) in zip(axes.flat,OPS.items()):
                for i,(name,title) in enumerate((('scaling_ddr','DDR，4 MiB/数组/核'),('scaling_hbm','HBM，4 MiB/数组/核'),('scaling_hbm_huge','HBM，16 MiB/数组/核，2 MiB 页'))):
                    ys=[self.val(self.get(family='scaling',case_name=name,operation=op,access=access,requested_threads_per_group=n)) for n in counts]
                    ax.plot(counts,[np.nan if y is None else y for y in ys],'-o',ms=4,label=title,color=COLORS[i])
                ax.set(title=label,xlabel='核心数',ylabel='GB/s（应用字节）');ax.set_xticks(counts);ax.tick_params(axis='x',labelrotation=90,labelsize=8)
            axes[0,0].legend(fontsize=8)
            content+=self.figure('scaling_'+access,fig,f'{access}：单 NUMA 1–38 核扩展。缺点表示同步/缺页筛选后不足 3 个样本。HBM 两条曲线同时改变了页大小和数组大小，不能直接归因于页大小。')
        rows=[]
        for name,label in (('scaling_ddr','DDR / 4 MiB'),('scaling_hbm','HBM / 4 MiB'),('scaling_hbm_huge','HBM / 16 MiB / 2 MiB页')):
            for access in ('temporal','stream'):
                for op,opname in OPS.items():
                    points=[self.get(family='scaling',case_name=name,operation=op,access=access,requested_threads_per_group=n) for n in (36,37,38)]
                    v37,v38=map(self.val,points[1:]);delta=(v38/v37-1)*100 if v37 and v38 else None
                    rows.append([esc(label),access,opname]+[self.metric(r) for r in points]+[number(delta)+'%' if delta is not None else '—'])
        content+='<h3>36、37、38 核放大比较</h3><p>每格给出中位数、有效样本范围、有效样本数及完全无上下文切换的严格样本数。38 相对 37 的变化是本轮观测，不能据此单独证明系统抢占的因果关系。</p>'
        content+=self.table(['配置','访问策略','动作','36 核 GB/s','37 核 GB/s','38 核 GB/s','38/37 变化'],rows)
        detail=[]
        for name,label in (('scaling_ddr','DDR / 4 MiB'),('scaling_hbm','HBM / 4 MiB'),('scaling_hbm_huge','HBM / 16 MiB / 大页')):
            for access in ('temporal','stream'):
                for n in counts:
                    rr=[self.get(family='scaling',case_name=name,operation=op,access=access,requested_threads_per_group=n) for op in OPS]
                    detail.append([label,access,str(n)]+[self.metric(r) for r in rr])
        content+='<details><summary>全部 10 档核心数的精确结果</summary>'+self.table(['配置','策略','核心数']+list(OPS.values()),detail)+'</details>'
        knees=[]
        for name,label in (('scaling_ddr','DDR'),('scaling_hbm','HBM 基础页'),('scaling_hbm_huge','HBM 大页')):
            endpoint=self.val(self.get(family='scaling',case_name=name,operation='read',access='stream',requested_threads_per_group=38))
            if not endpoint:continue
            hits=[n for n in counts if (self.val(self.get(family='scaling',case_name=name,operation='read',access='stream',requested_threads_per_group=n)) or 0)>=endpoint*.95]
            if hits:knees.append(f'{label} 首次在 {hits[0]} 核达到本轮 38 核只读带宽的 95%')
        if knees:content+='<p>'+esc('；'.join(knees)+'。这是本轮采样点定义的扩展拐点，不能外推为任意内核的最优线程数。')+'</p>'
        self.section('单 NUMA 核心数扩展与满核效应',content)

    def aggregate(self):
        configs=[('single','ddr_b',37,'单 NUMA DDR'),('full_die_b','die_b_ddr_152',38,'die B DDR'),('available','available_socket_ddr',38,'socket DDR（可用核）'),
                 ('single','hbm_b',37,'单 NUMA HBM'),('full_die_b','die_b_hbm_152',38,'die B HBM'),('available','available_socket_hbm',38,'socket HBM（可用核）')]
        if self.rows_for(family='full'):
            configs[2]=('full','full_socket_ddr',38,'socket DDR（完整 304 核）')
            configs[5]=('full','full_socket_hbm',38,'socket HBM（完整 304 核）')
        content='<p>聚合值来自同一次同步测试的总字节数除以所有线程的共同时间包络，绝不是把独立测试的峰值相加。die A 的可用上限 148 核，整个 socket 为 300 核；die B 可以使用完整 152 核。</p>'
        for access in ('temporal','stream'):
            rows=[]
            for family,name,n,label in configs:
                records=[self.get(family=family,case_name=name,requested_threads_per_group=n,operation=op,access=access) for op in OPS]
                first=next((r for r in records if r),None)
                rows.append([esc(label),str(first['threads']) if first else '—']+[self.metric(r) for r in records])
            content+=f'<h3>{access}</h3>'+self.table(['规模','实际核数']+[x+' GB/s' for x in OPS.values()],rows)
        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        for ax,kind in zip(axes,('ddr','hbm')):
            for i,(op,label) in enumerate(OPS.items()):
                for access,style in (('temporal','-'),('stream','--')):
                    rr=[self.get(family='aggregate',case_name='socket_'+kind,requested_threads_per_group=n,operation=op,access=access) for n in (8,16,37)]
                    ax.plot([r['threads'] if r else n*8 for r,n in zip(rr,(8,16,37))],[self.val(r) if self.val(r) is not None else np.nan for r in rr],style+'o',label=label+' '+access,color=COLORS[i],ms=4)
            ax.set(title='8 NUMA 聚合 '+kind.upper(),xlabel='实际核心总数',ylabel='GB/s（应用字节）');ax.legend(fontsize=7,ncol=2)
        content+=self.figure('aggregate',fig,'全 socket DDR 与 HBM 的扩展；36/37/38 核单 NUMA 结果另见上一节。')
        rows=[]
        for name,label in (('available_die_a_ddr','die A DDR'),('available_die_a_hbm','die A HBM'),('available_socket_ddr','socket DDR'),('available_socket_hbm','socket HBM')):
            for access in ('temporal','stream'):
                records=[self.get(family='available',case_name=name,operation=op,access=access) for op in OPS]
                first=next((r for r in records if r),None)
                rows.append([label,access,str(first['threads']) if first else '—']+[self.metric(r) for r in records])
        content+='<h3>避开占用核心后的上限配置</h3>'+self.table(['规模','策略','实际核数']+[x+' GB/s' for x in OPS.values()],rows)
        full=self.rows_for(family='full')
        if full:
            content+='<h3>后补的完整 152 / 304 核测试</h3>'
            rows=[]
            for name in ('full_die_a_ddr','full_die_a_hbm','full_socket_ddr','full_socket_hbm'):
                for access in ('temporal','stream'):
                    rr=[self.get(family='full',case_name=name,operation=op,access=access) for op in OPS]
                    first=next((r for r in rr if r),None)
                    rows.append([name,access,str(first['threads']) if first else '—']+[self.metric(r) for r in rr])
            content+=self.table(['配置','策略','实际核数']+list(OPS.values()),rows)
        else:content+='<p class="note">完整 die A 152 核 / socket 304 核尚未形成结果；主体测试期间保留 CPU 306–309。300 核结果不能标为 304 核结果。</p>'
        self.section('DDR 控制器共享与 HBM 聚合能力',content)

    def routes(self):
        names=[('ddr_b','本地 DDR，12→12'),('hbm_b','本地 HBM，12→28'),('peer_hbm_b','同 die HBM，12→29'),('cross_ddr_ba','跨 die DDR，12→8'),('cross_hbm_ba','跨 die HBM，12→24'),
               ('ddr_a','本地 DDR，8→8'),('hbm_a','本地 HBM，8→24'),('peer_hbm_a','同 die HBM，8→25'),('cross_ddr_ab','跨 die DDR，8→12'),('cross_hbm_ab','跨 die HBM，8→28')]
        content='<p>箭头表示「发起计算 NUMA → 内存所在 NUMA」。本轮路径带宽的所有输入和输出数组均放在箭头目标内存；copy/triad 因而衡量远程读写总应用字节，不等同于只把本地数组拷贝到远端的单向搬运。</p>'
        for n in (1,16,37):
            rows=[]
            for name,label in names:
                rr=[self.get(family='single',case_name=name,requested_threads_per_group=n,operation=op,access='stream') for op in OPS]
                first=next((r for r in rr if r),None)
                rows.append([label,str(first['threads']) if first else '—']+[self.metric(r) for r in rr])
            content+=f'<h3>单发起 NUMA，{n} 核请求，stream</h3>'+self.table(['路径','实际核数']+list(OPS.values()),rows)
        rows=[]
        aggregates=[('die_a_peer_hbm','die A 内四路轮转 HBM'),('die_b_peer_hbm','die B 内四路轮转 HBM'),('socket_peer_hbm','两个 die 各四路轮转 HBM'),('die_cross_ddr_ab','die A→B DDR'),('die_cross_ddr_ba','die B→A DDR'),('die_cross_hbm_ab','die A→B HBM'),('die_cross_hbm_ba','die B→A HBM'),('socket_cross_ddr','双向跨 die DDR'),('socket_cross_hbm','双向跨 die HBM')]
        for name,label in aggregates:
            for n in (16,37):
                rr=[self.get(family='aggregate',case_name=name,requested_threads_per_group=n,operation=op,access='stream') for op in OPS]
                first=next((r for r in rr if r),None)
                rows.append([label,str(n),str(first['threads']) if first else '—']+[self.metric(r) for r in rr])
        content+='<h3>四路 / 八路同步访问，stream</h3>'+self.table(['路径','请求核数/NUMA','实际总核数']+list(OPS.values()),rows)
        content+='<p>同 die HBM 轮转：8→25、9→26、10→27、11→24；另一 die 为 12→29、13→30、14→31、15→28。跨 die 四路为 8–11 分别访问 12–15 的 DDR 或 28–31 的 HBM；反向同理。双向项目同时运行两个方向，表中是双向合计。</p>'
        self.section('同 die 与跨 die 访问路径',content)

    def interleaved_placement(self):
        if not self.interleave_data:return
        data=self.interleave_data
        content='<p>2026-10-30 补充。socket 1 的 die 0/A 为计算 NUMA 8–11、HBM24–27；die 1/B 为计算 NUMA12–15、HBM28–31。两个 die 分别运行，每次四个 NUMA 同时访问。每组拥有独立数组和工作线程，交织只改变物理页面放置，不共享数组。</p>'
        content+='<p>三种布局：本地绑定是各组数组完全放在对应内存节点；环形绑定是各组数组完全放在下一个 HBM 节点；页交织是<strong>每个私有数组</strong>都按四节点顺序循环放置，例如 [28,29,30,31,28,29,30,31,…]。所有数组从同一节点开始，测试前后验证每页的精确顺序，而非只检查节点集合。</p>'
        content+='<p>本轮固定使用 Vec4，FP32，16 / 37 核每 NUMA（合计 64 / 148 核），temporal / stream，读、写、Copy、Triad。CPU306 上有轻量监控进程，所有案例排除该核。不同放置使用同一组实际 CPU、同一可执行文件、相同计时窗口及工作集；本地和环形对照均重新测量，不与 10 月 29 日的旧结果混合计算比值。</p>'
        content+='<p>主矩阵使用 5×0.5 秒；短窗口样本不足的 37 核/NUMA DDR 写入，四组 die/访问策略对照均将本地绑定和交织一起改为 5×20 秒重测，仍保持 0.1% 起止偏差阈值。表中采用各自最后预定条件；不同窗口不合并。</p>'
        content+='<p class="note">DDR 与基础页 HBM 均为 4 MiB/数组/核，交织单元为 OS 基础页；HBM 大页为 16 MiB/数组/核，以显式 2 MiB 页交织。大页与基础页两组同时改变数组大小，不能把两组差异全部归因于页大小。DDR 节点无空闲 hugetlb 池，未做 DDR 大页测试。每组 CPU 的成绩代表该组发起吞吐，不能当作单个存储 NUMA 的带宽。</p>'
        labels={'local':'本地绑定','ring':'环形绑定','interleave':'四节点页交织'}
        def table_rows(die,n):
            rows=[]
            for memory,pages,title in [('ddr','base','DDR / 基础页 / 4 MiB'),('hbm','base','HBM / 基础页 / 4 MiB'),('hbm','huge','HBM / 2 MiB 页 / 16 MiB')]:
                for access in ('temporal','stream'):
                    modes=['local','interleave']+(['ring'] if memory=='hbm' and pages=='base' else [])
                    for placement in modes:
                        records=[self.interleave_get(die,memory,pages,placement,op,access,n) for op in OPS]
                        first=next((r for r in records if r),None)
                        rows.append([title,access,labels[placement],str(first['threads']) if first else '—']+[self.metric(r) for r in records])
            return rows
        headers=['内存 / 页 / 每数组每核','策略','布局','实际核数']+[v+' GB/s' for v in OPS.values()]
        for die,title in [('a','die 0 / A：NUMA8–11'),('b','die 1 / B：NUMA12–15')]:
            content+='<h3>'+title+'，每 NUMA 37 核</h3>'+self.table(headers,table_rows(die,37))
            content+='<details><summary>'+title+'，每 NUMA 16 核的完整对照</summary>'+self.table(headers,table_rows(die,16))+'</details>'
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        configurations=[(die,memory,pages) for die in ('a','b') for memory,pages in [('ddr','base'),('hbm','base'),('hbm','huge')]]
        ticks=[f'die {die.upper()}\n'+('DDR' if memory=='ddr' else 'HBM 大页' if pages=='huge' else 'HBM 基础页') for die,memory,pages in configurations]
        x=np.arange(len(configurations))
        for ax,(op,label) in zip(axes.flat,OPS.items()):
            for i,placement in enumerate(('local','interleave','ring')):
                values=[self.val(self.interleave_get(die,memory,pages,placement,op,'stream')) for die,memory,pages in configurations]
                ax.bar(x+(i-1)*.24,[np.nan if v is None else v for v in values],width=.23,label=labels[placement],color=COLORS[i])
            ax.set(title=label,ylabel='GB/s',ylim=(0,None));ax.set_xticks(x,ticks,fontsize=8)
        handles,legend_labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.5,-.035),ncol=3,fontsize=9)
        content+=self.figure('interleave_comparison',fig,'同轮次对照：每 NUMA 37 核，四 NUMA 同步聚合，stream；未画出的环形布局表示该内存/页配置未安排环形测试。')
        ratios=[]
        for die in ('a','b'):
            for memory,pages,title in [('ddr','base','DDR 基础页'),('hbm','base','HBM 基础页'),('hbm','huge','HBM 大页')]:
                for op,label in OPS.items():
                    local=self.interleave_get(die,memory,pages,'local',op,'stream')
                    mixed=self.interleave_get(die,memory,pages,'interleave',op,'stream')
                    ring=self.interleave_get(die,memory,pages,'ring',op,'stream')
                    a,b,c=map(self.val,(local,mixed,ring))
                    ratios.append([f'die {die.upper()}',title,label,number(b/a*100,1)+'%' if a and b else '—',number(b/c*100,1)+'%' if b and c else '—'])
        content+='<h3>同轮次性能比例，stream，148 核</h3>'+self.table(['die','内存','动作','页交织 / 本地绑定','页交织 / 环形绑定'],ratios)
        content+='<p>本节新增的是带宽对照；后续延迟矩阵仍为原有独立绑定布局，不能解释为交织延迟。</p>'
        content+='<p>页交织令每组约四分之一页面位于本地、四分之三位于同 die 其他 NUMA，顺序扫描的应用字节比例在完整周期时同样如此；这不等于测得了各物理链路流量。页面交织没有增加 DDR 控制器数量，也不消除共享互连的竞争。是否受益应按动作和上述同轮对照判断。交织同时引入本地页面和多个远端目标，结果是这些因素的综合作用；不能用单组带宽的简单加权预测聚合结果，也不能据此定位某条物理互连的极限。</p>'
        total=[r for r in data['selected_summaries'] if r['scope']=='total']
        poor=[r for r in total if self.val(r) is None]
        content+=f'<p>独立补充数据集共有 {len(total)} 个聚合配置，{len(total)-len(poor)} 个满足至少 3 个有效观察；全部沿用主体报告的逐页校验、零计时缺页、0.1% 起止偏差和 99.8% 共同窗口规则。上下文切换样本保留完整耗时，未择优取峰值。各格同时列出样本范围、有效数、严格数和实际窗口。</p>'
        content+='<p>补充工具源码 SHA-256：<code>'+esc(data['metadata']['tool_source_sha256'])+'</code>。与主体工具源码不同，两个数据集分别汇总，绝不合并为同一重复测量。</p>'
        content+='<p>补充原始数据位于 <code>interleave-20261030/main</code>，包含每组/每线程页面数、CPU 映射、命令、日志及逐次拒绝原因；独立的 <code>interleave-20261030/selected-summary.csv</code> 和 <code>analysis.json</code> 保留全部统计依据。补测清单 <code>repair-01-short-input.json</code>、<code>repair-02-long-input.json</code> 记录实际案例和窗口；复现最终统计时应在独立目录分别重放这些清单，案例内的 seconds 字段优先于命令行默认窗口。</p>'
        content+='<pre>'+esc('source ~/.bashrc\ncd /home/ryz/vecops-neo-memory-interleave-20261030\ncmake -S . -B build/memory-interleave -DCMAKE_BUILD_TYPE=Release \\\n  -DCMAKE_C_COMPILER="$(command -v clang)" \\\n  -DCMAKE_CXX_COMPILER="$(command -v clang++)" \\\n  -DVECOPS_BUILD_MEMORY_TOOLS=ON -DVECOPS_BUILD_MEMORY=OFF \\\n  -DVECOPS_BUILD_TESTING=OFF -DVECOPS_BUILD_BENCHMARKS=OFF \\\n  -DVECOPS_BUILD_PYTHON=OFF -DVECOPS_BUILD_COMPILER=OFF \\\n  -DVECOPS_BUILD_NUMA_LOADER=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON\ncmake --build build/memory-interleave --target vecops-memory-bench -j8\npython3 tools/memory/CheckAssembly.py build/memory-interleave\npython3 tools/memory/RunSocketCampaign.py \\\n  --binary build/memory-interleave/tools/memory/vecops-memory-bench \\\n  --replay-manifest build/memory-bench/campaign-20261029/interleave-20261030/input.json \\\n  --output build/memory-bench/campaign-20261029/interleave-20261030/replay \\\n  --samples 5 --seconds .5 --retries 0 --exclude-cpus 306 \\\n  --default-memory-bind 8-15,24-31')+ '</pre>'
        self.section('die 内四 NUMA 页交织：新增实测与同轮对照',content)

    def kernels_and_caches(self):
        content='<p>scalar16/32/64 每次循环处理 16/32/64 B；vec1/2/4 在本机 SVE 64 B 向量长度下分别处理 64/128/256 B。标量使用明确的标量指令，禁用自动向量化与自动展开；Vec 内核经过指令审计。性能包含循环、函数分块、计时轮询以及实际编译结果的开销。</p>'
        for family,name,title,n,access in [('cache_kernels','cache_l1','L1D（每数组 4 KiB）',1,'temporal'),('cache_kernels','cache_l2','L2（每数组 128 KiB）',1,'temporal'),('kernels','kernel_ddr','DDR（每数组 4 MiB）',1,'stream'),('kernels','kernel_hbm','HBM（每数组 4 MiB）',1,'stream'),('kernels','kernel_ddr','DDR，32 核',32,'stream'),('kernels','kernel_hbm','HBM，32 核',32,'stream')]:
            rows=[]
            for k in KERNELS:
                rr=[self.get(family=family,case_name=name,kernel=k,requested_threads_per_group=n,operation=op,access=access) for op in OPS]
                rows.append([k]+[self.metric(r) for r in rr])
            content+=f'<h3>{esc(title)} · {access}</h3>'+self.table(['循环工作量']+list(OPS.values()),rows)
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for op,label in OPS.items():
            rr=sorted(self.rows_for(family='cache_curve',operation=op),key=lambda r:r['working_set_bytes'])
            axes[0].plot([r['working_set_bytes']/1024 for r in rr],[self.val(r) if self.val(r) is not None else np.nan for r in rr],'-o',ms=3,label=label)
        rr=sorted(self.rows_for(family='cache_latency'),key=lambda r:r['working_set_bytes'])
        axes[1].plot([r['working_set_bytes']/1024 for r in rr],[self.val(r) if self.val(r) is not None else np.nan for r in rr],'-o',color=COLORS[0],ms=4)
        for ax in axes:
            ax.set_xscale('log',base=2);ax.set_xlabel('每线程全部活动数组之和（KiB）')
            for size,label in ((32,'L1D 32 KiB'),(768,'L2 768 KiB')):ax.axvline(size,ls='--',alpha=.5,label=label)
        axes[0].set(title='单核 Vec4 带宽与工作集',ylabel='GB/s（应用字节）');axes[0].legend(fontsize=7)
        axes[1].set(title='随机依赖链延迟与工作集',ylabel='ns / dependent load');axes[1].legend(fontsize=7)
        content+=self.figure('cache_curve',fig,'容量坐标使用全部活动数组之和：Triad 为每数组容量的三倍，copy 为两倍。依赖链为 64 B 间距的随机单环。')
        rows=[]
        for cache,label in (('l1','L1D'),('l2','L2')):
            for scale in ('numa','die','socket'):
                rr=[self.get(family='cache_aggregate',case_name=f'cache_{cache}_{scale}',operation=op) for op in OPS]
                first=next((r for r in rr if r),None)
                rows.append([label,scale,str(first['threads']) if first else '—']+[self.metric(r) for r in rr])
        content+='<h3>私有缓存的同步聚合</h3>'+self.table(['层级','规模','实际核数']+list(OPS.values()),rows)
        controls=[]
        for cache,label in (('l1','L1D'),('l2','L2')):
            for scale in ('single','numa','die','socket'):
                for placement,title in (('ddr_base','DDR / 基础页'),('hbm_base','HBM / 基础页'),('hbm_huge','HBM / 2 MiB 页')):
                    rr=[self.get(family='cache_control',case_name=f'cache_control_{cache}_{placement}_{scale}',operation=op) for op in OPS]
                    first=next((r for r in rr if r),None)
                    controls.append([label,scale,title,str(first['threads']) if first else '—']+[self.metric(r) for r in rr])
        content+='<details><summary>缓存工作集的内存归属与页大小对照</summary><p>固定相同活动数组大小，比较 DDR 基础页、HBM 基础页与 HBM 大页。缓存驻留的活动区仍为每数组 4 KiB / 128 KiB；大页会按 2 MiB 粒度保留映射，未把映射填充部分计入工作集或数据字节。此对照用于观察地址翻译、物理分配与内存归属的综合影响。</p>'+self.table(['层级','规模','分配方式','实际核数']+list(OPS.values()),controls)+'</details>'
        content+='<p>这些是各核心私有缓存并行执行同一内核的应用带宽，不是共享缓存或片上互联总线带宽。Triad 包含浮点运算；结果不应直接解释为硬件 load/store 端口理论峰值。</p>'
        if self.rows_for(family='cache_final'):
            final=[]
            for cache,label in (('l1','L1D / 每数组 4 KiB'),('l2','L2 工作集 / 每数组 64 KiB')):
                for scale,title in (('numa_1','单核'),('numa_38','单 NUMA'),('die_a','die A 152 核'),('die_b','die B 152 核'),('socket','socket 304 核')):
                    rr=[self.get(family='cache_final',case_name=f'cache_final_{cache}_{scale}',operation=op) for op in OPS]
                    first=next((r for r in rr if r),None)
                    actual=str(first['threads']) if first else '—'
                    title={'die_a':'die A','die_b':'die B','socket':'socket'}.get(scale,title)+'（'+actual+' 核）'
                    final.append([label,title]+[self.metric(r) for r in rr])
            content+='<h3 id="cache-final">保守工作集与实际可用核数的最终缓存汇总</h3><p>每线程各数组为 4 KiB / 64 KiB，Triad 全部活动区分别为 12 KiB / 192 KiB。使用本地 HBM 的 2 MiB 页，进程默认绑定 socket 1 内存节点 8–15、24–31；晚期新增监控任务占用 CPU 380，因此 die A / socket 聚合避开该核，分别使用 151 / 303 核，die B 使用完整 152 核。L2 采用更小工作集为容量边缘和分配冲突留出更大余量；仍按实际内核吞吐报告。</p>'+self.table(['工作集','规模']+list(OPS.values()),final)
            fig,axes=plt.subplots(1,2,figsize=(12,4.5))
            counts=[1,2,4,8,15,24,32,36,37,38]
            for ax,cache,title in zip(axes,('l1','l2'),('L1D 工作集','L2 工作集')):
                for op,label in OPS.items():
                    rr=[self.get(family='cache_final',case_name=f'cache_final_{cache}_numa_{n}',operation=op) for n in counts]
                    ax.plot(counts,[self.val(r) if self.val(r) is not None else np.nan for r in rr],'-o',ms=3,label=label)
                ax.set(title=title,xlabel='单 NUMA 核心数',ylabel='GB/s（应用字节）');ax.set_xticks(counts);ax.tick_params(axis='x',labelrotation=90,labelsize=8);ax.legend(fontsize=8)
            content+=self.figure('cache_core_scaling',fig,'L1D / L2 工作集同样覆盖单 NUMA 的十档核心数；全部操作为 Vec4 temporal。')
        for folder,title in (('pmu','128 KiB/数组，原默认策略'),('pmu64-socket-bind','64 KiB/数组，默认 socket 1 绑定')):
            path=self.root/folder/'results.json'
            if not path.exists():continue
            records=json.loads(path.read_text())['records'];events=[]
            for record in records:
                counts={};coverage=[]
                with (path.parent/record['perf_csv']).open() as stream:
                    for parts in csv.reader(line for line in stream if line.strip() and not line.startswith('#')):
                        try:counts[parts[2]]=float(parts[0]);coverage.append(float(parts[4]))
                        except (ValueError,IndexError):continue
                access=counts.get('armv8_pmuv3_0/l2d_cache/u');refill=counts.get('armv8_pmuv3_0/l2d_cache_refill/u')
                events.append([record['placement'],str(record['trial']),number(access/1e9) if access else '—',number(refill/1e9) if refill else '—',number(refill/access*100,3)+'%' if access and refill is not None else '—',number(min(coverage),1)+'%' if coverage else '—'])
            content+='<details><summary>硬件计数器交叉检查 · '+title+'</summary><p>148 核、四 NUMA、Triad temporal，每种分配方式三次独立进程。计数覆盖整个程序，包含初始化、预热、所有样本和校验，未按计时窗口门控。下表是两类 PMU 事件的计数比值，不能直接视为校准后的未命中率，也不能换算成 DDR/HBM 物理流量。容量内工作集不自动证明 100% L2 命中；主文因此报告指定工作集下的吞吐。</p>'+self.table(['分配方式','进程轮次','L2 访问事件 /10⁹','L2 补充事件 /10⁹','补充/访问事件','计数覆盖'],events)+'</details>'
        c1=self.val(self.get(family='cache_latency',requested_size='4KiB'))
        c2=self.val(self.get(family='cache_latency',requested_size='128KiB'))
        b1=self.val(self.get(family='cache_kernels',case_name='cache_l1',kernel='vec4',operation='read'))
        b2=self.val(self.get(family='cache_kernels',case_name='cache_l2',kernel='vec4',operation='read'))
        if all((c1,c2,b1,b2)):
            content+=f'<p>本轮单核 Vec4 顺序读取在 L1D / L2 工作集下分别为 {number(b1)} / {number(b2)} GB/s，而 4 KiB / 128 KiB 随机依赖链为 {number(c1)} / {number(c2)} ns。顺序吞吐接近与依赖延迟分层同时成立，不能仅凭顺序带宽判断缓存层级是否命中。</p>'
        self.section('标量 / Vec 循环工作量与 L1D、L2',content)

    def transfers(self):
        content='<p>本节使用 source:destination 路由：copy 的输入数组在 source、输出在 destination；Triad 的两个输入都在 source、输出在 destination。计算线程均在 die B，单路固定 NUMA 12，四路为 12–15。与上一节“所有数组均在远端”的测试不同，本节直接测量不同内存之间的搬运。</p>'
        rows=[]
        labels={'ddr_to_hbm':'DDR 12→HBM 28','hbm_to_ddr':'HBM 28→DDR 12',
                'peer_hbm_forward':'HBM 28→29','peer_hbm_reverse':'HBM 29→28',
                'cross_ddr_forward':'DDR 12→8','cross_ddr_reverse':'DDR 8→12',
                'cross_hbm_forward':'HBM 28→24','cross_hbm_reverse':'HBM 24→28',
                'die_ddr_to_hbm':'四路 DDR→本地 HBM','die_hbm_to_ddr':'四路 HBM→本地 DDR',
                'die_peer_hbm':'四路同 die HBM 轮转','die_cross_ddr':'四路跨 die DDR，B→A',
                'die_cross_hbm':'四路跨 die HBM，B→A'}
        for name,label in labels.items():
            for n in ((16,37) if name.startswith('die_') else (1,16,37)):
                for access in ('temporal','stream'):
                    rr=[self.get(family='transfer',case_name=name,requested_threads_per_group=n,operation=op,access=access) for op in ('copy','triad')]
                    first=next((r for r in rr if r),None)
                    rows.append([label,str(first['threads']) if first else '—',access]+[self.metric(r) for r in rr]+[number(self.val(rr[0])/2) if self.val(rr[0]) else '—'])
        content+=self.table(['源→目的','实际总核数','策略','copy 总 GB/s','Triad 总 GB/s','copy 有效搬运 GB/s'],rows)
        content+='<p>copy 总带宽 = 源读取 + 目标写入；有效搬运带宽 = 总带宽 / 2。Triad 同时包含计算和两个输入流，不把其总字节带宽解释为单向互联线速。</p>'
        self.section('源与目标分置的实际数据搬运',content)

    def latency(self):
        content='<p>空载延迟通过随机单环指针追逐测量，每次加载依赖前次地址，间距 64 B，不能依靠多个独立请求掩盖延迟。矩阵为 16 MiB 工作集，超出私有 L2；数值包含地址翻译及循环开销。每轮先计算整段时间的平均 ns/依赖加载，再对各轮均值取中位数；它不是单次访问延迟的分位数。HBM 大页矩阵与基础页矩阵工作集相同。</p>'
        groups=[]
        for family,kind,label in (('latency_matrix','ddr','DDR / 4 KiB 页'),('latency_matrix','hbm','HBM / 4 KiB 页'),('latency_huge','hbm','HBM / 2 MiB 页')):
            cells={'local':[],'peer':[],'cross':[]}
            for row in self.rows_for(family=family):
                target=row['targets'][0]
                if (target>=24)!=(kind=='hbm'):continue
                node=target-16 if kind=='hbm' else target
                execution=row['execution'][0]
                key='local' if node==execution else 'peer' if (node<12)==(execution<12) else 'cross'
                value=self.val(row)
                if value is not None:cells[key].append(value)
            def span(values):return f'{number(min(values),1)}–{number(max(values),1)} (单元 n={len(values)})' if values else '—'
            groups.append([label]+[span(cells[k]) for k in ('local','peer','cross')])
        content+=self.table(['同一 16 MiB 工作集','本地 ns','同 die 邻居 ns','跨 die ns'],groups)
        content+='<p>本表范围是对应矩阵各单元中位数的最小–最大值，并非单个样本的波动范围。HBM 的优势首先体现在并行带宽；其依赖读取延迟是否更低，应按相同页大小与工作集的实测值判断。</p>'
        for family,title,targets in (('latency_matrix','4 KiB 基础页：DDR 与 HBM',list(range(8,16))+list(range(24,32))),('latency_huge','2 MiB 大页：HBM',list(range(24,32)))):
            matrix=np.full((8,len(targets)),np.nan)
            table=[]
            for i,execution in enumerate(range(8,16)):
                cells=[]
                for j,target in enumerate(targets):
                    r=self.get(family=family,execution=[execution],targets=[target]);v=self.val(r)
                    if v is not None:matrix[i,j]=v
                    cells.append(number(v,1))
                table.append([str(execution)]+cells)
            if np.isfinite(matrix).any():
                fig,ax=plt.subplots(figsize=(max(8,len(targets)*.7),5.4))
                im=ax.imshow(np.ma.masked_invalid(matrix),cmap='YlGnBu',aspect='auto');fig.colorbar(im,ax=ax,label='ns / dependent load')
                ax.set_xticks(range(len(targets)),[str(t) for t in targets]);ax.set_yticks(range(8),range(8,16));ax.set(xlabel='内存 NUMA',ylabel='计算 NUMA',title=title)
                for i in range(8):
                    for j in range(len(targets)):
                        if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>(np.nanmin(matrix)+np.nanmax(matrix))/2 else '#173645')
                content+=self.figure(family,fig,title+'，有效样本中位数。空白表示未执行或有效样本不足。')
            content+='<details><summary>精确数值表 · '+title+'</summary>'+self.table(['计算 NUMA']+list(map(str,targets)),table)+'</details>'
        rows=[]
        for name,label in (('same_numa','同 NUMA'),('same_die','同 die 不同 NUMA'),('cross_die','跨 die')):
            r=self.get(family='pingpong',case_name=name)
            rows.append([label,self.metric(r)])
        content+='<h3>缓存行所有权交接</h3>'+self.table(['两个线程位置','完整往返 RTT（ns）'],rows)+'<p>原子 ping-pong 测的是同一缓存行的完整往返，包含 acquire/release 原子操作与轮询开销，不报告 RTT/2 作为实测单向延迟。此项导出的缺页和切换计数仅覆盖发起线程；响应线程被系统抢占也可能增加 RTT，严格样本数不能证明响应线程完全无干扰。</p>'
        self.section('空载内存延迟与缓存行交接',content)

    def loaded(self):
        content='<p>指针追逐固定在 NUMA 12 偏移 0；背景线程按偏移 4 起选取，两者不共享核心。每线程每数组 4 MiB、4 KiB 基础页。表内带宽是同一次加载延迟测试的背景总带宽，不能与独立测试的峰值混用。</p>'
        rows=[]
        for name,label in (('local_ddr','本地 DDR'),('local_hbm','本地 HBM'),('peer_hbm','同 die HBM'),('cross_ddr','跨 die DDR'),('cross_hbm','跨 die HBM')):
            # Earlier main baselines used offset 4; only the follow-up offset-0 baseline is comparable.
            baselines=[r for r in self.rows_for(family='loaded_baseline',case_name='idle_'+name)
                       if self.cases[(r['run'],r['case_id'])].get('cpu_offset')==0]
            baseline=max(baselines,key=lambda r:(r['run'].startswith('repair'),r['run'])) if baselines else None
            for n in (4,16,32):
                for op in ('read','triad'):
                    r=self.get(family='loaded_latency',case_name='loaded_'+name,requested_threads_per_group=n,scope='chase:g0',background_operation=op)
                    bw=self.get(family='loaded_latency',case_name='loaded_'+name,requested_threads_per_group=n,scope='total',background_operation=op)
                    ratio=self.val(r)/self.val(baseline) if self.val(r) and self.val(baseline) else None
                    rows.append([label,str(n),OPS[op],self.metric(baseline),self.metric(r),number(ratio)+'×' if ratio else '—',self.metric(bw)])
        content+=self.table(['目标','背景核数','背景动作','空载 ns','有载 ns','延迟倍数','同时背景 GB/s'],rows)
        self.section('带宽负载下的延迟',content)

    def pages_and_balance(self):
        content='<p>HBM 的 4 GiB 物理内存中，约 3 GiB 预留为 2 MiB hugetlb 池；普通页可用池约 1 GiB。不同页池不能只看普通 MemFree 判断可分配容量。所有大页测试直接使用预留池，不修改系统配置。</p>'
        rows=[]
        for n in (1,8,15):
            for op in ('read','triad'):
                for access in ('temporal','stream'):
                    base=self.get(family='page_control',requested_threads_per_group=n,operation=op,access=access,pages='base')
                    huge=self.get(family='page_control',requested_threads_per_group=n,operation=op,access=access,pages='huge')
                    ratio=self.val(huge)/self.val(base) if self.val(huge) and self.val(base) else None
                    rows.append([str(n),OPS[op],access,self.metric(base),self.metric(huge),number(ratio)+'×' if ratio else '—'])
        content+='<h3>固定 16 MiB/数组/核的页大小对照</h3>'+self.table(['核心数','动作','策略','4 KiB 页 GB/s','2 MiB 页 GB/s','大页/基础页'],rows)
        rows=[]
        for name,label in (('huge_hbm_single','单 NUMA 本地 HBM'),('huge_hbm_die','die B 四路本地 HBM'),('huge_hbm_socket','socket 八路本地 HBM'),('huge_hbm_cross','die A→B 四路 HBM')):
            for n in (1,16,37):
                rr=[self.get(family='huge_hbm',case_name=name,requested_threads_per_group=n,operation=op,access='stream') for op in OPS]
                first=next((r for r in rr if r),None)
                rows.append([label,str(n),str(first['threads']) if first else '—']+[self.metric(r) for r in rr])
        content+='<h3>HBM 大工作集，2 MiB 页，stream</h3>'+self.table(['规模','请求核数/NUMA','实际总核数']+list(OPS.values()),rows)
        rows=[]
        for node in range(8,16):
            rr=[self.get(family='balance',case_name=f'balance_{kind}_{node}',access=access) for kind,access in (('ddr','temporal'),('ddr','stream'),('hbm','temporal'),('hbm','stream'))]
            rows.append([str(node)]+[self.metric(r) for r in rr])
        content+='<h3>8 个 NUMA 的一致性检查：16 核只读，独立串行测量</h3>'+self.table(['计算 NUMA','DDR temporal','DDR stream','HBM temporal','HBM stream'],rows)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,op in zip(axes,('read','triad')):
            for name,label in (('size_ddr','DDR'),('size_hbm','HBM')):
                rr=sorted(self.rows_for(family='working_set',case_name=name,operation=op),key=lambda r:r['per_thread_array_bytes'])
                ax.plot([mib(r['per_thread_array_bytes']) for r in rr],[self.val(r) if self.val(r) is not None else np.nan for r in rr],'-o',label=label)
            ax.set_xscale('log',base=2);ax.set(title=OPS[op]+' · 单核 temporal',xlabel='每数组容量 MiB',ylabel='GB/s');ax.legend()
        content+=self.figure('working_set',fig,'2–64 MiB 每数组容量稳定性检查。单核带宽接近不代表 DDR 与 HBM 聚合能力相同。')
        self.section('页大小、工作集与 NUMA 一致性',content)

    def methodology(self):
        meta=self.data['metadata'];count=len(self.data['cases'])
        selected=self.rows_for();good=[r for r in selected if self.val(r) is not None];poor=[r for r in selected if self.val(r) is None]
        main=[r for r in self.raw_rows if r['scope'] in ('total','chase:g0','round-trip')]
        strict=sum(r['samples_strict'] for r in main);usable=sum(r['samples_usable'] for r in main);raw=sum(r['samples_recorded'] for r in main)
        content=f'<p>远程主机：<b>920f-4 / {esc(meta.get("host"))}</b>；项目：<code>/home/ryz/vecops-neo</code>；构建目录：<code>/home/ryz/vecops-neo/build/memory-bench</code>。实验日期 2026-10-29（UTC 时间戳保留在原始记录中），仅使用 socket 1，系统报告 2.0 GHz、performance governor，SVE 向量长度 64 B。</p>'
        content+=self.table(['对象','配置 / 口径'],[
            ['die A','计算 NUMA 8–11，CPU 304–455，共享一个 DDR 控制器；本地 HBM NUMA 24–27'],
            ['die B','计算 NUMA 12–15，CPU 456–607，共享一个 DDR 控制器；本地 HBM NUMA 28–31'],
            ['单 NUMA','38 核；HBM 4 GiB；DDR OS 节点约 63 GiB；每核 L1D 32 KiB / L2 768 KiB，缓存行 64 B'],
            ['物理 die 依据','两组 die 划分来自用户确认；系统 die_id 为 -1，未以此伪造拓扑。距离矩阵同 die 20、跨 die 30。'],
            ['机器占用','socket 0 有其他工作；主体测试排除其他测试任务使用的 CPU 306–309；该任务结束并复查负载后另行补测完整 304 核。所有核心均未设置系统级隔离，仍可能有系统线程。'],
            ['编译器',esc(meta.get('compiler'))],['编译选项','<code>'+esc(meta.get('build_flags'))+'</code>'],
            ['工具源码 SHA-256','<code>'+esc(meta.get('tool_source_sha256'))+'</code>'],
            ['校验','每个分配页在测试前后查询实际 NUMA 驻留；读写结果检查；标量/向量/NT 指令静态审计。'],
        ])
        content+='<p>本节配置数量、源码哈希及历史占用说明对应 10 月 29 日主体数据集；10 月 30 日交织补充的环境、数量、独立构建和复现命令见新增交织章节。</p>'
        content+='<h3>字节、时间和动作定义</h3><p>所有带宽用十进制 GB/s（10⁹ B/s）。read/write 每元素计 4 B，copy 计 8 B，Triad 为 a=b+αc，计 12 B。copy 的单向有效搬运量为表中总带宽的一半。它们均是应用数据字节，未用硬件计数器推算 DDR/HBM 线速，不计 RFO、写回或协议流量。</p>'
        content+='<p>temporal 使用普通缓存访问；stream 请求非时间局部加载和存储。本 ARM 构建已确认 LDNT1/STNT1 提示实际存在。NT 是硬件提示，并不保证完全绕过缓存。普通写的完成屏障也不等于把所有脏缓存行持久化到 DDR/HBM。</p>'
        content+='<p>每个 NUMA 独立分配私有数组、计算字节数与局部时间包络，再汇总所有组。线程固定 CPU，校准批次后预热 50 ms，主测试通常 5×150 ms；所有线程等待共同未来起点和截止时刻。计入完整批次，内存完成屏障计入耗时，线程汇合屏障位于计时之外。聚合分母从最早实际开始到最晚实际结束，防止线程先后启动造成虚高。补测的实际时间以 CSV 的 target_seconds 为准。</p>'
        content+='<h3>结果质量规则</h3><p>工具严格状态 ok 还要求计时期间没有任何上下文切换。多核共享系统下此规则常被触发，因此本报告同时展示两层质量：主统计只接受逐页驻留正确、无计时缺页/CPU 迁移、起止偏差各不超过目标时长 0.1%、共同工作窗口不少于目标时长 99.8% 的样本；发生上下文切换但仍满足这些条件的样本保留完整耗时并进入同一个中位数，不另选更快的“干净峰值”。“严格”列是其中工具状态 ok 且零切换的数量。</p>'
        distinct=len({c['id'] for c in self.data['cases']})
        content+=f'<p>已完成 {count:,} 次配置执行（含复测），涉及 {distinct:,} 个矩阵条目；主指标原始行 {raw:,}，符合本报告同步/缺页条件 {usable:,}，完全严格行 {strict:,}。按预定补测优先级汇总后，{len(good):,} 个主指标有至少 3 个有效样本，{len(poor):,} 个不足。不足者不画点、不给性能结论；所有拒绝行和线程明细仍在原始 CSV 中。</p>'
        content+='<p>部分复测仅给测试进程设置 <code>numactl --membind=8-15</code> 默认分配策略，以减少非测试数组区域的 NUMA 提示缺页；每个数据数组的显式 VMA 绑定仍优先，并在测试前后逐页验证。该控制没有修改全局 NUMA 平衡、频率或其他进程。绑定条件在表格和 CSV 中注明。任务策略与 VMA 策略的优先关系见 <a href="https://www.kernel.org/doc/html/v5.10/admin-guide/mm/numa_memory_policy.html">Linux 5.10 内存策略文档</a>；避免推断全部结束偏差都由同一机制导致。</p>'
        content+='<p>优先使用最后预定的实验条件；只有目标时长、默认策略、全部实际 CPU 映射、页大小、数组大小和内核均相同的重复进程才合并，所有符合相同质量标准的观察等权进入中位数。不同条件不混合，亦不选较快的一轮。CSV 的 pooled_runs / source_csvs 可追溯每条组成记录。后续 CPU 380 出现新任务，晚期缓存测量排除该核；两个 die A DDR 流式写配置按原约定待空闲后补充，不改变 CPU 映射来替代原配置。</p>'
        content+='<p>表格采用有效样本中位数；范围是最小值–最大值，不是置信区间。每个案例通常只有 5 次重复，接近的结果应结合波动看待。少数高核数短窗口写入出现约 10–17 ms 的结束偏差，原始样本按同一规则拒绝；后续复测保持 0.1% 阈值，使用 0.5 s 或 20 s 窗口，不推断其具体硬件或系统原因。没有随机化全套案例的执行顺序；本次结果代表共享主机该时间段的可复现观测。</p>'
        if poor:
            content+='<details><summary>有效样本不足的案例（保留，不填补或外推）</summary>'+self.table(['案例','动作','策略','核数','有效/原始','主要拒绝状态'],[[esc(r['case_name']),esc(r['operation']),esc(r['access']),str(r['threads']),f'{r["samples_usable"]}/{r["samples_recorded"]}',esc(r['rejected_statuses'])] for r in poor])+'</details>'
        self.section('测试环境、方法与可信范围',content)

    def conclusions(self):
        def r(family,name,op='read',access='stream',n=None):
            kw=dict(family=family,case_name=name,operation=op,access=access)
            if n is not None:kw['requested_threads_per_group']=n
            return self.get(**kw)
        one=r('single','ddr_b',n=37);die=r('full_die_b','die_b_ddr_152');socket=r('available','available_socket_ddr')
        h1=r('single','hbm_b',n=37);h4=r('full_die_b','die_b_hbm_152');h8=r('available','available_socket_hbm')
        if self.val(r('full','full_socket_ddr')):socket=r('full','full_socket_ddr')
        if self.val(r('full','full_socket_hbm')):h8=r('full','full_socket_hbm')
        facts=[]
        if all(self.val(x) for x in (one,die,socket)):
            facts.append(f'DDR 只读（stream）：单 NUMA 37 核 {number(self.val(one))} GB/s，die B 152 核 {number(self.val(die))} GB/s，socket {socket["threads"]} 核 {number(self.val(socket))} GB/s。四 NUMA 共享控制器，增加发起 NUMA 没有形成四倍 DDR 带宽；两个 die 合计约 {number(self.val(socket)/self.val(die))}×。')
        if all(self.val(x) for x in (h1,h4,h8)):
            facts.append(f'本地 HBM 只读（stream）：单 NUMA 37 核 {number(self.val(h1))} GB/s，die B 四路 {number(self.val(h4))} GB/s，socket 八路 {number(self.val(h8))} GB/s。后者为同步聚合实测，实际 {h8["threads"]} 核。')
        local=r('single','hbm_b',n=16);peer=r('single','peer_hbm_b',n=16);cross=r('single','cross_hbm_ba',n=16)
        if all(self.val(x) for x in (local,peer,cross)):
            facts.append(f'同为 16 核只读（stream），NUMA 12 访问本地 / 同 die 邻居 / 跨 die HBM 分别为 {number(self.val(local))} / {number(self.val(peer))} / {number(self.val(cross))} GB/s，位置选择对可用带宽影响明显。')
        # Present all operations rather than asserting a universal 38-core regression.
        comparisons=[]
        for name,label in (('scaling_ddr','DDR'),('scaling_hbm','HBM')):
            for op in OPS:
                a=r('scaling',name,op,n=37);b=r('scaling',name,op,n=38)
                if self.val(a) and self.val(b):comparisons.append(f'{label} {OPS[op]} {(self.val(b)/self.val(a)-1)*100:+.1f}%')
        if comparisons:facts.append('38 核相对 37 核（stream）的实测变化：'+'；'.join(comparisons)+'。不能把单个动作的变化推广为所有工作负载。')
        ddr=self.val(self.get(family='latency_matrix',execution=[12],targets=[12]))
        hbm=self.val(self.get(family='latency_matrix',execution=[12],targets=[28]))
        huge=self.val(self.get(family='latency_huge',execution=[12],targets=[28]))
        if all((ddr,hbm,huge)):
            facts.append(f'NUMA 12 的 16 MiB 随机依赖链：DDR / HBM 基础页分别 {number(ddr)} / {number(hbm)} ns，HBM 大页 {number(huge)} ns。高 HBM 聚合带宽不意味着在本轮相同基础页条件下具有更低依赖延迟。')
        cache1=self.get(family='cache_final',case_name='cache_final_l1_socket',operation='triad')
        cache2=self.get(family='cache_final',case_name='cache_final_l2_socket',operation='triad')
        if self.val(cache1) and self.val(cache2):
            facts.append(f'私有缓存工作集同步聚合：{cache2["threads"]} 核下 L1D / L2 的 Triad 为 {number(self.val(cache1)/1000)} / {number(self.val(cache2)/1000)} TB/s（应用字节）。这是各核私有工作集吞吐的汇总。')
        if self.interleave_data:
            pieces=[]
            for die in ('a','b'):
                for op in ('read','triad'):
                    local=self.val(self.interleave_get(die,'hbm','base','local',op,'stream'))
                    mixed=self.val(self.interleave_get(die,'hbm','base','interleave',op,'stream'))
                    if local and mixed:pieces.append(f'die {die.upper()} {OPS[op]} {number(mixed)} GB/s（本地绑定的 {mixed/local*100:.1f}%）')
            if pieces:facts.append('10 月 30 日新增四 NUMA 页交织，148 核、基础页 HBM、stream：'+'；'.join(pieces)+'。比例均与同轮重测的独立绑定比较；完整动作、DDR、大页和 64 核对照见新增章节。')
        content='<p class="lead">这份报告覆盖 socket 1 的 DDR、HBM、同 die / 跨 die 访问、私有缓存、核心数扩展及空载 / 有载延迟。所有数字均来自保留的原始测量。</p>'
        full=[r for r in self.main if r['family']=='full' and r['case_name'] in ('full_socket_ddr','full_socket_hbm')]
        if len(full)==16 and all(self.val(r) is not None for r in full):
            content+='<p>完整 304 核 DDR/HBM 的 16 个动作/策略组合均有至少 3 个有效样本。DDR、HBM、L1D 和 L2 工作集的单 NUMA 曲线覆盖 1、2、4、8、15、24、32、36、37、38 核。</p>'
        content+='<ol>'+''.join('<li>'+esc(f)+'</li>' for f in facts)+'</ol>'
        if not facts:content+='<p>数据尚在采集中；本版本仅作排版校验，最终生成时将自动填入满足质量条件的结论。</p>'
        content+='<p class="note">阅读表格：粗体是中位数；下一行给出最小–最大值、有效 n 和严格 n。带宽均为 GB/s；延迟均为 ns。共享主机环境下，本报告不声称这些数字是隔离运行的硬件极限。</p>'
        pending=[r for r in self.main if self.val(r) is None]
        if pending and all(r['case_name'] in ('full_die_a_ddr','available_die_a_ddr') and r['operation']=='write' and r['access']=='stream' for r in pending):
            content+='<p class="note">待补：die A 独立 DDR 流式写入的 148 / 152 核两组有效重复数不足。CPU 380 后续出现新监控任务，按“避开占用核心、满核稍后补测”的安排保留原数据，等待该核空闲后补充。晚期缓存聚合相应使用 151 / 152 / 303 核，并在表中列出实际核数。</p>'
        self.section('主要发现',content)

    def reproduce(self):
        content='<p>原始目录每个案例保存 CSV、终端日志、完整参数 JSON；manifest.json 保存测试矩阵，states.jsonl 保存完成时间及退出状态，before / latest-health / after 保存内存、频率与 CPU 状态。CSV 包含线程级明细和拒绝原因；summary.csv 保留全部轮次，selected-summary.csv 按既定补测优先级选择结果，qualified_median 仅在有效样本不少于 3 时填写；analysis.json 保留逐样本可用性标记。</p>'
        content+='<pre>'+esc('''# On 920f-4 (the shell must initialize the compiler environment)
source ~/.bashrc
cd /home/ryz/vecops-neo
cmake -S . -B build/memory-bench -DCMAKE_BUILD_TYPE=Release \\
  -DCMAKE_C_COMPILER="$(command -v clang)" \\
  -DCMAKE_CXX_COMPILER="$(command -v clang++)" \\
  -DVECOPS_BUILD_MEMORY_TOOLS=ON
cmake --build build/memory-bench --target vecops-memory-bench -j8
python3 tools/memory/CheckAssembly.py build/memory-bench
python3 tools/memory/RunSocketCampaign.py \\
  --binary build/memory-bench/tools/memory/vecops-memory-bench \\
  --output build/memory-bench/campaign-replay/main \\
  --replay-manifest build/memory-bench/campaign-20261029/main/manifest.json \\
  --samples 5 --seconds .15 --retries 0 --exclude-cpus 306-309
# The original main run preceded the scaling extension; the follow-up is retained separately.
python3 tools/memory/RunSocketCampaign.py \\
  --binary build/memory-bench/tools/memory/vecops-memory-bench \\
  --output build/memory-bench/campaign-replay/scaling \\
  --replay-manifest build/memory-bench/campaign-20261029/scaling/manifest.json \\
  --samples 5 --seconds .15 --retries 0 --exclude-cpus 306-309
# Final cache matrix: per-case durations are retained in the manifest.
python3 tools/memory/RunSocketCampaign.py \\
  --binary build/memory-bench/tools/memory/vecops-memory-bench \\
  --output build/memory-bench/campaign-replay/cache-final \\
  --replay-manifest build/memory-bench/campaign-20261029/cache-final-input.json \\
  --samples 5 --seconds .15 --retries 0 --exclude-cpus 380 \\
  --default-memory-bind 8-15,24-31
# On the local analysis host
python3 tools/memory/SummarizeSocketCampaign.py build/memory-bench/campaign-20261029
python3 tools/memory/BuildSocketReport.py build/memory-bench/campaign-20261029''')+'</pre>'
        content+='<p>复现完整 304 核项目时，先确认占用任务已结束，再使用单独输出目录、<code>--phase full --exclude-cpus ""</code>。不要在共享主机上直接与其他负载重叠运行。</p>'
        content+='<details><summary>全部主指标索引（含 temporal、stream、全部规模及质量）</summary>'+self.table(['CSV','案例','动作','策略','实际核数','每数组/核','指标中位数','有效/严格'],[[f'<code>{esc(r["source_csv"])}</code>',esc(r['case_name']),esc(r['operation']),esc(r['access']),str(r['threads']),esc(r['requested_size']),number(self.val(r)),f'{r["samples_usable"]}/{r["samples_strict"]}'] for r in self.rows_for()])+'</details>'
        self.section('复现与完整数据索引',content)

    def sdma_supplement(self):
        path=self.root/'sdma-20261031'/'run-03'/'report-section.html'
        if path.exists():
            self.section('920f-lx SDMA：设备、通道与完整搬运测试',path.read_text())
            self.images.extend(sorted((path.parent/'report').glob('*.png')))

    def build(self):
        self.conclusions();self.methodology();self.sdma_supplement();self.core_scaling();self.aggregate();self.routes();self.interleaved_placement();self.transfers();self.kernels_and_caches();self.latency();self.loaded();self.pages_and_balance();self.reproduce()
        nav=''.join(f'<a href="#s{i}">{esc(title)}</a>' for i,(title,_) in enumerate(self.sections))
        body=''.join(f'<section id="s{i}"><h2><span>{i+1:02}</span>{esc(title)}</h2>{content}</section>' for i,(title,content) in enumerate(self.sections))
        document='''<!doctype html><!-- SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors; SPDX-License-Identifier: MIT --><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>920f-4 Socket 1 内存层次性能报告</title><style>
:root{--navy:#15364a;--teal:#147d92;--muted:#607786;--line:#dce6ea}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f2f6f7;color:#203744;font:15px/1.75 system-ui,"Noto Sans CJK SC",sans-serif}header{background:var(--navy);color:white;padding:48px max(28px,calc((100vw - 1250px)/2));border-bottom:7px solid #35b3af}header .eyebrow{letter-spacing:3px;color:#95dcd9;font-size:13px}h1{font-size:36px;line-height:1.35;margin:12px 0}header p{color:#d5e5eb;margin:6px 0}nav{max-width:1300px;margin:20px auto;padding:12px 24px;display:flex;flex-wrap:wrap;gap:8px 20px}nav a{color:var(--teal);text-decoration:none;font-size:13px}main{max-width:1300px;margin:auto;padding:0 24px 50px}section{background:white;padding:30px 34px;margin:24px 0;border:1px solid var(--line);border-radius:10px;scroll-margin-top:16px}h2{font-size:24px;line-height:1.4;color:var(--navy);margin:0 0 22px}h2 span{color:#5aa7ae;margin-right:15px;font-weight:400}h3{font-size:18px;margin:30px 0 12px;color:var(--teal)}p{margin:13px 0}.lead{font-size:18px}.note{border-left:4px solid #d69a49;background:#fff9ef;padding:12px 16px}.muted,small{color:var(--muted)}small{display:block;font-size:10px;line-height:1.6;font-variant-numeric:tabular-nums;white-space:nowrap}li{padding:7px 0}.scroll{overflow-x:auto;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:12px;line-height:1.6}th{text-align:left;background:#eaf2f5;color:var(--navy);font-weight:650;white-space:nowrap}td,th{padding:10px 11px;border-bottom:1px solid var(--line);vertical-align:top}td b{font-size:14px;font-variant-numeric:tabular-nums}tbody tr:nth-child(even){background:#f8fbfc}figure{margin:24px 0}figure img{display:block;max-width:100%;height:auto;margin:auto}figcaption{font-size:12px;color:var(--muted);margin-top:8px}code{font:12px/1.7 ui-monospace,monospace;overflow-wrap:anywhere}pre{background:#edf3f5;padding:20px;overflow:auto;font:12px/1.7 ui-monospace,monospace}details{border:1px solid var(--line);border-radius:5px;margin:18px 0;padding:12px}summary{cursor:pointer;color:var(--teal);font-weight:600}footer{max-width:1250px;margin:0 auto 30px;color:var(--muted);font-size:12px}@media(max-width:700px){h1{font-size:26px}main{padding:0 10px}section{padding:20px 14px}header{padding:30px 20px}h2{font-size:21px}}@media print{body{background:white}nav{display:none}section{border:0;padding:8px;break-inside:auto}figure,table{break-inside:avoid}details{display:block}header{padding:20px;color:#15364a;background:white}header p{color:#607786}h1{font-size:28px}}
</style></head><body><header><div class="eyebrow">MEMORY PERFORMANCE · MEASURED EVIDENCE</div><h1>920f-4 · Socket 1<br>内存层次性能报告</h1><p>DDR / HBM · NUMA 与 die 间访问 · L1D / L2 · 带宽与延迟</p><p>2026-10-29 主体 · 2026-10-30 页交织补充　|　BiSheng Clang 19.1.7　|　FP32　|　SVE 64 B</p></header>'''+f'<nav>{nav}</nav><main>{body}</main><footer>报告自包含全部图像；测量数据、质量标记与复现命令随报告一起交付。生成时间：{datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")}（Asia/Shanghai）。</footer></body></html>'
        if (self.root/'sdma-20261031'/'run-03'/'report-section.html').exists():
            document=document.replace('2026-10-30 页交织补充', '2026-10-30 页交织补充 · 2026-10-31 920f-lx SDMA 补充')
        (self.out/'920f-4-socket1-report-zh.html').write_text(document,encoding='utf-8')
        print(json.dumps(dict(report=str(self.out/'920f-4-socket1-report-zh.html'),figures=len(self.images),completed_cases=len(self.data['cases']),interleave_cases=len([r for r in self.interleave_data['selected_summaries'] if r['scope']=='total']) if self.interleave_data else 0),ensure_ascii=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();Report(args.root).build()


if __name__=='__main__':main()
