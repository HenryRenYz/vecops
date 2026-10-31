# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Render a self-contained Chinese SDMA supplement from audited observations."""
import argparse
import base64
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fontManager.addfont(FONT.get_file())
plt.rcParams.update({'font.family':FONT.get_name(),'axes.unicode_minus':False,
                     'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})


def esc(x):return html.escape(str(x))
def num(x):return '—' if x is None else f'{x:,.2f}'
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--allow-partial',action='store_true');args=p.parse_args()
    root=args.root;data=json.loads((root/'analysis.json').read_text());env=data['environment']
    execution=json.loads((root/'execution.json').read_text()) if (root/'execution.json').exists() else {}
    job_id=str(execution.get('job_id',env.get('scheduler_job') or 'unknown'))
    if not args.allow_partial:
        if not env.get('finished') or not (root/'audit-complete.json').exists():raise RuntimeError('Campaign/final audit has not finished')
        completed=json.loads((root/'audit-complete.json').read_text())
        if any(completed[k]!=data[k] for k in ['cases','raw_rows']) or completed['binary_sha256']!=env['binary_sha256']:raise RuntimeError('Final audit marker is stale')
    if data['audit_errors']:raise RuntimeError('Refusing to render unaudited data')
    rows=[r for r in data.get('selected_summaries',data['summaries']) if r['scope']=='total']
    out=root/'report';out.mkdir(exist_ok=True)
    def good(r):return r and r['usable']>=3
    def val(r):return r['payload_GB_s'] if good(r) else None
    def find(name):return next((r for r in rows if r['name']==name),None)
    def metric(r):
        if not r:return '未执行'
        if r['family']=='correctness':return f'数据/驻留校验通过<small>计时有效 {r["usable"]}/{r["recorded"]}；不作为性能结论</small>'
        if not good(r):return f'有效样本不足（{r["usable"]}/{r["recorded"]}）'
        return f'<b>{num(val(r))}</b><small>{num(r["minimum"])}–{num(r["maximum"])}；n={r["usable"]}，严格={r["strict"]}</small>'
    images=[]
    def figure(name,fig,caption):
        fig.tight_layout(pad=1.8);path=out/(name+'.png');fig.savefig(path,bbox_inches='tight',dpi=150);plt.close(fig);images.append(path)
        return '<figure><img alt="'+esc(caption)+'" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"><figcaption>'+esc(caption)+'</figcaption></figure>'
    body='<p class="lead">新增直接调用 sdma-dk 的异步拷贝测试。本节数据来自 920f-lx 计算节点 '+esc(env['host'])+'，与前文 920f-4 的历史 CPU 数据分别标注；本节 CPU 对照也在同一计算节点重新测量。</p>'
    body+='<p class="note">范围：全部可见 SDMA 设备，源/目的数组和控制线程限制在 socket 1。设备编号不代表 NUMA、die 或 socket 归属。结果是此内存范围和当前共享通道/安全模式下的观测，未声明整个机器的理论 DMA 峰值。</p>'
    main_rows=[r for r in rows if r['family']!='correctness']
    body+=table(['环境或口径','记录'],[
        ['节点 / 内核',esc(env['host'])+' / '+esc(env['kernel'])],
        ['可见设备 / 物理引擎',esc(env['visible_devices'])+' / 物理引擎数量未知'],
        ['设备 NUMA 归属','检查到的 sysfs 无明确属性，保留 unknown'],
        ['编译器',esc(env.get('compiler',''))],['基础页 / SIMD',esc(env.get('base_page_bytes',''))+' B / '+esc(env.get('backend',''))],
        ['工具源码 SHA-256','<code>'+esc(env.get('tool_source_sha256',''))+'</code>'],
        ['SDK',esc(env['packages'])],['驱动配置','shared；share_chns='+esc(env['share_chns'])+'；safe_mode='+esc(env['safe_mode'])+'；未修改全局参数'],
        ['调度','Donau 作业 '+esc(job_id)+'，独占单节点；296 个 socket 1 隔离计算核心；32 GiB；控制程序另绑 CPU 606'],
        ['带宽','主指标为 payload GB/s，搬运字节只计一次；与 CPU 历史读+写 GB/s 比较时须统一口径'],
        ['测试覆盖',f'{env.get("planned_cases",data["cases"])} 个计划配置，含补测共 {data["cases"]} 次执行，{data["raw_rows"]:,} 条原始行；{sum(good(r) for r in main_rows)} 个性能配置至少有 3 个有效样本'],
        ['审计',f'{len(data["audit_errors"])} 项错误；逐样本检查字节数、完成批次、时间包络、线程/组汇总和逐页计数'],
        ['ELF SHA-256','<code>'+esc(env['binary_sha256'])+'</code>'],
        ['SDK / 头文件 SHA-256','<code>'+esc(env['sdk_sha256'])+'<br>'+esc(env['header_sha256'])+'</code>']])
    if all(r['pages']=='base' for r in rows):
        body+='<p>本轮全部成绩使用 base 页。现有大页池不足，未修改全局池配置；SDMA 大页路径保留为支持功能，本轮没有给出其性能结论。</p>'
    body+='<h3>驱动推荐设备与拓扑边界</h3><p>以下是在每个 NUMA 的首个计算 CPU 上调用已安装 SDK 的 sdma_nearest_id（经驱动 ioctl 查询）所得推荐值，仅描述驱动的选择建议。它没有给出物理引擎数量或设备的独占服务范围。</p>'
    body+='<p>NUMA 8–15 的所有计算核心均核验为 physical_package_id=1。die0/1 标签沿用 NUMA 8–11、12–15 两组；该节点 Linux die_id 为 -1，SDMA 推荐值没有被用来推导物理 die。</p>'
    body+=table(['执行 NUMA','采样 CPU','驱动推荐设备'],[[n,str(int(n)*38),str(d)] for n,d in env['recommended_devices'].items()])
    facts=[]
    for route,label in [('24:24','HBM→HBM'),('8:8','DDR→DDR')]:
        candidates=[r for r in rows if r['family']=='saturation' and r['routes']==route and good(r)]
        if candidates:
            best=max(candidates,key=lambda r:val(r))
            facts.append(f'固定 NUMA 8/24 内存池的 {label} 参数扫描中，最高有效配置中位数为 {num(val(best))} GB/s（sdma{best["devices"]}，{best["channels"]} 通道、{best["threads"]} 个控制线程）；该配置范围 {num(best["minimum"])}–{num(best["maximum"])} GB/s。')
    for memory,label in [('ddr-ddr','DDR→DDR'),('hbm-hbm','HBM→HBM')]:
        candidates=[r for r in rows if r['family']=='socket-saturation' and memory in r['name'] and good(r)]
        if candidates:
            best=max(candidates,key=lambda r:val(r))
            facts.append(f'socket 1 八 NUMA 的 {label} 饱和扫描最高有效配置为 {num(val(best))} GB/s，{best["channels"]} 通道/组、共 {best["threads"]} 个控制线程，'+('页交织' if best['interleave'] else '绑定')+f'，约占 {num(best["controller_equivalent_cores"])} 个 CPU 等效核心。')
    local=find('route-24-24');near=find('route-24-25');far=find('route-24-28')
    if all(good(r) for r in [local,near,far]):
        facts.append(f'相同源 NUMA 24 与控制条件下，HBM 目的 24、25、28 分别为 {num(val(local))}、{num(val(near))}、{num(val(far))} GB/s。该组数据反映 DMA 发起路径，不能直接套用 CPU 读写的带宽比例。')
    four=find('all-devices-4-hbm-spread')
    if good(four):
        facts.append(f'本轮最高同步聚合观测为 {num(val(four))} GB/s：4 个可见设备访问跨两组分布的 HBM 私有数组，4 个控制线程约占 {num(four["controller_equivalent_cores"])} 个等效核心。读+写应用口径为 {num(four["total_GB_s"])} GB/s；这不是物理 fabric 线速。')
    bound=find('aggregate-socket-hbm-hbm-i0');interleaved=find('aggregate-socket-hbm-hbm-i1')
    if good(bound) and good(interleaved):
        facts.append(f'同条件八 NUMA HBM copy：绑定 {num(val(bound))} GB/s，页交织 {num(val(interleaved))} GB/s，交织/绑定为 {val(interleaved)/val(bound)*100:.1f}%。设备按驱动推荐分配，总共 8 个控制线程。')
    cpu=find('cpu-equal-array-24:24');dma=find(f'device-{env["recommended_devices"]["8"]}-24:24-ch16')
    if good(cpu) and good(dma):
        facts.append(f'匹配 64 MiB/数组、base 页、CPU 304 的单线程对照：CPU vec4 stream 为 {num(val(cpu))} GB/s，SDMA 为 {num(val(dma))} GB/s，约 {val(dma)/val(cpu):.2f} 倍。SDMA 采用 16 通道、64 KiB、depth=16、batch=8；这个倍率没有混入不同工作集的最高成绩。')
    if facts:body+='<h3>本轮实测要点</h3><ul>'+''.join('<li>'+esc(f)+'</li>' for f in facts)+'</ul>'
    body+='<h3>各设备的通道扩展与饱和扫描</h3><p>先固定执行 CPU 304 和相同内存路径，对所有设备扫描 1、2、4、8、16、32 个通道（每个控制线程私有源/目的数组各 64 MiB，64 KiB 描述符，depth=16，batch=8）。再固定每组每数组 128 MiB，扫描 16、64、128 通道与 1、4、16 个控制线程（256 KiB、depth=batch=4）。多个控制线程分摊固定数组和通道数。</p>'
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    labels={'8:8':'DDR→DDR','8:24':'DDR→HBM','24:8':'HBM→DDR','24:24':'HBM→HBM'}
    for ax,(route,label) in zip(axes.flat,labels.items()):
        for d in env['visible_devices']:
            x=[1,2,4,8,16,32];y=[val(find(f'device-{d}-{route}-ch{n}')) for n in x]
            ax.plot(x,[np.nan if v is None else v for v in y],'-o',label=f'sdma{d}')
        ax.set(title=label,xlabel='每组通道数',ylabel='payload GB/s',xticks=x);ax.grid(alpha=.2)
    axes[0,0].legend();body+=figure('sdma-device-channels',fig,'相同 socket 1 源/目的内存、CPU 304 发起，各设备的通道扩展。空点表示有效样本不足。')
    peaks=[]
    for d in env['visible_devices']:
        for route,label in labels.items():
            candidates=[r for r in rows if r['family']=='saturation' and str(r['devices'])==str(d) and r['routes']==route and good(r)]
            best=max(candidates,key=lambda r:val(r)) if candidates else None
            peaks.append([f'sdma{d}',label,metric(best),str(best['channels']) if best else '—',str(best['threads']) if best else '—',num(best['controller_equivalent_cores']) if best else '—',num(best['cpu_seconds_per_GiB']) if best else '—'])
    body+='<p>下表列出饱和参数扫描中最高的有效配置中位数，保留其通道/线程条件和全部扫描明细；它是配置搜索结果，未与独立确认测量混同，也未称为理论峰值。</p>'
    body+=table(['设备','路径','扫描最高 payload GB/s','通道','控制线程','CPU 等效核数','CPU s/GiB'],peaks)
    body+='<h3>驱动推荐设备对应 NUMA 组的对称饱和对照</h3><p>固定池扫描之后，对驱动推荐设备 3 补充 NUMA 12/28 的同参数扫描，与推荐设备 2 的 NUMA 8/24 扫描分别对应。推荐值仅用于选取对照组，不改变设备归属为 unknown 的记录。每组每数组 128 MiB，256 KiB 描述符、depth=batch=4。</p>'
    local_rows=[]
    first={}
    for node,device in sorted(env['recommended_devices'].items(),key=lambda x:int(x[0])):first.setdefault(device,int(node))
    for device,node in first.items():
        for src,dst,label in [(node,node,'DDR→DDR'),(node,node+16,'DDR→HBM'),(node+16,node,'HBM→DDR'),(node+16,node+16,'HBM→HBM')]:
            candidates=[r for r in rows if r['family'] in ['saturation','local-saturation'] and str(r['devices'])==str(device) and r['routes']==f'{src}:{dst}' and good(r)]
            best=max(candidates,key=lambda r:val(r)) if candidates else None
            local_rows.append([f'sdma{device}',f'{src}→{dst} / {label}',metric(best),str(best['channels']) if best else '—',str(best['threads']) if best else '—',num(best['controller_equivalent_cores']) if best else '—'])
    body+=table(['设备','源/目的内存','扫描最高 GB/s','通道','控制线程','CPU 等效核数'],local_rows)
    body+='<h3>请求大小、队列深度与提交开销</h3>'
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,route in zip(axes,['8:24','24:24']):
        for chunk in ['4KiB','16KiB','64KiB','256KiB','1MiB']:
            x=[1,4,16,64];y=[val(find(f'queue-{route}-{chunk}-q{d}')) for d in x]
            ax.plot(x,[np.nan if v is None else v for v in y],'-o',label=chunk)
        ax.set(xlabel='每通道在途描述符上限',ylabel='payload GB/s',title=labels[route],xticks=[1,4,16,64]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);body+=figure('sdma-queue',fig,'单通道请求大小与 depth 扫描，batch=min(8,depth)，控制线程使用驱动推荐设备。')
    body+='<p>CPU 时间由 CLOCK_THREAD_CPUTIME_ID 测量，已先验证单个忙轮询线程约占 1 个等效核心。setup_seconds 是分配、初始化、pin、通道创建及完整验证拷贝的准备时间；带宽测量排除准备阶段。聚合行的 setup_seconds 与 drain_ns 分别取各线程准备时间、排空时间的最大值；带宽分母仍取全体最早开始到最后完成。CPU 等效核数=计时内控制线程 CPU 秒之和/共同耗时，CPU s/GiB 使用有效搬运字节。轮询可占满控制核心，不能把 DMA 吞吐直接解释为零 CPU 开销。该指标仅统计控制线程，其他核心上的驱动/中断开销未计为整机 CPU 成本。</p>'
    body+='<h3>socket 1 全部 DDR/HBM 节点的 16×16 拷贝矩阵</h3><p>每个单向案例一个控制线程，位于源内存对应的计算 NUMA，设备采用驱动推荐值；16 通道、depth=16、batch=8、256 KiB、每数组 64 MiB。含 HBM 的路径使用当时已有大页池（若不足则整个矩阵选择 base）；具体 pages 记录在每条案例中。没有按测得速度重选设备。</p>'
    nodes=list(range(8,16))+list(range(24,32));matrix=np.full((16,16),np.nan)
    for i,src in enumerate(nodes):
        for j,dst in enumerate(nodes):
            v=val(find(f'route-{src}-{dst}'));matrix[i,j]=np.nan if v is None else v
    fig,ax=plt.subplots(figsize=(12,10));im=ax.imshow(np.ma.masked_invalid(matrix),cmap='YlGnBu',vmin=0)
    ax.set(xticks=range(16),yticks=range(16),xticklabels=nodes,yticklabels=nodes,xlabel='目的内存 NUMA',ylabel='源内存 NUMA',title='SDMA 单向有效搬运带宽 / GB/s')
    for i in range(16):
        for j in range(16):
            if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=7,color='white' if matrix[i,j]>np.nanmax(matrix)*.6 else '#17364a')
    ax.axhline(7.5,color='#d48035');ax.axvline(7.5,color='#d48035');fig.colorbar(im,ax=ax,shrink=.8,label='payload GB/s')
    body+=figure('sdma-route-matrix',fig,'源 DDR 8–15 / HBM 24–31，目的相同；同一行的设备和控制 NUMA 固定。缺格为有效样本不足。')
    body+='<h3>单向与双向同时搬运</h3><p>每方向 8 通道、64 KiB、depth=16、batch=8、每数组 64 MiB。双向同时案例使用两组独立源/目的数组；同执行 NUMA 时使用不同控制 CPU。表中“双向合计”按两方向完成字节之和除以统一包络计算。单向正反两次运行的带宽之和仅作对照。</p>'
    direction=[]
    for src,dst in [(8,9),(8,12),(24,25),(24,28),(8,24),(8,28),(24,12)]:
        f=find(f'direction-{src}-{dst}-forward');r=find(f'direction-{src}-{dst}-reverse');b=find(f'direction-{src}-{dst}-both')
        ratio=val(b)/(val(f)+val(r)) if all(good(x) for x in [f,r,b]) else None
        direction.append([f'{src} ↔ {dst}',metric(f),metric(r),metric(b),num(ratio)])
    body+=table(['节点对','正向 GB/s','反向 GB/s','双向同时合计 GB/s','合计/两次单向之和'],direction)
    body+='<h3>die / socket 聚合与内存交织</h3><p>每个执行 NUMA 持有私有数组并管理 8 通道；组内数组各 64 MiB、256 KiB 描述符、depth=16、batch=8。内存交织沿用源/目的节点池分别交织的规则，交织粒度为实际页大小。共同起止偏差各不得超过目标时长的 0.1%。</p>'
    aggregate=[]
    for label in ['die0','die1','socket']:
        for memory in ['ddr-ddr','ddr-hbm','hbm-ddr','hbm-hbm','hbm-ring','hbm-crossdie','ddr-crossdie']:
            a=find(f'aggregate-{label}-{memory}-i0');b=find(f'aggregate-{label}-{memory}-i1')
            ratio=val(b)/val(a) if good(a) and good(b) else None
            aggregate.append([label,esc(memory),metric(a),metric(b),num(ratio)])
    body+=table(['范围','内存路径','绑定 GB/s','页交织 GB/s','交织/绑定'],aggregate)
    body+='<h3>socket 1 两个推荐设备的高并发饱和扫描</h3><p>八个执行 NUMA 同时运行；每组 16 或 32 通道、1/4/16 个控制线程、每线程每数组 64 MiB、256 KiB 描述符、depth=batch=4。原始窗口为 2 s，补测窗口见完整明细。下表给出每种内存/页策略的参数扫描最高有效配置，列出实际控制核心成本。</p>'
    saturation=[]
    for memory in ['ddr-ddr','ddr-hbm','hbm-ddr','hbm-hbm']:
        for interleave in [False,True]:
            candidates=[r for r in rows if r['family']=='socket-saturation' and memory in r['name'] and r['interleave']==interleave and good(r)]
            best=max(candidates,key=lambda r:val(r)) if candidates else None
            saturation.append([memory,'交织' if interleave else '绑定',metric(best),str(best['channels']) if best else '—',str(best['threads']) if best else '—',num(best['controller_equivalent_cores']) if best else '—'])
    body+=table(['路径','页策略','扫描最高 GB/s','每组通道','总控制线程','CPU 等效核数'],saturation)
    body+='<h3>全部可见设备同时运行与共享通路</h3><p>按设备 ID 顺序加入 1 至 4 个设备，每设备一组、16 通道、64 MiB/数组、256 KiB 描述符。same 访问同一内存节点上的私有数组；split 分别访问 8、9、10、11（或 HBM 24–27）。控制线程分别放在 NUMA 8、9、10、11。spread 按 8、12、9、13（或 HBM 24、28、25、29）分布到两个 NUMA 组。该实验研究资源共享和 socket 1 内存范围内的聚合上限。</p>'
    all_devices=[]
    for n in range(1,len(env['visible_devices'])+1):
        for memory in ['ddr','hbm']:
            same=find(f'all-devices-{n}-{memory}-same');split=find(f'all-devices-{n}-{memory}-split');spread=find(f'all-devices-{n}-{memory}-spread')
            all_devices.append([str(n),memory,metric(same),metric(split),metric(spread),num(split['controller_equivalent_cores']) if good(split) else '—'])
    body+=table(['并行设备数','内存','同节点 GB/s','die 内分节点 GB/s','两组分布 GB/s','die 内分节点 CPU 等效核数'],all_devices)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,memory in zip(axes,['ddr','hbm']):
        for placement,label in [('same','同一节点'),('split','die 内分节点'),('spread','两组分布')]:
            x=list(range(1,len(env['visible_devices'])+1));y=[val(find(f'all-devices-{n}-{memory}-{placement}')) for n in x]
            ax.plot(x,[np.nan if v is None else v for v in y],'-o',label=label)
        ax.set(xlabel='同时运行的可见 SDMA 设备数',ylabel='payload GB/s',title=memory.upper()+'→'+memory.upper(),xticks=x);ax.grid(alpha=.2)
    axes[0].legend();body+=figure('sdma-all-devices',fig,'按设备 ID 顺序加入设备，私有数组；改变内存节点分布来观察共享和聚合。曲线不是物理引擎数量的推断。')
    body+='<h3>同机 CPU copy 对照</h3><p>以下对照固定每数组 64 MiB、base 页、一个 CPU/控制线程和同一源/目的节点。SDMA 使用对应控制 NUMA 的驱动推荐设备、16 通道、64 KiB、depth=16、batch=8；CPU 使用 vec4 stream。两者统一为 payload GB/s。额外 1/4/16/37 核 CPU 数据保留在完整表中，数组为每线程 4 MiB，不能把不同工作集直接归因于引擎差异。</p>'
    cpu=[]
    d=env['recommended_devices']['8']
    for route,label in labels.items():
        dma=find(f'device-{d}-{route}-ch16');c=find(f'cpu-equal-array-{route}')
        cpu.append([label,metric(c),metric(dma),num(val(dma)/val(c)) if good(c) and good(dma) else '—',num(dma['controller_equivalent_cores']) if good(dma) else '—'])
    body+=table(['路径','CPU payload GB/s','SDMA payload GB/s','SDMA/CPU','SDMA CPU 等效核数'],cpu)
    poor=[r for r in main_rows if not good(r)]
    body+='<h3>质量、限制与复现</h3><p>每一计时阶段更换输入代际、毒化目的数组，完整检查源数据、已完成拷贝覆盖区和未触及后缀哨兵；所有分配页在前后查询精确 NUMA 驻留。只有 SDK 确认完成的批次计入字节，末尾排空时间包含在分母。工具严格 ok 还要求无上下文切换；本节接受符合共同窗口与零缺页条件的 interrupted 样本并保留完整耗时，不从同一配置中挑最快样本。每个性能结论要求至少 3 个有效样本。样本不足时按观测到的起止偏差预定更长窗口补测，选用最后预定条件的完整样本集合；不同时间窗口不混合，原始短窗口和补测条件均保留。</p>'
    body+='<p>sdma-dk 的共享查询在请求 CQ 代际尚未到达时可返回 -8（SDMA_RNDCNT_ERR）；后端保留请求继续查询，其他错误立即失败，排空有 30 s 上限。首次兼容性失败单独保留，未纳入成绩。所有主统计均为当前 safe_mode 与共享通道路径，无法外推关闭安全模式或独占通道的成绩。</p>'
    body+='<p>单次样本中的每描述符平均完成速率没有被标记为内存访问延迟；本轮也没有把应用搬运字节换算为物理 fabric、DDR 或 HBM 总线流量。单设备瓶颈可能来自提交开销、通道、内存控制器或互联，需结合通道/线程/设备/内存池对照判断，不能仅凭相近带宽认定物理连接。</p>'
    if poor:body+='<details><summary>有效样本不足的性能配置：'+str(len(poor))+'</summary>'+table(['案例','有效/原始','payload GB/s'],[[esc(r['name']),f'{r["usable"]}/{r["recorded"]}',metric(r)] for r in poor])+'</details>'
    body+='<details><summary>全部配置及质量明细</summary>'+table(['案例 ID','实际源:目的','类型','页','请求大小/范围','总线程','窗口 s','通道','depth/batch/chunk','payload GB/s','CPU 等效核数','排空 μs'],[[esc(r['id']),esc(r['routes']),esc(r['family']),esc(r['pages']),esc(r['size'])+(' /组' if r.get('size_scope')=='group-array' else ' /线程'),str(r.get('threads','—')),num(r['seconds']),str(r['channels']) if r['engine']=='sdma' else '—',esc(f'{r["depth"]}/{r["batch"]}/{r["chunk"]}'),metric(r),num(r.get('controller_equivalent_cores')),num(r['drain_ns']/1000 if r['drain_ns'] is not None else None)] for r in rows])+'</details>'
    body+='<p>远端 SSH 别名 <code>920f-lx</code>，项目 <code>/home/share/youxin/vecops-neo-sdma-20261031</code>，构建 <code>build/sdma</code>；Donau 作业 '+esc(job_id)+' 的资源申请、完整任务脚本和退出状态随 execution 目录保留。原始数据与逐配置命令：<code>sdma-20261031/run-03/manifest.json</code>；汇总 <code>analysis.json / summary.csv</code>；编译输入、ELF 和构建审计在 <code>provenance/</code>。按设备、路由、线程、通道和页策略查阅对应 CSV，未把不同配置混合求平均。未随机化全部案例顺序，结果代表该节点此时段的观测。</p>'
    (root/'report-section.html').write_text('<!-- SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors; SPDX-License-Identifier: MIT -->\n'+body)
    print(json.dumps(dict(section=str(root/'report-section.html'),figures=len(images),insufficient=len(poor)),ensure_ascii=False))


if __name__=='__main__':main()
