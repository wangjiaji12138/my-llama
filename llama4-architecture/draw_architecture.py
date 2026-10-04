"""Reproducible, editable SVG figures for Meta's Llama 4 reference implementation."""
from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parent
INK = '#243247'
MUTED = '#647187'
LINE = '#8B99AD'
BG = '#FBFCFE'
BLUE = '#E8F2FF'
BLUE_INK = '#356BB6'
GREEN = '#E7F3EA'
GREEN_INK = '#397957'
PURPLE = '#EEEAFB'
PURPLE_INK = '#7160A3'
PEACH = '#FFF0DE'
PEACH_INK = '#A26A27'

class Figure:
    def __init__(self, w, h, title, desc):
        self.w, self.h = w, h
        self.parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke="{LINE}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker>
<marker id="arrow-purple" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke="{PURPLE_INK}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker>
</defs><rect width="100%" height="100%" fill="{BG}"/>
<g font-family="PingFang SC, Hiragino Sans GB, Noto Sans CJK SC, sans-serif" fill="{INK}">''']
    def add(self, s): self.parts.append(s)
    def rect(self,x,y,w,h,fill='white',stroke='#D6DEE8',r=18,dash=None,sw=1.8):
        ds=f' stroke-dasharray="{dash}"' if dash else ''
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{ds}/>')
    def text(self,x,y,s,size=26,weight=400,color=INK,anchor='start',family=None):
        fam=f' font-family="{family}"' if family else ''
        self.add(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}"{fam}>{escape(s)}</text>')
    def lines(self,x,y,ss,size=24,dy=36,color=MUTED,anchor='start'):
        for i,s in enumerate(ss): self.text(x,y+i*dy,s,size,color=color,anchor=anchor)
    def path(self,d,color=LINE,sw=2.4,arrow=True,dash=None):
        marker='arrow-purple' if color==PURPLE_INK else 'arrow'
        ds=f' stroke-dasharray="{dash}"' if dash else ''
        ar=f' marker-end="url(#{marker})"' if arrow else ''
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"{ar}{ds}/>')
    def circle(self,x,y,r=17,label='+',fill='white',color=INK):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{LINE}" stroke-width="1.8"/>')
        if label: self.text(x,y+9,label,28,500,color,'middle')
    def icon(self,name,x,y,color=INK,size=40):
        # Purpose-drawn diagram symbols; all remain editable vector geometry.
        self.add(f'<g transform="translate({x},{y}) scale({size/40})" stroke="{color}" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">')
        shapes={
          'text':'<path d="M6 8h28M20 8v26M13 34h14M8 8v5M32 8v5"/>',
          'image':'<rect x="3" y="5" width="34" height="29" rx="5"/><circle cx="28" cy="13" r="3"/><path d="M5 29l10-11 8 8 5-5 7 8"/>',
          'tiles':''.join(f'<rect x="{a}" y="{b}" width="13" height="13" rx="2"/>' for a in [4,23] for b in [4,23]),
          'token':'<rect x="2" y="9" width="11" height="22" rx="3"/><rect x="16" y="9" width="9" height="22" rx="3"/><rect x="28" y="9" width="10" height="22" rx="3"/>',
          'embed':'<path d="M5 7h30v27H5zM5 16h30M5 25h30M15 7v27M25 7v27"/>',
          'vision':'<path d="M2 20q18-24 36 0Q20 44 2 20z"/><circle cx="20" cy="20" r="7"/>',
          'adapter':'<path d="M3 6h10v10H3zM3 24h10v10H3zM17 11h9M17 29h9M24 11l7 9-7 9"/><rect x="29" y="15" width="9" height="10" rx="2"/>',
          'merge':'<path d="M3 8h10l12 12h12M3 32h10l12-12M31 14l6 6-6 6"/>',
          'stack':'<path d="M4 11L20 3l16 8-16 8zM4 20l16 8 16-8M4 29l16 8 16-8"/>',
          'head':'<path d="M4 28V12M14 28V7M24 28V17M34 28V3M2 35h36"/>',
          'sample':'<rect x="5" y="5" width="30" height="30" rx="6"/><circle cx="13" cy="13" r="1.5"/><circle cx="27" cy="13" r="1.5"/><circle cx="20" cy="20" r="1.5"/><circle cx="13" cy="27" r="1.5"/><circle cx="27" cy="27" r="1.5"/>',
          'chat':'<path d="M8 5h25a4 4 0 014 4v18a4 4 0 01-4 4H16L7 37v-6H6a4 4 0 01-4-4V11a6 6 0 016-6zM10 14h20M10 22h14"/>',
          'norm':'<path d="M4 20h8M28 20h8M20 4v8M20 28v8"/><circle cx="20" cy="20" r="7"/>',
          'network':'<circle cx="7" cy="20" r="4"/><circle cx="32" cy="7" r="4"/><circle cx="32" cy="20" r="4"/><circle cx="32" cy="33" r="4"/><path d="M11 18l17-9M11 20h17M11 22l17 9"/>',
          'cache':'<ellipse cx="20" cy="8" rx="15" ry="6"/><path d="M5 8v22c0 8 30 8 30 0V8M5 19c0 8 30 8 30 0"/>',
          'expert':'<rect x="9" y="9" width="22" height="22" rx="4"/><path d="M4 14h5M4 26h5M31 14h5M31 26h5M14 4v5M26 4v5M14 31v5M26 31v5M15 15h10v10H15z"/>',
        }
        self.add(shapes[name]); self.add('</g>')
    def box(self,x,y,w,h,title,subtitle,icon,fill=BLUE,accent=BLUE_INK,code=None,ts=29):
        self.rect(x,y,w,h,fill,stroke='none',r=18)
        self.icon(icon,x+20,y+22,accent,36)
        self.text(x+70,y+48,title,ts,600)
        subs=subtitle if isinstance(subtitle,list) else [subtitle]
        self.lines(x+22,y+89,subs,23,31)
        if code: self.text(x+22,y+h-18,code,18,color=accent,family='Menlo, monospace')
    def label(self,x,y,num,title,en):
        self.rect(x,y-31,44,40,INK,stroke='none',r=9)
        self.text(x+22,y-2,num,20,600,'white','middle')
        self.text(x+61,y,title,29,600)
        self.text(x+61,y+34,en,18,color=MUTED)
    def header(self,kicker,title,subtitle):
        self.text(70,65,kicker,22,600,BLUE_INK)
        self.text(67,143,title,66,650)
        self.text(71,191,subtitle,27,color=MUTED)
    def save(self,name):
        self.add('</g></svg>')
        (OUT/name).write_text('\n'.join(self.parts),encoding='utf-8')

def overview():
    f=Figure(2400,1500,'Llama 4：从图文输入到逐词生成','图标与文字架构图，展示文本和图像编码、早期融合、共享 Transformer 主干、输出和自回归反馈。')
    f.header('LLAMA 4  /  ARCHITECTURE AT A GLANCE  /  01','Llama 4 · 模块功能与调用全景','原生多模态  ·  Early Fusion  ·  Mixture of Experts  ·  自回归生成')
    for x,w,name,stats,color,accent in [(70,1100,'SCOUT','16 路由专家 / MoE 层   ·   17B 激活 / 109B 总参数   ·   10M 上下文',BLUE,BLUE_INK),(1200,1130,'MAVERICK','128 路由专家 / MoE 层   ·   17B 激活 / 400B 总参数   ·   1M 上下文',PURPLE,PURPLE_INK)]:
        f.rect(x,232,w,88,color,stroke='none',r=14)
        f.text(x+24,269,name,25,650,accent)
        f.text(x+24,302,stats,22)
    f.label(85,404,'01','输入与编码','INPUT → REPRESENTATION')
    f.label(1190,404,'02','早期融合','EARLY FUSION')
    f.label(1580,404,'03','共享主干','DECODER BACKBONE')
    f.label(2030,404,'04','输出生成','AUTOREGRESSIVE OUTPUT')
    # Independent text / image routes.
    f.box(85,500,245,140,'文本 / 对话','指令与对话历史','text',ts=28)
    f.box(385,500,305,140,'格式化 + 分词','角色标记 → token IDs','token',code='ChatFormat · Tokenizer',ts=28)
    f.box(745,500,340,140,'Token Embedding','token IDs → 语言向量','embed',code='tok_embeddings(tokens)',ts=27)
    for a,b in [(330,385),(690,745)]:f.path(f'M{a} 570H{b}')
    f.box(85,760,245,140,'图像 / 多图','输入图像像素','image',GREEN,GREEN_INK,ts=28)
    f.box(385,760,305,140,'动态图像切片','缩放 · 填充 · 归一化','tiles',GREEN,GREEN_INK,code='image_transform(...)',ts=27)
    f.box(745,760,340,140,'视觉编码器 · ViT','Patch Embedding → ViT','vision',GREEN,GREEN_INK,code='VisionEncoder.forward',ts=27)
    for a,b in [(330,385),(690,745)]:f.path(f'M{a} 830H{b}')
    f.path('M915 900V955')
    f.box(745,955,340,162,'视觉适配与投影',['Pixel Shuffle → MLP','压缩 token · 对齐语言维度'],'adapter',GREEN,GREEN_INK,code='vision_projection',ts=27)
    f.text(390,990,'图像位置同时写入',23,color=MUTED)
    f.text(390,1025,'<|patch|> 占位 token',23,500,GREEN_INK)
    f.path('M537 760V691H537V640',dash='6 7')
    f.text(562,705,'位置标记',19,color=MUTED)
    # Merge from text and vision into one token stream.
    f.path('M1085 570H1140V660H1190')
    f.path('M1085 1030H1140V850H1190')
    f.rect(1190,600,300,342,'#F2F6F5',stroke='#BCCFC5',r=20)
    f.icon('merge',1214,627,GREEN_INK,42)
    f.text(1273,659,'Early Fusion',29,600)
    f.lines(1215,715,['按 image mask 替换','占位符 → 视觉向量'],23,35)
    # Token strip as a concrete data representation.
    for i,(s,col) in enumerate([('T',BLUE),('T',BLUE),('V',GREEN),('V',GREEN),('T',BLUE)]):
        f.rect(1215+i*49,810,40,46,col,stroke='#BDCCD6',r=7)
        f.text(1235+i*49,841,s,23,600,anchor='middle')
    f.text(1215,899,'图文交错 · 共享主干',22,color=GREEN_INK)
    # Repeated decoder block, compositional not parameter-count scaled.
    f.rect(1590,517,380,540,'#F0ECF8',stroke='#D5CCE6',r=20)
    f.rect(1580,507,380,540,'#F7F4FC',stroke='#D5CCE6',r=20)
    f.rect(1570,497,380,540,'white',stroke='#B9ADCE',r=20)
    f.icon('stack',1597,526,PURPLE_INK,42)
    f.text(1652,558,'Decoder Block × N',29,600)
    f.text(1599,602,'逐层更新图文上下文表示',23,color=MUTED)
    f.box(1598,638,324,125,'Attention',['GQA · iRoPE · KV Cache'],'network',PURPLE,PURPLE_INK,ts=27)
    f.path('M1760 763V803')
    f.box(1598,803,324,132,'FFN / MoE',['Dense：SwiGLU','MoE：共享 + Top-1 路由'],'expert',PURPLE,PURPLE_INK,ts=27)
    f.text(1599,987,'每层含 RMSNorm 与残差连接',22,color=MUTED)
    f.path('M1490 705H1598')
    f.path('M1760 935V953H1982V577H2020')
    f.box(2020,500,305,155,'归一化 + LM Head',['最终表示 → 词表 logits'],'head',PEACH,PEACH_INK,code='norm → output',ts=24)
    f.path('M2172 655V740')
    f.box(2020,740,305,155,'下一个 token',['temperature / top-p','或 argmax 贪心选择'],'sample',PEACH,PEACH_INK,ts=27)
    f.path('M2172 895V970')
    f.box(2020,970,305,135,'解码与输出','流式文本 / 代码','chat',PEACH,PEACH_INK,code='tokenizer.decode',ts=27)
    # Decode recurrence routed around, separated from forward signal.
    f.path('M2325 814H2360V458H914V500',PURPLE_INK,dash='8 8')
    f.rect(1334,441,455,32,BG,stroke='none',r=0)
    f.text(1560,465,'生成 token 回到嵌入层 · 复用 KV Cache',21,500,PURPLE_INK,'middle')
    # Notes and real invocation entry point.
    f.rect(70,1170,2260,192,'#F3F5F9',stroke='none',r=20)
    f.icon('network',95,1198,BLUE_INK,36)
    f.text(149,1224,'推理执行顺序',27,600)
    f.text(350,1224,'chat_completion()  →  encode_dialog_prompt()  →  generate()  →  Transformer.forward()',25,500,INK,family='Menlo, PingFang SC, sans-serif')
    f.lines(101,1276,['有图像：首次 prefill 调用 vision_embeddings()；随后逐 token 解码，注意力读取并更新各层 KV Cache。','FFN 类型与层数按 checkpoint 配置；Maverick 使用 Dense / MoE 交替层。图中 N 表示重复的 Decoder 层。'],23,37)
    f.text(72,1412,'图例',22,600)
    for x,col,label in [(155,BLUE,'文本路径'),(365,GREEN,'视觉路径'),(575,PURPLE,'语言主干'),(785,PEACH,'输出路径')]:
        f.rect(x,1393,26,26,col,stroke='#C8D2DF',r=6);f.text(x+39,1414,label,22)
    f.path('M1040 1407H1103');f.text(1120,1414,'数据 / 调用方向',22,color=MUTED)
    f.path('M1370 1407H1433',PURPLE_INK,dash='7 7');f.text(1450,1414,'标记依赖或生成反馈',22,color=MUTED)
    f.text(72,1467,'依据：Meta Llama 4 官方介绍、模型卡与 llama-models 0e0b8c5 参考实现  ·  推理架构示意  ·  2026-10-04',19,color=MUTED)
    f.text(2325,1467,'01 / 02',21,600,MUTED,'end')
    f.save('llama4-architecture-overview.svg')

def details():
    f=Figure(2400,1850,'Llama 4：Decoder、Attention 与 MoE 展开','放大展示 pre-norm 残差层、GQA 注意力、RoPE/NoPE 层类型及带输入门控的 Top-1 路由专家。')
    f.header('LLAMA 4  /  INSIDE THE DECODER  /  02','从一层 Decoder，看懂 Attention 与 MoE','模块负责什么 · 数据怎样流动 · 参考实现如何调用')
    # 3 columns, actual nested computation.
    f.label(80,298,'A','单层 Transformer','PRE-NORM + RESIDUAL')
    f.label(740,298,'B','注意力展开','GQA + iRoPE + KV CACHE')
    f.label(1490,298,'C','MoE 专家路由','TOP-1 ROUTING + SHARED EXPERT')
    # Residual decoder left
    f.text(332,393,'输入隐状态 x',25,500,anchor='middle')
    f.path('M332 414V455')
    f.box(160,455,345,104,'RMSNorm','稳定注意力输入的数值尺度','norm',BLUE,BLUE_INK,ts=28)
    f.path('M332 559V609')
    f.box(160,609,345,126,'Self-Attention','从历史图文 token 聚合信息','network',BLUE,BLUE_INK,code='Attention.forward',ts=27)
    f.path('M332 735V789');f.circle(332,810,21)
    f.path('M332 430H100V810H311',arrow=True)
    f.text(83,643,'残',21,color=MUTED,anchor='middle');f.text(83,676,'差',21,color=MUTED,anchor='middle')
    f.path('M332 831V881')
    f.box(160,881,345,104,'RMSNorm','稳定前馈网络输入的数值尺度','norm',PURPLE,PURPLE_INK,ts=28)
    f.path('M332 985V1035')
    f.box(160,1035,345,134,'Dense FFN / MoE',['非线性特征变换','按层配置选择对应实现'],'expert',PURPLE,PURPLE_INK,ts=27)
    f.path('M332 1169V1229');f.circle(332,1250,21)
    f.path('M332 857H575V1250H353')
    f.text(592,1080,'残',21,color=MUTED);f.text(592,1113,'差',21,color=MUTED)
    f.path('M332 1271V1320');f.text(332,1360,'输出到下一层',26,500,anchor='middle')
    f.text(100,1440,'h = x + Attention(RMSNorm(x))',22,500,BLUE_INK,family='Menlo, monospace')
    f.text(100,1481,'y = h + FFN_or_MoE(RMSNorm(h))',22,500,PURPLE_INK,family='Menlo, monospace')
    # Attention column.
    f.box(740,375,630,110,'Q / K / V 投影','a = RMSNorm(x)；wq(a), wk(a), wv(a)','embed',BLUE,BLUE_INK,ts=29)
    f.path('M1055 485V527')
    f.rect(740,527,630,255,'#F3F7FC',stroke='#CAD8E9',r=18)
    f.text(765,570,'按层选择位置编码与注意力范围',27,600)
    f.rect(765,595,277,159,BLUE,stroke='none',r=13)
    f.text(786,632,'RoPE 层',25,600,BLUE_INK)
    f.lines(786,670,['Q/K 旋转位置编码','可选 QK Norm','配置分块因果注意力'],22,30)
    f.rect(1063,595,280,159,PURPLE,stroke='none',r=13)
    f.text(1084,632,'NoPE 层',25,600,PURPLE_INK)
    f.lines(1084,670,['不使用旋转位置编码','使用全局因果注意力','可选长上下文温度缩放'],22,30)
    f.path('M1055 782V824')
    f.box(740,824,630,126,'KV Cache + GQA',['缓存历史 K/V，避免重新计算','多组 Q 头共享较少的 K/V 头'],'cache',BLUE,BLUE_INK,ts=29)
    f.path('M1055 950V992')
    f.box(740,992,630,128,'因果注意力 + 输出投影',['SDPA(Q, K, V, mask) → wo','融合可访问的历史信息，回到语言隐藏维度'],'network',BLUE,BLUE_INK,ts=29)
    f.path('M1055 1120V1167');f.text(1055,1204,'Attention 输出 → 残差相加',25,500,anchor='middle')
    # Attention mask key visual
    f.text(741,1280,'iRoPE：在层之间交错两类注意力',25,600)
    def maskgrid(x,y,local):
        for row in range(8):
            for col in range(8):
                active=col<=row and (not local or row//4==col//4)
                f.rect(x+col*18,y+row*18,15,15,BLUE_INK if active else '#E8EDF3',stroke='none',r=2)
    maskgrid(748,1310,True);maskgrid(1080,1310,False)
    f.lines(910,1345,['分块因果','局部块内可见','非滑动窗口'],20,29)
    f.lines(1240,1345,['全局因果','看所有历史','仍屏蔽未来'],20,29)
    f.text(744,1491,'注意力类型与 FFN 类型分别由配置决定。',22,color=MUTED)
    # MoE column: split, router, gating and shared branch.
    f.rect(1660,375,470,77,PURPLE,stroke='none',r=15)
    f.text(1895,425,'输入 z = RMSNorm(h)',28,600,anchor='middle')
    f.path('M1895 452V480',arrow=False)
    f.path('M1895 480H1648V530')
    f.path('M1895 480H2100V530')
    f.box(1490,530,315,168,'共享专家',['每个 token 都经过','SwiGLU 变换'],'expert',GREEN,GREEN_INK,code='shared_expert(z)',ts=28)
    f.box(1880,530,450,168,'Router',['logits = z · Wrouter','Top-1 选中路由专家 e'],'network',PURPLE,PURPLE_INK,code='matmul → topk → sigmoid',ts=29)
    f.path('M2100 698V742')
    f.text(2120,729,'e / gₑ',22,500,PURPLE_INK)
    f.rect(1880,742,450,106,PEACH,stroke='none',r=14)
    f.text(1905,783,'输入门控：z′ = sigmoid(logitₑ) · z',24,600)
    f.text(1905,821,'先缩放输入，再送入选中的专家',23,color=MUTED)
    f.path('M2100 480H2360V795H2330')
    f.text(2339,774,'z',24,500,PURPLE_INK)
    f.path('M2100 848V897')
    f.rect(1850,897,480,280,'#F7F5FC',stroke='#C6BBDD',r=18,dash='7 7')
    f.text(1874,938,'路由专家池 · E 个独立 SwiGLU',24,600)
    for x,t,active in [(1874,'专家 1',False),(2022,'专家 e',True),(2170,'专家 E',False)]:
        f.rect(x,978,126,116,PURPLE if active else '#EEF0F4',stroke=PURPLE_INK if active else '#D8DFE8',r=13,sw=2.2 if active else 1)
        f.icon('expert',x+44,990,PURPLE_INK if active else '#98A3B4',36)
        f.text(x+63,1064,t,24,600,PURPLE_INK if active else MUTED,'middle')
    f.text(2090,1139,'仅 1 个被选中 · E = 16 / 128',23,500,PURPLE_INK,'middle')
    f.path('M1648 698V1225H1874')
    f.path('M2090 1177V1225H1916')
    f.circle(1895,1225,21)
    f.path('M1895 1246V1298')
    f.text(1895,1340,'共享输出 + 路由专家输出',27,600,anchor='middle')
    f.rect(1490,1380,840,145,'#F3F5F9',stroke='none',r=16)
    f.text(1516,1427,'MoE(z) = Shared(z) + Expertₑ(gₑ · z)',27,600,PURPLE_INK)
    f.text(1516,1468,'gₑ = sigmoid(logitₑ)  ·  门控位置依照 Meta 参考实现',23,color=MUTED)
    f.text(1516,1500,'共享专家与路由专家并行；两条分支的输出相加。',22,color=MUTED)
    # Bottom detailed runtime legend.
    f.rect(70,1580,2260,161,'#EDF2F7',stroke='none',r=18)
    f.text(98,1624,'源码调用',25,650)
    f.text(272,1624,'Transformer.forward → layers[i].forward → attention(attention_norm(x)) → feed_forward(ffn_norm(h))',23,500,INK,family='Menlo, PingFang SC, sans-serif')
    f.lines(100,1670,['Dense 层调用 FeedForward；MoE 层调用 MoE.forward。共享专家和路由专家的基本非线性都是 SwiGLU。','图示为计算语义；层数、位置编码间隔、分块大小与 MoE 分布，以 checkpoint 配置为准。'],23,36)
    f.text(72,1797,'依据：Meta llama-models 0e0b8c5  ·  model.py / moe.py / ffn.py  ·  掩码为概念示意；参考实现的解码细节见配套说明',19,color=MUTED)
    f.text(2325,1797,'02 / 02',21,600,MUTED,'end')
    f.save('llama4-decoder-attention-moe.svg')

if __name__ == '__main__':
    overview()
    details()
    print('Generated two editable SVG diagrams.')
