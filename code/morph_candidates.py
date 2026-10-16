"""Category-specific morphological candidates for a fixed interval.

We analyze the input with UniDic (fugashi), find the predicate chain that ends the interval,
and re-realize it with one category feature changed (tense, aspect, voice, causation,
benefaction, transitivity, or politeness). Lexical categories use small substitution tables
(connectives, honorific nouns, demonstrative series). The candidates only enter the pool;
the generators score and the selector chooses among them.
"""
import functools
import itertools

import fugashi

# ------------------------------------------------------------------ conjugation
GODAN_ROWS = {
    'カ': 'かきくけこ', 'ガ': 'がぎぐげご', 'サ': 'さしすせそ', 'タ': 'たちつてと', 'ナ': 'なにぬねの',
    'バ': 'ばびぶべぼ', 'マ': 'まみむめも', 'ラ': 'らりるれろ', 'ワア': 'わいうえお'}
GODAN_TE = {'カ': ('いて', 'いた'), 'ガ': ('いで', 'いだ'), 'サ': ('して', 'した'), 'タ': ('って', 'った'),
            'ナ': ('んで', 'んだ'), 'バ': ('んで', 'んだ'), 'マ': ('んで', 'んだ'), 'ラ': ('って', 'った'),
            'ワア': ('って', 'った')}
IKU = {'行く', '往く', 'いく'}


def vtype(lemma, ctype):
    if ctype.startswith('五段'):
        row = ctype.split('-')[1].replace('行', '')
        if row.startswith('ワ') or row == 'ア':
            row = 'ワア'
        return ('godan', row) if row in GODAN_ROWS else None
    if ctype.startswith('上一段') or ctype.startswith('下一段'):
        return ('ichidan', None)
    if ctype.startswith('サ行変格') or lemma.endswith('する'):
        return ('suru', None)
    if ctype.startswith('カ行変格') or lemma in ('来る', 'くる'):
        return ('kuru', None)
    return None


def conj(lemma, vt, form):
    """form in mizen, renyo, te, ta, dict for a verb lemma of verb type vt."""
    kind, row = vt
    if kind == 'godan':
        stem, rowchars = lemma[:-1], GODAN_ROWS[row]
        if form == 'dict':
            return lemma
        if form == 'mizen':
            return stem + rowchars[0]
        if form == 'renyo':
            return stem + rowchars[1]
        te, ta = GODAN_TE[row]
        if lemma in IKU or lemma.endswith('行く'):
            te, ta = 'って', 'った'
        return stem + (te if form == 'te' else ta)
    if kind == 'ichidan':
        stem = lemma[:-1]
        return {'dict': lemma, 'mizen': stem, 'renyo': stem, 'te': stem + 'て', 'ta': stem + 'た'}[form]
    if kind == 'suru':
        pre = lemma[:-2]
        return pre + {'dict': 'する', 'mizen': 'し', 'renyo': 'し', 'te': 'して', 'ta': 'した'}[form]
    if kind == 'kuru':
        pre = lemma[:-2] if lemma.endswith('来る') or lemma.endswith('くる') else ''
        k = '来' if lemma.endswith('来る') else ''
        if k:
            return pre + {'dict': '来る', 'mizen': '来', 'renyo': '来', 'te': '来て', 'ta': '来た'}[form]
        return pre + {'dict': 'くる', 'mizen': 'こ', 'renyo': 'き', 'te': 'きて', 'ta': 'きた'}[form]
    raise ValueError(vt)


def voice_lemma(lemma, vt, voice):
    if voice == 'active':
        return lemma, vt
    kind = vt[0]
    if voice == 'passive':
        if kind == 'godan':
            return conj(lemma, vt, 'mizen') + 'れる', ('ichidan', None)
        if kind == 'ichidan':
            return conj(lemma, vt, 'mizen') + 'られる', ('ichidan', None)
        if kind == 'suru':
            return lemma[:-2] + 'される', ('ichidan', None)
        return (lemma[:-2] + ('来られる' if lemma.endswith('来る') else 'こられる')), ('ichidan', None)
    if voice == 'causative':
        if kind == 'godan':
            return conj(lemma, vt, 'mizen') + 'せる', ('ichidan', None)
        if kind == 'ichidan':
            return conj(lemma, vt, 'mizen') + 'させる', ('ichidan', None)
        if kind == 'suru':
            return lemma[:-2] + 'させる', ('ichidan', None)
        return (lemma[:-2] + ('来させる' if lemma.endswith('来る') else 'こさせる')), ('ichidan', None)
    raise ValueError(voice)


AUX_VERB = {'くれる': ('ichidan', None), 'もらう': ('godan', 'ワア'), 'あげる': ('ichidan', None),
            'いる': ('ichidan', None), 'くる': ('kuru', None), 'しまう': ('godan', 'ワア'),
            'おく': ('godan', 'カ'), 'ある': ('godan', 'ラ')}


def realize(lemma, vt, voice='active', benef=None, aspect=None, polite=False, past=False):
    lem, t = voice_lemma(lemma, vt, voice)
    for aux in [benef, aspect]:
        if aux:
            lem, t = conj(lem, t, 'te') + aux, AUX_VERB[aux]
    if polite:
        return conj(lem, t, 'renyo') + ('ました' if past else 'ます')
    return conj(lem, t, 'ta' if past else 'dict')


# ------------------------------------------------------------------ lexicons
TRANSITIVITY_PAIRS = [
    ('開く', '開ける'), ('閉まる', '閉める'), ('始まる', '始める'), ('終わる', '終える'), ('入る', '入れる'),
    ('出る', '出す'), ('決まる', '決める'), ('変わる', '変える'), ('続く', '続ける'), ('増える', '増やす'),
    ('減る', '減らす'), ('上がる', '上げる'), ('下がる', '下げる'), ('付く', '付ける'), ('集まる', '集める'),
    ('止まる', '止める'), ('起きる', '起こす'), ('落ちる', '落とす'), ('壊れる', '壊す'), ('直る', '直す'),
    ('治る', '治す'), ('残る', '残す'), ('戻る', '戻す'), ('回る', '回す'), ('流れる', '流す'),
    ('育つ', '育てる'), ('立つ', '立てる'), ('建つ', '建てる'), ('並ぶ', '並べる'), ('届く', '届ける'),
    ('変わる', '変える'), ('伝わる', '伝える'), ('広がる', '広げる'), ('広まる', '広める'), ('高まる', '高める'),
    ('深まる', '深める'), ('強まる', '強める'), ('弱まる', '弱める'), ('進む', '進める'), ('見つかる', '見つける'),
    ('助かる', '助ける'), ('乗る', '乗せる'), ('寝る', '寝かせる'), ('消える', '消す'), ('切れる', '切る'),
    ('割れる', '割る'), ('焼ける', '焼く'), ('溶ける', '溶かす'), ('沸く', '沸かす'), ('動く', '動かす'),
    ('生まれる', '生む'), ('冷える', '冷やす'), ('混ざる', '混ぜる'), ('重なる', '重ねる'), ('備わる', '備える'),
    ('伸びる', '伸ばす'), ('延びる', '延ばす'), ('無くなる', '無くす'), ('なくなる', 'なくす'), ('上がる', '挙げる'),
    ('つながる', 'つなげる'), ('繋がる', '繋げる'), ('まとまる', 'まとめる'), ('集まる', '集める'), ('当たる', '当てる'),
    ('受かる', '受ける'), ('助かる', '助ける'), ('離れる', '離す'), ('外れる', '外す'), ('隠れる', '隠す'),
    ('現れる', '現す'), ('表れる', '表す'), ('倒れる', '倒す'), ('汚れる', '汚す'), ('崩れる', '崩す'),
    ('上がる', '上げる'), ('蒸し上がる', '蒸し上げる'), ('出来る', '作る'), ('できる', '作る'),
]
TRANSITIVITY_PAIRS += [tuple(x.split('/')) for x in (
    'あがる/あげる あく/あける しまる/しめる はじまる/はじめる はいる/いれる でる/だす きまる/きめる かわる/かえる '
    'つづく/つづける ふえる/ふやす へる/へらす つく/つける あつまる/あつめる とまる/とめる おきる/おこす おちる/おとす '
    'こわれる/こわす なおる/なおす のこる/のこす もどる/もどす まわる/まわす ながれる/ながす そだつ/そだてる たつ/たてる '
    'ならぶ/ならべる とどく/とどける つたわる/つたえる ひろがる/ひろげる ひろまる/ひろめる すすむ/すすめる '
    'みつかる/みつける たすかる/たすける きえる/けす なくなる/なくす うごく/うごかす ひえる/ひやす まざる/まぜる '
    'かさなる/かさねる そなわる/そなえる のびる/のばす つながる/つなげる まとまる/まとめる あたる/あてる '
    'はなれる/はなす はずれる/はずす かくれる/かくす たおれる/たおす よごれる/よごす くずれる/くずす').split()]
PAIR = {}
for a, b in TRANSITIVITY_PAIRS:
    PAIR.setdefault(a, set()).add(b)
    PAIR.setdefault(b, set()).add(a)

SUBSTITUTIONS = dict(
    style=[('けど', 'が'), ('けど', 'けれども'), ('けれど', 'けれども'), ('でも', 'しかし'), ('だから', 'そのため'),
           ('ですから', 'ので'), ('から', 'ので'), ('なんか', 'など'), ('なんて', 'など'), ('いっぱい', 'たくさん'),
           ('一杯', 'たくさん'), ('ちょっと', '少し'), ('すごく', 'とても'), ('じゃない', 'ではない'),
           ('じゃありません', 'ではありません'), ('しょっちゅう', 'たびたび'), ('やっぱり', 'やはり'), ('でしょ', 'でしょう'),
           ('って', 'という'), ('とか', 'など'), ('もう', 'すでに'), ('だ', 'である'), ('です', 'である')],
    honorific=[('たち', '方'), ('達', '方'), ('さん', '様'), ('家族', 'ご家族'), ('先生たち', '先生方'),
               ('教授たち', '先生方'), ('客さん', 'お客'), ('いる', 'いらっしゃる'), ('言う', 'おっしゃる'),
               ('くれた', 'くださった'), ('くれました', 'くださいました'), ('もらう', 'いただく'),
               ('もらった', 'いただいた'), ('もらいました', 'いただきました'), ('行く', '参る'), ('する', 'いたす')],
    sentence_connection=[('。', '、'), ('ので', 'から'), ('から', 'ので'), ('けど', 'が'), ('が、', 'ので、')],
)
DEMONSTRATIVES = ['これ', 'それ', 'あれ', 'この', 'その', 'あの', 'ここ', 'そこ', 'あそこ', 'こんな', 'そんな', 'あんな',
                  'こう', 'そう', 'ああ', 'こちら', 'そちら', 'あちら', 'このような', 'そのような', 'あのような']


@functools.lru_cache(maxsize=1)
def tagger():
    return fugashi.Tagger()


def tokens(text):
    out, pos = [], 0
    for w in tagger()(text):
        start = text.find(w.surface, pos)
        if start < 0:
            continue
        end = start + len(w.surface)
        f = w.feature
        out.append(dict(surface=w.surface, start=start, end=end, pos1=f.pos1, pos2=f.pos2,
                        ctype=f.cType or '', cform=f.cForm or '', lemma=f.orthBase or w.surface))
        pos = end
    return out


def predicate(text, s, e):
    """Return (prefix, lemma, vtype, features, suffix) for the predicate chain ending the interval."""
    toks = [t for t in tokens(text) if t['start'] >= s and t['end'] <= e]
    if not toks:
        return None
    tail = ''
    while toks and toks[-1]['pos1'] in ('補助記号', '記号'):
        tail = toks[-1]['surface'] + tail
        toks = toks[:-1]
    for i, t in enumerate(toks):
        if t['pos1'] != '動詞':
            continue
        rest = toks[i + 1:]
        if all(r['pos1'] in ('助動詞',) or (r['pos1'] == '助詞' and r['pos2'] == '接続助詞' and r['surface'] in ('て', 'で'))
               or (r['pos1'] == '動詞' and r['pos2'] == '非自立可能') for r in rest):
            lemma, vt = t['lemma'], vtype(t['lemma'], t['ctype'])
            prefix = text[s:t['start']]
            if vt and vt[0] == 'suru' and i > 0 and toks[i - 1]['pos1'] == '名詞' and t['lemma'] == '為る':
                lemma, vt = 'する', ('suru', None)
            if t['lemma'] in ('為る',):
                lemma, vt = 'する', ('suru', None)
            if t['lemma'] in ('来る',) and t['surface'][0] != '来':
                lemma = 'くる'
            if vt is None:
                return None
            lem = [r['lemma'] for r in rest]
            after_te = [r['lemma'] for k, r in enumerate(rest) if k > 0 and rest[k - 1]['surface'] in ('て', 'で')]
            feats = dict(past='た' in lem or 'だ' in lem, polite='ます' in lem,
                         voice='passive' if any(x in lem for x in ('れる', 'られる')) else
                         'causative' if any(x in lem for x in ('せる', 'させる')) else 'active',
                         benef=next((x for x in ('くれる', '呉れる', 'もらう', '貰う', 'あげる', '上げる') if x in after_te), None),
                         aspect=next((x for x in ('いる', '居る', 'くる', '来る', 'しまう', '仕舞う', 'おく', '置く') if x in after_te), None))
            norm_aux = {'呉れる': 'くれる', '貰う': 'もらう', '上げる': 'あげる', '居る': 'いる', '来る': 'くる',
                        '仕舞う': 'しまう', '置く': 'おく'}
            feats['benef'] = norm_aux.get(feats['benef'], feats['benef'])
            feats['aspect'] = norm_aux.get(feats['aspect'], feats['aspect'])
            if feats['voice'] != 'active' and t['ctype'].startswith('五段') is False and t['lemma'].endswith('れる'):
                pass
            return prefix, lemma, vt, feats, tail
        break
    return None


def verb_variants(text, s, e, cat):
    found = predicate(text, s, e)
    if not found:
        return []
    prefix, lemma, vt, f, tail = found
    grid = []
    base = dict(voice=f['voice'], benef=f['benef'], aspect=f['aspect'], polite=f['polite'], past=f['past'])
    if cat == 'tense':
        grid = [dict(base, past=not f['past']), dict(base, past=not f['past'], aspect='いる'),
                dict(base, aspect=None, past=not f['past'])]
    elif cat == 'aspect':
        for asp, past in itertools.product([None, 'いる', 'くる', 'しまう', 'おく'], [False, True]):
            grid.append(dict(base, aspect=asp, past=past))
    elif cat == 'passive':
        grid = [dict(base, voice='passive' if f['voice'] != 'passive' else 'active'),
                dict(base, voice='passive', aspect='いる'), dict(base, voice='passive', past=not f['past'])]
    elif cat == 'causative':
        grid = [dict(base, voice='causative'), dict(base, voice='active'), dict(base, voice='causative', benef='くれる')]
    elif cat == 'benefactive':
        grid = [dict(base, benef=b) for b in (None, 'くれる', 'もらう', 'あげる')]
    elif cat == 'style':
        grid = [dict(base, polite=not f['polite'])]
    lemmas = [lemma]
    if cat == 'transitivity':
        lemmas = sorted(PAIR.get(lemma, []))
        grid = [base, dict(base, aspect='いる')]
        # Compound verbs: swap the second component (e.g. 蒸し上げる -> 蒸し上がる).
        comp = compound_tail(text, s, e)
        if comp:
            head, second = comp
            for alt in sorted(PAIR.get(second, [])):
                lemmas.append(head + alt)
    out = set()
    for lem in lemmas:
        t = vt if lem == lemma else vtype(lem, guess_ctype(lem))
        if t is None:
            continue
        for g in grid:
            try:
                out.add(prefix + realize(lem, t, **g) + tail)
            except (ValueError, KeyError, IndexError):
                continue
            if cat == 'transitivity' and prefix.endswith(('を', 'が')):
                swapped = prefix[:-1] + ('が' if prefix.endswith('を') else 'を')
                try:
                    out.add(swapped + realize(lem, t, **g) + tail)
                except (ValueError, KeyError, IndexError):
                    pass
    return sorted(out)


def compound_tail(text, s, e):
    """(renyo form of the first verb, lemma of the second verb) for V1-V2 compounds in the interval."""
    toks = [t for t in tokens(text) if t['start'] >= s and t['end'] <= e]
    for a, b in zip(toks, toks[1:]):
        if a['pos1'] == '動詞' and b['pos1'] == '動詞' and b['pos2'] == '非自立可能' and b['lemma'] in PAIR:
            return text[a['start']:a['end']], b['lemma']
    return None


def guess_ctype(lemma):
    toks = tokens(lemma)
    verbs = [t for t in toks if t['pos1'] == '動詞']
    if len(verbs) == 1 and verbs[0]['end'] == len(lemma):
        return verbs[0]['ctype']
    if lemma.endswith('る') and lemma[-2:-1] in 'えけげせぜてでねへべぺめれいきぎしじちぢにひびぴみり':
        return '下一段-ラ行'
    return '五段-ラ行'


def lexical_variants(span, cat):
    out = set()
    for a, b in SUBSTITUTIONS.get(cat, []):
        for x, y in ((a, b), (b, a)):
            if x in span:
                out.add(span.replace(x, y, 1))
    if cat == 'demonstrative':
        for d in DEMONSTRATIVES:
            if d in span:
                for k in 'こそあ':
                    alt = k + d[1:] if d[0] in 'こそ' else None
                    if d.startswith('あそこ'):
                        alt = {'こ': 'ここ', 'そ': 'そこ', 'あ': 'あそこ'}[k]
                    elif d[0] == 'あ':
                        alt = k + d[1:] if k != 'あ' else d
                    if alt and alt != d:
                        out.add(span.replace(d, alt, 1))
    if cat == 'honorific':
        for p in ('お', 'ご'):
            out.add(p + span)
    return sorted(out)


def candidates(text, s, e, cat, limit=24):
    span = text[s:e]
    cands = verb_variants(text, s, e, cat) + lexical_variants(span, cat)
    cands = [c for c in dict.fromkeys(cands) if c and c != span and len(c) <= max(44, 4 * len(span))]
    return cands[:limit]
