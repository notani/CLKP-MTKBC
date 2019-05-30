#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mojimoji import han_to_zen
from multiprocessing import Pool
from neologdn import normalize
from pyknp import KNP
import argparse
import logging
import pandas as pd
import sys

verbose = False
logger = None


def init_logger(name='logger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)
    return logger


def get_mrphs(terms):
    """Parse concepts."""
    knp = KNP(jumanpp=True)

    return {term: knp.parse(han_to_zen(normalize(term.replace('_', '')))).mrph_list()
            for term in terms}


def get_mrphs_parallel(terms, n_jobs):
    """Parse concepnts parallelly."""
    pool = Pool(processes=n_jobs)
    n_terms_per_worker = len(terms) // n_jobs
    if len(terms) % n_jobs > 0:
        n_terms_per_worker += 1
    if verbose:
        logger.info('{} terms per worker'.format(n_terms_per_worker))
    terms_splited = [terms[i*n_terms_per_worker:(i+1)*n_terms_per_worker]
                     for i in range(n_jobs)]
    results = pool.map(get_mrphs, terms_splited)
    return {term: mrph for dic in results for term, mrph in dic.items()}


def get_surfaceTexts(_facts, n_jobs=2):
    """Generate surfaceTexts."""
    facts = _facts.copy()
    facts.loc[:, '_start'] = facts['start'].apply(lambda s: s.split('/')[3])
    facts.loc[:, '_end'] = facts['end'].apply(lambda s: s.split('/')[3])
    facts.loc[:, '_rel'] = facts['rel'].apply(lambda s: s.lower())
    facts.loc[:, 'text'] = ''

    terms = list(facts['_start'].unique())
    terms.extend(facts['_end'].unique())
    if n_jobs > 1:
        mrphs = get_mrphs_parallel(terms, n_jobs)
    else:
        mrphs = get_mrphs(terms)

    func = {
        '/r/antonym': get_surface_antonym,
        '/r/atlocation': get_surface_atlocation,
        '/r/capableof': get_surface_capableof,
        '/r/causes': get_surface_causes,
        '/r/causesdesire': get_surface_causesdesire,
        '/r/createdby': get_surface_createdby,
        '/r/definedas': get_surface_definedas,
        '/r/derivedfrom': get_surface_derivedfrom,
        '/r/desires': get_surface_desires,
        '/r/distinctfrom': get_surface_distinctfrom,
        '/r/entails': get_surface_entails,
        '/r/formof': get_surface_formof,
        '/r/etymologicallyrelatedto': get_surface_etymologicallyrelatedto,
        '/r/hasa': get_surface_hasa,
        '/r/hascontext': get_surface_hascontext,
        '/r/hasfirstsubevent': get_surface_hasfirstsubevent,
        '/r/haslastsubevent': get_surface_haslastsubevent,
        '/r/hasprerequisite': get_surface_hasprerequisite,
        '/r/hasproperty': get_surface_hasproperty,
        '/r/instanceof': get_surface_instanceof,
        '/r/isa': get_surface_isa,
        '/r/locatednear': get_surface_locatednear,
        '/r/madeof': get_surface_madeof,
        '/r/mannerof': get_surface_mannerof,
        '/r/motivatedbygoal': get_surface_motivatedbygoal,
        '/r/partof': get_surface_partof,
        '/r/receivesaction': get_surface_receivesaction,
        '/r/relatedto': get_surface_relatedto,
        '/r/similarto': get_surface_similarto,
        '/r/symbolof': get_surface_symbolof,
        '/r/synonym': get_surface_synonym,
        '/r/usedfor': get_surface_usedfor,
    }
    for idx, row in facts.iterrows():
        _mrphs = {'start': mrphs[row['_start']],
                  'end': mrphs[row['_end']]}
        f = func[row['_rel']]
        facts.loc[idx, 'text'] = f(_mrphs).format(start=row['_start'],
                                                  end=row['_end'])
        facts.loc[idx, 'text'] = facts.ix[idx, 'text'].replace('_', '')

    return facts[['rel', 'start', 'end', 'text']]


# Conditions

def get_category(mrph):
    """Get category of mrph given by JUMAN++."""
    imis = mrph.imis.split()
    for item in imis:
        if item.startswith('カテゴリ'):
            return item.split(':')[-1]
    raise ValueError('No category for ' + mrph.midasi)


def is_category_abstract(mrphs):
    """Return True if an argument is not a physical thing (e.g. festival.)"""
    try:
        mrph = [mrph for mrph in mrphs if mrph.bunrui != '名詞性名詞接尾辞'][-1]
    except IndexError:
        return False
    try:
        return get_category(mrphs[-1]) == '抽象物'
    except ValueError:
        return False


def is_creature(mrphs):
    """Return True if an argument is creature."""
    try:
        # Filter out suffixes e.g. 熊*さん*
        mrph = [mrph for mrph in mrphs if mrph.bunrui != '名詞性名詞接尾辞'][-1]
    except IndexError:
        return False

    try:
        categories = get_category(mrph).split(';')
    except ValueError:
        return False

    creature = set(['人', '動物'])
    flag = False
    for category in categories:
        if category.startswith('人工物'):
            return False
        if category in creature:
            flag = True
    return flag


def is_vp(mrphs):
    """Return true if an argument is a verb phrase.

    怒る(動詞)
    怒ら(動詞) せる(動詞性接尾辞)
    """
    is_verb = mrphs[-1].hinsi == '動詞'
    is_verb_suffix = (len(mrphs) >= 2) and \
                     (mrphs[-1].bunrui == '動詞性接尾辞')
    return is_verb or is_verb_suffix


def is_np(mrphs):
    """Return true if an argument is a noun phrase.

    リンゴ(名詞)
    クマ(名詞) さん(名詞性名詞接尾辞)
    食べ(名詞[KNP]/動詞[JUMAN]) 放題(名詞性述語接尾辞)
    100(数詞) メートル(名詞性名詞助数辞)
    10(数詞) 時(名詞性名詞助数辞) 半(名詞性特殊接尾辞)
    """
    is_noun = mrphs[-1].hinsi == '名詞'
    is_noun_suffix = (len(mrphs) >= 2) and \
                     (mrphs[-1].hinsi == '接尾辞') and \
                     (mrphs[-1].bunrui.startswith('名詞性'))
    return is_noun or is_noun_suffix


def is_ap(mrphs):
    """Return true if an argument is a adjective phrase.

    きれい(形容詞)
    買い(動詞) やすい(形容詞性述語接尾辞)
    庶民(名詞) 的な(形容詞性名詞接尾辞)
    """
    is_adj = mrphs[-1].hinsi == '形容詞'
    is_adj_suffix = (len(mrphs) >= 2) and \
                    (mrphs[-1].hinsi == '接尾辞') and \
                    (mrphs[-1].bunrui.startswith('形容詞性'))
    return is_adj or is_adj_suffix


def is_sahen(mrphs):
    """Return true if an argument is a sahen-noun (サ変名詞.)"""
    return mrphs[-1].bunrui == 'サ変名詞'


def is_vpkoto(mrphs, vp=False):
    """Return True if an argument is a VP + 'koto (こと)', which is a noun phrase."""
    if len(mrphs) == 1:
        return False
    if mrphs[-1].hinsi != '名詞' or mrphs[-1].yomi != 'こと':
        return False
    if vp:  # Already known to be a VP
        return True
    return is_vp(mrphs)


def get_vpkoto(mrphs, label):
    """Return a VP + koto (こと) form of an argument.

    Assume a given arg is a VP or sahen-noun.
    """
    surface = label
    if is_sahen(mrphs):  # e.g. 運転
        surface += 'すること'
    elif is_vpkoto(mrphs, vp=True):  # e.g. 走ること
        pass
    else:  # e.g. 走る
        surface += 'こと'
    return surface


def is_apkoto(mrphs, ap=False):
    """Return True if the given mrphs are AP + 'こと'
    """
    if len(mrphs) == 1:
        return False
    if mrphs[-1].hinsi != '名詞' or mrphs[-1].yomi != 'こと':
        return False
    if ap:  # Already known to be a AP
        return True
    return is_ap(term[:-1], mrphs=mrphs)


def get_apkoto(mrphs, label):
    """Return AP + koto

    Assume a given arg is a AP

    - ～なこと
      イ形容詞アウオ段
      イ形容詞イ段
      イ形容詞イ段特殊
    - ～こと
      ナ形容詞
      ナ形容詞特殊
      ナノ形容詞
      タル形容詞
    """
    surface = label
    if mrphs[-1].katuyou1[0] == 'ナ':  # ナ形容詞 (e.g. きれい)
        surface += 'な'
    return surface + 'こと'


def get_surface_antonym(mrphs):
    """Return surfaceText for /r/Antonym

    Returns:
    {start}は{end}の反対である。

    Constraints:
    - start=NP
    - end=NP

    Ref:
    (ja)
    /r/Antonym  [[start]]は[[end]]の反義語である        0

    (en)
    /r/Antonym  [[start]] is the opposite of [[end]]    4556
    """
    start = ''.join(m.midasi for m in mrphs['start'])
    end = ''.join(m.midasi for m in mrphs['end'])
    return '{start}は{end}の反対である。'.format(start=start, end=end)


def get_surface_antonym_test():
    knp = KNP(jumanpp=True)
    start, end = '需要', '供給'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_antonym(mrphs).format(start=start, end=end)
    assert '需要は供給の反対である。' == output, output


def get_surface_atlocation(mrphs):
    """Return surfaceText for /r/AtLocation

    Returns:
    {end}で{start}を見ることがある。 (if end is abstructraction)
    {start}は{end}にいる。 (if start is a creature)
    {start}は{end}にある。 (otherwise)

    Constraints:
    - start=NP
    - end=NP

    Ref:
    (ja)
    /r/AtLocation	[[end]]に行くと[[start]]を見ることができる．	7029
    /r/AtLocation	[[start]]は[[end]]にある．	2371
    /r/AtLocation	[[start]]は[[end]]にいる．	1411

    (en)
    /r/AtLocation	You are likely to find [[start]] in [[end]]	11391
    /r/AtLocation	Something you find in [[end]] is [[start]]	2250
    /r/AtLocation	Something you find at [[end]] is [[start]]	2228
    /r/AtLocation	Something you find on [[end]] is [[start]]	765
    /r/AtLocation	You are likely to find [[start]] in [[end]].	456
    /r/AtLocation	Somewhere [[start]] can be is in [[end]]	255
    /r/AtLocation	Something you find under [[end]] is [[start]]	183
    /r/AtLocation	Somewhere [[start]] can be is on [[end]]	96
    /r/AtLocation	Somewhere [[start]] can be is [[end]]	33
    /r/AtLocation	You are likely to find [[start]] on [[end]]	15
    /r/AtLocation	In [[end]], you can [[start]]	7
    /r/AtLocation	You are likely to find [[start]] at [[end]]	6
    /r/AtLocation	[[start]] can be at [[end]]	4
    /r/AtLocation	Something you find in [[end]] is [[start]]]	1
    /r/AtLocation	You are likely to find [[start]] near [[end]]
    """

    if is_category_abstract(mrphs['end']):  # e.g. お祭り
        return '{end}で{start}を見ることがある。'

    surface = '{start}は{end}に'

    if is_creature(mrphs['start']):  # e.g. お父さん
        surface += 'いる。'
    else:  # e.g. テーブル
        surface += 'ある。'

    return surface


def get_surface_atlocation_test():
    knp = KNP(jumanpp=True)
    start, end = 'お父さん', '家'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_atlocation(mrphs).format(start=start, end=end)
    assert 'お父さんは家にいる。' == output, output

    start, end = 'テーブル', '家'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_atlocation(mrphs).format(start=start, end=end)
    assert 'テーブルは家にある。' == output, output

    start, end = '屋台', 'お祭り'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_atlocation(mrphs).format(start=start, end=end)
    assert 'お祭りで屋台を見ることがある。' == output, output


def get_surface_capableof(mrphs):
    """Return surfaceText for /r/CapableOf

    Returns:
    {start}は{end}ことができる。

    Constraints:
    - start=NP
    - end=VP

    Ref:
    (ja)
    /r/CapableOf	[[start]]は[[end]]能力を持っている．	842
    /r/CapableOf	[[start]]ができることの一つに[[end]]がある．	548
    /r/CapableOf	[[start]]は[[end]]ことができる．	459
    /r/CapableOf	[[start]]の能力の一つに[[end]]がある．	339

    (en)
    /r/CapableOf	[[start]] can [[end]]	8958
    /r/CapableOf	An activity [[start]] can do is [[end]]	426
    /r/CapableOf	[[start]] can [[end]].	233
    /r/CapableOf	[[start]] may [[end]]	160
    /r/CapableOf	[[start]] often [[end]]	25
    /r/CapableOf	[[start]] sometimes [[end]]	9
    """
    surface = '{start}は{end}'

    if is_sahen(mrphs['end']):  # e.g. 運転
        surface += 'すること'
    elif is_vpkoto(mrphs['end'], vp=True):  # e.g. 走ること
        pass
    else:  # e.g. 走る
        surface += 'こと'

    surface += 'ができる。'

    return surface


def get_surface_capableof_test():
    knp = KNP(jumanpp=True)
    start, end = 'お父さん', '運転'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_capableof(mrphs).format(start=start, end=end)
    assert 'お父さんは運転することができる。' == output, output

    start, end = 'お父さん', '運転する'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_capableof(mrphs).format(start=start, end=end)
    assert 'お父さんは運転することができる。' == output, output

    start, end = 'お父さん', '運転すること'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_capableof(mrphs).format(start=start, end=end)
    assert 'お父さんは運転することができる。' == output, output


def get_surface_causes(mrphs):
    """Return surfaceText for /r/Causes

    Returns:
    {start}は{end}の原因になることがある。

    Constraints:
    - start=VP
    - end=NP/VP

    Ref:
    (ja)
    /r/Causes   {end}の原因には{start}がある．      651
    /r/Causes   {start}による影響の一つに{end}がある．      385
    /r/Causes   {start}の結果に{end}がある．        301
    /r/Causes   {start}はあなたを{end}気持ちにさせる．      124
    /r/Causes   {start}による影響で{end}．  114
    /r/Causes   {start}が原因で{end}ことがある．    86
    /r/Causes   {start}の結果，{end}ことがある．    45

    (en)
    /r/Causes   Sometimes {start} causes {end}      128
    /r/Causes   The effect of {start} is {end}.     121
    /r/Causes   Something that might happen as a consequence of {start} is {end}    47
    /r/Causes   The effect of {start} is {end}      5
    /r/Causes   Sometimes {start} causes you to {end}       2
    """

    surface = get_vpkoto(mrphs['start'], '{start}') + 'は'

    if is_vp(mrphs['end']):
        surface += get_vpkoto(mrphs['end'], '{end}')
    else:  # NP
        surface += '{end}'

    surface += 'の原因になることがある。'

    return surface


def get_surface_causes_test():
    knp = KNP(jumanpp=True)
    start, end = '運転', '交通事故'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_causes(mrphs).format(start=start, end=end)
    assert '運転することは交通事故の原因になることがある。' == output, output

    start, end = 'お酒を飲む', '肥満'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_causes(mrphs).format(start=start, end=end)
    assert 'お酒を飲むことは肥満の原因になることがある。' == output, output

    start, end = '走る', '転ぶ'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_causes(mrphs).format(start=start, end=end)
    assert '走ることは転ぶことの原因になることがある。' == output, output

    start, end = 'エキササイズ', '汗をかく'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_causes(mrphs).format(start=start, end=end)
    assert 'エキササイズすることは汗をかくことの原因になることがある。' == output, output


def get_surface_causesdesire(mrphs):
    """Return surfaceText for /r/CausesDesire

    Returns:
    {start}はあなたを{end}気にさせることがある。

    Constraints:
    - start=NP
    - end=VP

    Ref:
    (ja)
    /r/CausesDesire     {start}はあなたが{end}ことをやる気にさせる．        580

    (en)
    /r/CausesDesire     {start} would make you want to {end}        1800
    """
    start = ''.join(mrph.midasi for mrph in mrphs['start'])
    surface = '{start}はあなたを'.format(start=start)

    if is_sahen(mrphs['end']):
        surface += ''.join(mrph.midasi for mrph in mrphs['end']) + 'する'
    elif is_vpkoto(mrphs['end'], vp=True):
        surface += ''.join(mrph.midasi for mrph in mrphs['end'][:-1])
    else:
        surface += ''.join(mrph.midasi for mrph in mrphs['end'])

    surface += '気にさせることがある。'

    return surface


def get_surface_causesdesire_test():
    kknp = KNP(jumanpp=True)
    start, end = '晴天', '散歩'
    mrphs = {'start': kknp.parse(start).mrph_list(),
             'end': kknp.parse(end).mrph_list()}
    output = get_surface_causesdesire(mrphs).format(start=start, end=end)
    assert '晴天はあなたを散歩する気にさせることがある。' == output, output

    start, end = '肥満', '走る'
    mrphs = {'start': kknp.parse(start).mrph_list(),
             'end': kknp.parse(end).mrph_list()}
    output = get_surface_causesdesire(mrphs).format(start=start, end=end)
    assert '肥満はあなたを走る気にさせることがある。' == output, output

    start, end = '恋愛', 'お洒落すること'
    mrphs = {'start': kknp.parse(start).mrph_list(),
             'end': kknp.parse(end).mrph_list()}
    output = get_surface_causesdesire(mrphs).format(start=start, end=end)
    assert '恋愛はあなたをお洒落する気にさせることがある。' == output, output


def get_surface_createdby(mrphs):
    """Return surfaceText for /r/CreatedBy

    Returns:
    {start}は{end}によって作られる。

    Constraints:
    - start=NP
    - end=VP

    Ref:
    (ja)
    /r/CreatedBy        {start}は{end}によって作られる      0

    (en)
    /r/CreatedBy        {start} is created by {end}.        136
    """
    start = ''.join(m.midasi for m in mrphs['start'])
    end = get_vpkoto(mrphs['end'], ''.join(m.midasi for m in mrphs['end']))
    return '{start}は{end}によって作られる。'.format(start=start, end=end)


def get_surface_createdby_test():
    kknp = KNP(jumanpp=True)
    start, end = 'パン', '焼く'
    mrphs = {'start': kknp.parse(start).mrph_list(),
             'end': kknp.parse(end).mrph_list()}
    output = get_surface_createdby(mrphs).format(start=start, end=end)
    assert 'パンは焼くことによって作られる。' == output, output

    start, end = 'ケーキ', '焼くこと'
    mrphs = {'start': kknp.parse(start).mrph_list(),
             'end': kknp.parse(end).mrph_list()}
    output = get_surface_createdby(mrphs).format(start=start, end=end)
    assert 'ケーキは焼くことによって作られる。' == output, output

    start, end = '小説', '執筆'
    mrphs = {'start': kknp.parse(start).mrph_list(),
             'end': kknp.parse(end).mrph_list()}
    output = get_surface_createdby(mrphs).format(start=start, end=end)
    assert '小説は執筆することによって作られる。' == output, output


def get_surface_definedas(mrphs):
    """Return surfaceText for /r/DefinedAs

    Returns:
    {start}は{end}のことである。

    Constraints:
    - start=NP
    - end=NP

    Ref:
    (ja)
    /r/DefinedAs        {start}とは{end}のことである．      75

    (en)
    /r/DefinedAs    {start} is the {end}        1286
    /r/DefinedAs    {start} is the {end}.       131
    /r/DefinedAs    {start} can be defined as {end}     7
    """
    start = ''.join(m.midasi for m in mrphs['start'])
    end = ''.join(m.midasi for m in mrphs['end'])
    return '{start}は{end}のことである。'.format(start=start, end=end)


def get_surface_definedas_test():
    knp = KNP(jumanpp=True)
    start, end = 'パン', 'ブレッド'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_definedas(mrphs).format(start=start, end=end)
    assert 'パンはブレッドのことである。' == output, output

    start, end = 'ブラッド', '血'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_definedas(mrphs).format(start=start, end=end)
    assert 'ブラッドは血のことである。' == output, output

    start, end = '本', '書籍'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_definedas(mrphs).format(start=start, end=end)
    assert '本は書籍のことである。' == output, output


def get_surface_derivedfrom(mrphs):
    """Return surfaceText for /r/DerivedFrom

    Returns:
    {start}は{end}に由来する。

    Constraints:
    - start=NP
    - end=NP

    Ref:
    (ja)
    /r/DerivedFrom      {start}は{end}に由来する    0

    (en)
    /r/DerivedFrom      {start} is derived from {end}       0
    """
    start = ''.join(m.midasi for m in mrphs['start'])
    end = ''.join(m.midasi for m in mrphs['end'])
    return '{start}は{end}に由来する。'.format(start=start, end=end)


def get_surface_derivedfrom_test():
    knp = KNP(jumanpp=True)
    start, end = '燕尾服', 'ツバメ'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_derivedfrom(mrphs).format(start=start, end=end)
    assert '燕尾服はツバメに由来する。' == output, output


def get_surface_desires(mrphs):
    """Return surfaceText for /r/Desires

    Returns:
    {start}は{end}を望む。

    Constraints:
    - start=NP
    - end=VP

    Ref:
    (ja)
    /r/Desires  {start}は{end}を望む．      612
    /r/Desires  {start}は{end}ことを望む．  92
    /r/Desires  {start}は{end}を欲する．    70
    /r/Desires  {start}は{end}ことを欲する．        36
    /r/Desires  {start}は{end}が欲しい．    31

    (en)
    /r/Desires  {start} wants {end} 937
    /r/Desires  {start} wants to {end}      606
    /r/Desires  {start} like to {end}       19
    """
    start = ''.join(m.midasi for m in mrphs['start'])
    end = get_vpkoto(mrphs['end'], ''.join(m.midasi for m in mrphs['end']))
    return '{start}は{end}を望む。'.format(start=start, end=end)


def get_surface_desires_test():
    knp = KNP(jumanpp=True)
    start, end = '人間', '食べる'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_desires(mrphs).format(start=start, end=end)
    assert '人間は食べることを望む。' == output, output
    start, end = '人間', '長生き'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_desires(mrphs).format(start=start, end=end)
    assert '人間は長生きすることを望む。' == output, output
    start, end = '選手', '勝つこと'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_desires(mrphs).format(start=start, end=end)
    assert '選手は勝つことを望む。' == output, output
    start, end = '大統領', '国の繁栄'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_desires(mrphs).format(start=start, end=end)
    assert '大統領は国の繁栄することを望む。' == output, output  # これは仕方ない


def get_surface_distinctfrom(mrphs):
    """Return surfaceText for /r/DistinctFrom

    Returns:
    {start}は{end}ではない。

    Constraints:
    - start=NP/VP/AP
    - end=NP/VP/AP

    Ref:
    (ja)
    /r/DistinctFrom     {start}は{end}ではない      0

    (en)
    /r/DistinctFrom     {start} is not {end}        1921
    """
    start = ''.join(m.midasi for m in mrphs['start'])
    if is_vp(mrphs['start']):
        start = get_vpkoto(mrphs['start'], start)
    elif is_ap(mrphs['start']):
        start = get_apkoto(mrphs['start'], start)
    end = ''.join(m.midasi for m in mrphs['end'])
    if is_vp(mrphs['end']):
        end = get_vpkoto(mrphs['end'], end)
    elif is_ap(mrphs['end']):
        end = get_apkoto(mrphs['end'], end)
    return '{start}は{end}ではない。'.format(start=start, end=end)


def get_surface_distinctfrom_test():
    knp = KNP(jumanpp=True)
    start, end = 'パン', 'ご飯'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_distinctfrom(mrphs).format(start=start, end=end)
    assert 'パンはご飯ではない。' == output, output
    start, end = '歩く', '走る'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_distinctfrom(mrphs).format(start=start, end=end)
    assert '歩くことは走ることではない。' == output, output
    start, end = 'きれい', '汚い'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_distinctfrom(mrphs).format(start=start, end=end)
    assert 'きれいなことは汚いことではない。' == output, output


def get_surface_entails(mrphs):
    """Return surfaceText for /r/Entails

    Returns:
    {start}は{end}を含意する。

    Constraints:
    - start=NP/VP
    - end=NP/VP

    Ref:
    (ja)
    /r/Entails  [[start]] entails [[end]]       0

    (en)
    /r/Entails  [[start]]は[[end]]を含意する    0
    """
    start = ''.join(m.midasi for m in mrphs['start'])
    if is_vp(mrphs['start']):
        start = get_vpkoto(mrphs['start'], start)
    end = ''.join(m.midasi for m in mrphs['end'])
    if is_vp(mrphs['end']):
        end = get_vpkoto(mrphs['end'], end)
    return '{start}は{end}を含意する。'.format(start=start, end=end)


def get_surface_entails_test():
    knp = KNP(jumanpp=True)
    start, end = '寝る', '目を閉じる'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_entails(mrphs).format(start=start, end=end)
    assert '寝ることは目を閉じることを含意する。' == output, output
    start, end = '睡眠', '目を閉じる'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_entails(mrphs).format(start=start, end=end)
    assert '睡眠は目を閉じることを含意する。' == output, output


def get_surface_etymologicallyrelatedto(mrphs):
    """Return surfaceText for /r/EtymologicallyRelatedTo

    Returns:
    {start}は{end}と語源的に関係している。

    Constraints:
    - start=NP/VP/AP
    - end=NP/VP/AP

    Ref:
    (ja)
    /r/EtymologicallyRelatedTo  [[start]]は[[end]]と語源的な関係を持っている    0

    (en)
    /r/EtymologicallyRelatedTo  [[start]] is etimologically related to [[end]]  0
    """
    start = ''.join(m.midasi for m in mrphs['start'])
    if is_vp(mrphs['start']):
        start = get_vpkoto(mrphs['start'], start)
    elif is_ap(mrphs['start']):
        start = get_apkoto(mrphs['start'], start)
    end = ''.join(m.midasi for m in mrphs['end'])
    if is_vp(mrphs['end']):
        end = get_vpkoto(mrphs['end'], end)
    elif is_ap(mrphs['end']):
        end = get_apkoto(mrphs['end'], end)
    return '{start}は{end}と語源的に関係している。'.format(start=start, end=end)


def get_surface_etymologicallyrelatedto_test():
    knp = KNP(jumanpp=True)
    start, end = '睡眠', '就寝'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_etymologicallyrelatedto(mrphs).format(start=start,
                                                               end=end)
    assert '睡眠は就寝と語源的に関係している。' == output, output
    start, end = '起床', '起きる'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_etymologicallyrelatedto(mrphs).format(start=start,
                                                               end=end)
    assert '起床は起きることと語源的に関係している。' == output, output
    start, end = '美しい', 'きれい'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_etymologicallyrelatedto(mrphs).format(start=start,
                                                               end=end)
    assert '美しいことはきれいなことと語源的に関係している。' == output, output


def get_surface_formof(mrphs):
    """Return surfaceText for /r/FormOf

    Returns:
    {start}は{end}の形態である。

    Constraints:
    - start=NP/VP/AP
    - end=NP/VP/AP

    Ref:
    (ja)
    /r/FormOf	[[start]]は[[end]]の形態である	0

    (en)
    /r/FormOf	[[start]] is a form of [[end]]	0
    """
    start = ''.join(m.midasi for m in mrphs['start'])
    end = ''.join(m.midasi for m in mrphs['end'])
    return '{start}は{end}の形態である。'.format(start=start, end=end)


def get_surface_formof_test():
    knp = KNP(jumanpp=True)
    start, end = '走れ', '走る'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_formof(mrphs).format(start=start, end=end)
    assert '走れは走るの形態である。' == output, output
    start, end = '起床', '起きる'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_formof(mrphs).format(start=start, end=end)
    assert '起床は起きるの形態である。' == output, output
    start, end = 'きれいに', 'きれい'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_formof(mrphs).format(start=start, end=end)
    assert 'きれいにはきれいの形態である。' == output, output


def get_surface_hasa(mrphs):
    """Return surfaceText for /r/HasA

    Returns:
    {start}は{end}を持っている。

    Constraints:
    - start=NP
    - end=NP

    Ref:
    (ja)
    /r/HasA	[[start]]は[[end]]を持っている。	1

    (en)
    /r/HasA     [[start]] has [[end]]   820
    /r/HasA [[start]] contains [[end]]      212
    /r/HasA [[start]] have [[end]]  104
    """
    start = ''.join(m.midasi for m in mrphs['start'])
    end = ''.join(m.midasi for m in mrphs['end'])
    return '{start}は{end}を持っている。'.format(start=start, end=end)


def get_surface_hasa_test():
    knp = KNP(jumanpp=True)
    start, end = '人間', '口'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_hasa(mrphs).format(start=start, end=end)
    assert '人間は口を持っている。' == output, output


def get_surface_hascontext(mrphs):
    """Return surfaceText for /r/HasContext

    Returns:
    {start}は{end}という文脈を持っている。

    Constraints:
    - start=NP/VP/AP
    - end=NP

    Ref:
    (ja)
    /r/HasContext	[[start]]は[[end]]という文脈を持っている	0

    (en)
    /r/HasContext       [[start]] has a context of [[end]]      0
    """
    start = ''.join(m.midasi for m in mrphs['start'])
    if is_vp(mrphs['start']):
        start = get_vpkoto(mrphs['start'], start)
    elif is_ap(mrphs['start']):
        start = get_apkoto(mrphs['start'], start)
    end = ''.join(m.midasi for m in mrphs['end'])
    return '{start}は{end}という文脈を持っている。'.format(start=start, end=end)


def get_surface_hascontext_test():
    knp = KNP(jumanpp=True)
    start, end = 'ボールを蹴ること', 'サッカー'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_hascontext(mrphs).format(start=start, end=end)
    assert 'ボールを蹴ることはサッカーという文脈を持っている。' == output, output
    start, end = '誓いの言葉', '結婚式'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_hascontext(mrphs).format(start=start, end=end)
    assert '誓いの言葉は結婚式という文脈を持っている。' == output, output
    start, end = '錚々たる', 'メンバー'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_hascontext(mrphs).format(start=start, end=end)
    assert '錚々たることはメンバーという文脈を持っている。' == output, output


def get_surface_hasfirstsubevent(mrphs):
    """Return surfaceText for /r/HasFirstSubevent

    Returns:
    あなたが{start}際に最初にすることは{end}である。

    Constraints:
    - start=VP
    - end=NP/VP

    Ref:
    (ja)
    /r/HasFirstSubevent あなたが{start}の際に最初にすることは{end}である．  159
    /r/HasFirstSubevent     あなたが{start}ときに最初にすることは{end}ことである．      125

    (en)
    /r/HasFirstSubevent The first thing you do when you {start} is {end}    1539
    """

    if is_sahen(mrphs['start']):
        start = ''.join(mrph.midasi for mrph in mrphs['start']) + 'する'
    elif is_vpkoto(mrphs['start'], vp=True):
        start = ''.join(mrph.midasi for mrph in mrphs['start'][:-1])
    else:
        start = ''.join(mrph.midasi for mrph in mrphs['start'])

    end = ''.join(m.midasi for m in mrphs['end'])
    if is_vp(mrphs['end']):
        end = get_vpkoto(mrphs['end'], end)

    return 'あなたが{start}際に最初にすることは{end}である。'.format(start=start, end=end)


def get_surface_hasfirstsubevent_test():
    knp = KNP(jumanpp=True)
    start, end = '就寝', 'ベッドに行く'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_hasfirstsubevent(mrphs).format(start=start, end=end)
    assert 'あなたが就寝する際に最初にすることはベッドに行くことである。' == output, output
    start, end = '寝る', 'ベッドに行く'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_hasfirstsubevent(mrphs).format(start=start, end=end)
    assert 'あなたが寝る際に最初にすることはベッドに行くことである。' == output, output


def get_surface_haslastsubevent(mrphs):
    """Return surfaceText for /r/HasLastsubevent

    Returns:
    あなたが{start}際に最後にすることは{end}である。

    Constraints:
    - start=VP
    - end=NP/VP

    Ref:
    (ja)
    /r/HasLastSubevent  あなたが[[start]]の際に最後にすることは[[end]]である．  207
    /r/HasLastSubevent  あなたが[[start]]ときに最後にすることは[[end]]ことである．      101

    (en)
    /r/HasLastSubevent  The last thing you do when you [[start]] is [[end]]    1426
    /r/HasLastSubevent  The last thing you do when you [[end]] is [[end]]      1
    """

    if is_sahen(mrphs['start']):
        start = ''.join(mrph.midasi for mrph in mrphs['start']) + 'する'
    elif is_vpkoto(mrphs['start'], vp=True):
        start = ''.join(mrph.midasi for mrph in mrphs['start'][:-1])
    else:
        start = ''.join(mrph.midasi for mrph in mrphs['start'])

    end = ''.join(m.midasi for m in mrphs['end'])
    if is_vp(mrphs['end']):
        end = get_vpkoto(mrphs['end'], end)

    return 'あなたが{start}際に最後にすることは{end}である。'.format(start=start, end=end)


def get_surface_haslastsubevent_test():
    knp = KNP(jumanpp=True)
    start, end = '就寝', 'ベッドに行く'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_haslastsubevent(mrphs).format(start=start, end=end)
    assert 'あなたが就寝する際に最後にすることはベッドに行くことである。' == output, output
    start, end = '寝る', 'ベッドに行く'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_haslastsubevent(mrphs).format(start=start, end=end)
    assert 'あなたが寝る際に最後にすることはベッドに行くことである。' == output, output


def get_surface_hasprerequisite(mrphs):
    """Return surfaceText for /r/HasPrerequisite

    Returns:
    (start=NP)
    あなたが{start}の際に、それより先にすることは{end}である。
    (start=VP)
    あなたが{start}際に、それより先にすることは{end}である。

    Constraints:
    - start=NP/VP
    - end=NP/VP

    Ref:
    (ja)
    /r/HasPrerequisite  あなたが[[start]]ときに，それより先にすることは[[end]]ことである．      224
    /r/HasPrerequisite  あなたが[[start]]のときに，それより先にすることは[[end]]である．        123

    (en)
    /r/HasPrerequisite  If you want to [[start]] then you should [[end]]        6576
    /r/HasPrerequisite  Something you need to do before you [[start]] is [[end]]        1298
    /r/HasPrerequisite  [[start]] requires [[end]]      158
    /r/HasPrerequisite  If you want to [[end]] then you should [[end]]  2
    """

    if is_sahen(mrphs['start']):
        start = ''.join(mrph.midasi for mrph in mrphs['start']) + 'する'
    elif is_np(mrphs['start']):
        start = ''.join(mrph.midasi for mrph in mrphs['start']) + 'の'
    elif is_vpkoto(mrphs['start'], vp=True):
        start = ''.join(mrph.midasi for mrph in mrphs['start'][:-1])
    else:
        start = ''.join(mrph.midasi for mrph in mrphs['start'])

    end = ''.join(m.midasi for m in mrphs['end'])
    if is_vp(mrphs['end']):
        end = get_vpkoto(mrphs['end'], end)

    return 'あなたが{start}際に、それより先にすることは{end}である。'.format(start=start, end=end)


def get_surface_hasprerequisite_test():
    knp = KNP(jumanpp=True)
    start, end = '就寝', 'ベッドに行く'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_hasprerequisite(mrphs).format(start=start, end=end)
    assert 'あなたが就寝する際に、それより先にすることはベッドに行くことである。' == output, output
    start, end = '病気', 'ベッドに行く'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_hasprerequisite(mrphs).format(start=start, end=end)
    assert 'あなたが病気の際に、それより先にすることはベッドに行くことである。' == output, output


def get_surface_hasproperty(mrphs):
    """Return surfaceText for /r/HasProperty

    Returns:
    {start}は{end}。

    Constraints:
    - start=NP
    - end=AP

    Ref:
    (ja)
    /r/HasProperty      [[start]]は[[end]]という特徴がある．    1346

    (en)
    /r/HasProperty      [[start]] is [[end]]    3258
    /r/HasProperty      [[start]] can be [[end]]        411
    /r/HasProperty      [[start]] are [[end]]   165
    /r/HasProperty      [[start]] is generally [[end]]  62
    """

    start = ''.join(mrph.midasi for mrph in mrphs['start'])
    end = ''.join(m.midasi for m in mrphs['end'])

    return '{start}は{end}。'.format(start=start, end=end)


def get_surface_hasproperty_test():
    knp = KNP(jumanpp=True)
    start, end = 'バラ', '赤い'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_hasproperty(mrphs).format(start=start, end=end)
    assert 'バラは赤い。' == output, output


def get_surface_instanceof(mrphs):
    """Return surfaceText for /r/InstanceOf

    Returns:
    {start}は{end}の一例である。

    Constraints:
    - start=NP
    - end=NP

    Ref:
    (ja)
    /r/InstanceOf       [[start]]は[[end]]の具体例である．      451
    /r/InstanceOf       [[start]]は[[end]]の一例である．        422

    (en)
    /r/InstanceOf       [[start]] is an example of [[end]]      1
    /r/InstanceOf       [[start]] is an instance of [[end]]     1
    """

    start = ''.join(mrph.midasi for mrph in mrphs['start'])
    end = ''.join(m.midasi for m in mrphs['end'])

    return '{start}は{end}の一例である。'.format(start=start, end=end)


def get_surface_instanceof_test():
    knp = KNP(jumanpp=True)
    start, end = 'グーグル', '企業'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_instanceof(mrphs).format(start=start, end=end)
    assert 'グーグルは企業の一例である。' == output, output


def get_surface_isa(mrphs):
    """Return surfaceText for /r/IsA

    Returns:
    {start}は{end}である。

    Constraints:
    - start=NP
    - end=NP

    Ref:
    (ja)
    /r/IsA      [[start]]は[[end]]の一種である．        7728

    (en)
    /r/IsA      [[start]] is [[end]]    11284
    /r/IsA      [[start]] is a kind of [[end]]. 738
    /r/IsA      [[start]] was [[end]]   494
    /r/IsA      [[start]] is a type of [[end]]  415
    /r/IsA      [[start]] is a kind of [[end]]  337
    /r/IsA      [[start]] is a type of [[end]]. 314
    /r/IsA      [[start]] are [[end]]   141
    /r/IsA      [[start]] is a [[end]]  53
    /r/IsA      A [[start]] is a [[end]]        1
    """

    start = ''.join(mrph.midasi for mrph in mrphs['start'])
    end = ''.join(m.midasi for m in mrphs['end'])

    return '{start}は{end}である。'.format(start=start, end=end)


def get_surface_isa_test():
    knp = KNP(jumanpp=True)
    start, end = 'グーグル', '企業'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_isa(mrphs).format(start=start, end=end)
    assert 'グーグルは企業である。' == output, output


def get_surface_locatednear(mrphs):
    """Return surfaceText for /r/LocatedNear

    Returns:
    通常{start}は{end}の近くにある/いる。

    Constraints:
    - start=NP
    - end=NP

    Ref:
    (ja)
    /r/LocatedNear      [[start]]は通常[[end]]の近くにある      0

    (en)
    /r/LocatedNear      [[start]] is typically near [[end]]     41
    """

    start = ''.join(mrph.midasi for mrph in mrphs['start'])
    end = ''.join(m.midasi for m in mrphs['end'])

    surface = '通常{start}は{end}の近くに'
    if is_creature(mrphs['start']):  # e.g. お父さん
        surface += 'いる。'
    else:  # e.g. テーブル
        surface += 'ある。'

    return surface.format(start=start, end=end)


def get_surface_locatednear_test():
    knp = KNP(jumanpp=True)
    start, end = 'テーブル', '椅子'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_locatednear(mrphs).format(start=start, end=end)
    assert '通常テーブルは椅子の近くにある。' == output, output
    start, end = 'お父さん', 'お母さん'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_locatednear(mrphs).format(start=start, end=end)
    assert '通常お父さんはお母さんの近くにいる。' == output, output


def get_surface_madeof(mrphs):
    """Return surfaceText for /r/MadeOf

    Returns:
    {start}は{end}から作られる。

    Constraints:
    - start=NP
    - end=NP

    Ref:
    (ja)
    /r/MadeOf   [[start]]は[[end]]から作られる．        444
    /r/MadeOf   [[start]]は[[end]]でできている．        404

    (en)
    /r/MadeOf   [[start]] is made of [[end]].   257
    /r/MadeOf   [[start]] is made of [[end]]    25
    /r/MadeOf   [[start]] can be made of [[end]]        15
    /r/MadeOf   [[start]] may be made of [[end]]        5
    /r/MadeOf   [[start]] may be made from [[end]]      1
    """

    start = ''.join(mrph.midasi for mrph in mrphs['start'])
    end = ''.join(m.midasi for m in mrphs['end'])

    return '{start}は{end}から作られる。'.format(start=start, end=end)


def get_surface_madeof_test():
    knp = KNP(jumanpp=True)
    start, end = 'テーブル', '木'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_madeof(mrphs).format(start=start, end=end)
    assert 'テーブルは木から作られる。' == output, output


def get_surface_mannerof(mrphs):
    """Return surfaceText for /r/MannerOf

    Returns:
    {start}は{end}ための方法である。

    Constraints:
    - start=VP
    - end=VP

    Ref:
    (ja)
    /r/MannerOf	{start}は{end}するための方法である	0

    (en)
    /r/MannerOf	{start} is a way to {end}	0
    """

    start = ''.join(mrph.midasi for mrph in mrphs['start'])
    end = ''.join(m.midasi for m in mrphs['end'])
    if is_sahen(mrphs['end']):
        end += 'する'
    elif is_vpkoto(mrphs['end'], vp=True):
        end = ''.join(m.midasi for m in mrphs['end'][:-1])

    return '{start}は{end}ための方法である。'.format(start=start, end=end)


def get_surface_mannerof_test():
    knp = KNP(jumanpp=True)
    start, end = '運動', '痩せる'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_mannerof(mrphs).format(start=start, end=end)
    assert '運動は痩せるための方法である。' == output, output
    start, end = '運動', '痩せること'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_mannerof(mrphs).format(start=start, end=end)
    assert '運動は痩せるための方法である。' == output, output
    start, end = '運動', '健康促進'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_mannerof(mrphs).format(start=start, end=end)
    assert '運動は健康促進するための方法である。' == output, output


def get_surface_motivatedbygoal(mrphs):
    """Return surfaceText for /r/MotivatedByGoal

    Returns:
    あなたは{end}ために{start}ことがある。

    Constraints:
    - start=VP
    - end=VP

    Ref:
    (ja)
    /r/MotivatedByGoal  あなたは[[end]]ために[[start]]必要がある．      710
    /r/MotivatedByGoal  あなたは[[end]]ために[[start]]ことがある．      550
    /r/MotivatedByGoal  あなたは[[end]]のために[[start]]ことがある．    398
    /r/MotivatedByGoal  もし[[end]]ことを望むなら，あなたは[[start]]べきである．        381
    /r/MotivatedByGoal  あなたは[[end]]のために[[start]]必要がある．    334

    (en)
    /r/MotivatedByGoal  You would [[start]] because you want to [[end]] 1517
    /r/MotivatedByGoal  You would [[start]] because you want [[end]]    728
    /r/MotivatedByGoal  You would [[start]] because [[end]]     153
    /r/MotivatedByGoal  If you want to [[end]] then you should [[start]]        2
    /r/MotivatedByGoal  You would [[start]] because you [[end]] 1
    """

    if is_sahen(mrphs['start']):
        start = ''.join(mrph.midasi for mrph in mrphs['start']) + 'する'
    elif is_vpkoto(mrphs['start'], vp=True):
        start = ''.join(mrph.midasi for mrph in mrphs['start'][:-1])
    else:
        start = ''.join(mrph.midasi for mrph in mrphs['start'])

    if is_sahen(mrphs['end']):
        end = ''.join(mrph.midasi for mrph in mrphs['end']) + 'する'
    elif is_vpkoto(mrphs['end'], vp=True):
        end = ''.join(mrph.midasi for mrph in mrphs['end'][:-1])
    else:
        end = ''.join(mrph.midasi for mrph in mrphs['end'])

    return 'あなたは{end}ために{start}ことがある。'.format(start=start, end=end)


def get_surface_motivatedbygoal_test():
    knp = KNP(jumanpp=True)
    start, end = '運動する', '痩せる'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_motivatedbygoal(mrphs).format(start=start, end=end)
    assert 'あなたは痩せるために運動することがある。' == output, output
    start, end = '運動', '痩せる'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_motivatedbygoal(mrphs).format(start=start, end=end)
    assert 'あなたは痩せるために運動することがある。' == output, output
    start, end = '運動すること', '痩せること'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_motivatedbygoal(mrphs).format(start=start, end=end)
    assert 'あなたは痩せるために運動することがある。' == output, output


# def get_surface_obstructedby(mrphs):  # Not found in English ConceptNet
#     """Return surfaceText for /r/ObstructedBy

#     Returns:
#     {start}は{end}によって防がれる。

#     Constraints:
#     - start=VP
#     - end=NP/VP

#     Ref:
#     (ja)
#     /r/ObstructedBy     [[start]]は[[end]]によって防がれる      0

#     (en)
#     /r/ObstructedBy     [[start]] is a goal that can be prevented by [[end]]    0
#     """

#     start = get_vpkoto(mrphs['start'],
#                        ''.join(mrph.midasi for mrph in mrphs['start']))
#     end = ''.join(m.midasi for m in mrphs['end'])
#     if is_vp(mrphs['end']):
#         end = get_vpkoto(mrphs['end'], end)

#     return '{start}は{end}によって防がれる。'.format(start=start, end=end)


# def get_surface_obstructedby_test():
#     knp = KNP(jumanpp=True)
#     start, end = '食べ過ぎる', '注意'
#     mrphs = {'start': knp.parse(start).mrph_list(),
#              'end': knp.parse(end).mrph_list()}
#     output = get_surface_obstructedby(mrphs).format(start=start, end=end)
#     assert '食べ過ぎることは注意によって防がれる。' == output, output


def get_surface_partof(mrphs):
    """Return surfaceText for /r/PartOf

    Returns:
    {start}は{end}の一部分である。

    Constraints:
    - start=NP
    - end=NP

    Ref:
    (ja)
    /r/PartOf   [[start]]は[[end]]の一部である．        6236
    /r/PartOf   [[start]]は[[end]]の一部である。        1

    (en)
    /r/PartOf   [[start]] is part of [[end]]    893
    /r/PartOf   [[start]] is part of [[end]].   370
    /r/PartOf   [[end]] has [[start]]   9
    /r/PartOf   [[start]] has [[end]]   1
    """

    start = ''.join(mrph.midasi for mrph in mrphs['start'])
    end = ''.join(m.midasi for m in mrphs['end'])

    return '{start}は{end}の一部分である。'.format(start=start, end=end)


def get_surface_partof_test():
    knp = KNP(jumanpp=True)
    start, end = 'タイヤ', '車'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_partof(mrphs).format(start=start, end=end)
    assert 'タイヤは車の一部分である。' == output, output


def get_surface_receivesaction(mrphs):
    """Return surfaceText for /r/Receivesaction

    Returns:
    あなたは{end}を{start}。

    Constraints:
    - start=NP
    - end=VP

    Ref:
    (ja)
    /r/ReceivesAction	[[start]]は[[end]]される	0

    (en)
    /r/ReceivesAction	[[start]] is [[end]]	48
    /r/ReceivesAction	[[start]] can be [[end]]	41
    /r/ReceivesAction	[[start]] are [[end]]	3
    """

    start = ''.join(mrph.midasi for mrph in mrphs['start'])

    if is_sahen(mrphs['end']):
        end = ''.join(mrph.midasi for mrph in mrphs['end']) + 'する'
    elif is_vpkoto(mrphs['end'], vp=True):
        mrphs['end'] = mrphs['end'][:-1]
    else:
        end = ''.join(m.midasi for m in mrphs['end'])

    return 'あなたは{start}を{end}。'.format(start=start, end=end)


def get_surface_receivesaction_test():
    knp = KNP(jumanpp=True)
    start, end = 'ケーキ', '切る'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_receivesaction(mrphs).format(start=start, end=end)
    assert 'あなたはケーキを切る。' == output, output


def get_surface_relatedto(mrphs):
    """Return surfaceText for /r/RelatedTo

    Returns:
    {start}は{end}と関係している。

    Constraints:
    - start=NP/VP/AP
    - end=NP/VP/AP

    Ref:
    (ja)
    /r/RelatedTo        [[start]]から[[end]]を連想することがある．      10375
    /r/RelatedTo        [[start]]と[[end]]は同時に見かける場合が多い．  274
    /r/RelatedTo        [[start]]は[[end]]は何かしらの関係を持っている。        5

    (en)
    /r/RelatedTo        [[start]] is related to [[end]] 108997
    """
    start = ''.join(m.midasi for m in mrphs['start'])
    if is_vp(mrphs['start']):
        start = get_vpkoto(mrphs['start'], start)
    elif is_ap(mrphs['start']):
        start = get_apkoto(mrphs['start'], start)
    end = ''.join(m.midasi for m in mrphs['end'])
    if is_vp(mrphs['end']):
        end = get_vpkoto(mrphs['end'], end)
    elif is_ap(mrphs['end']):
        end = get_apkoto(mrphs['end'], end)
    return '{start}は{end}と関係している。'.format(start=start, end=end)


def get_surface_relatedto_test():
    knp = KNP(jumanpp=True)
    start, end = '睡眠', '就寝'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_relatedto(mrphs).format(start=start, end=end)
    assert '睡眠は就寝と関係している。' == output, output
    start, end = '起床', '起きる'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_relatedto(mrphs).format(start=start, end=end)
    assert '起床は起きることと関係している。' == output, output
    start, end = '美しい', 'きれい'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_relatedto(mrphs).format(start=start, end=end)
    assert '美しいことはきれいなことと関係している。' == output, output


def get_surface_similarto(mrphs):
    """Return surfaceText for /r/SimilarTo

    Returns:
    {start}は{end}に似ている。

    Constraints:
    - start=NP/VP/AP
    - end=NP/VP/AP

    Ref:
    (ja)
    /r/SimilarTo        [[start]]は[[end]]に似ている    0

    (en)
    /r/SimilarTo        [[start]] is similar to [[end]] 0
    """
    start = ''.join(m.midasi for m in mrphs['start'])
    if is_vp(mrphs['start']):
        start = get_vpkoto(mrphs['start'], start)
    elif is_ap(mrphs['start']):
        start = get_apkoto(mrphs['start'], start)
    end = ''.join(m.midasi for m in mrphs['end'])
    if is_vp(mrphs['end']):
        end = get_vpkoto(mrphs['end'], end)
    elif is_ap(mrphs['end']):
        end = get_apkoto(mrphs['end'], end)
    return '{start}は{end}に似ている。'.format(start=start, end=end)


def get_surface_similarto_test():
    knp = KNP(jumanpp=True)
    start, end = '睡眠', '就寝'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_similarto(mrphs).format(start=start, end=end)
    assert '睡眠は就寝に似ている。' == output, output
    start, end = '起床', '起きる'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_similarto(mrphs).format(start=start, end=end)
    assert '起床は起きることに似ている。' == output, output
    start, end = '美しい', 'きれい'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_similarto(mrphs).format(start=start, end=end)
    assert '美しいことはきれいなことに似ている。' == output, output


def get_surface_symbolof(mrphs):
    """Return surfaceText for /r/Symbolof

    Returns:
    {start}は{end}の象徴である。

    Constraints:
    - start=NP
    - end=NP

    Ref:
    (ja)
    /r/SymbolOf [[start]]はあなたに[[end]]を想像させる．        554
    /r/SymbolOf [[start]]は[[end]]の象徴である．        158
    /r/SymbolOf [[start]]は[[end]]のたとえとして用いられる．    35

    (en)
    /r/SymbolOf [[start]] is a symbol of [[end]]        3
    """
    start = ''.join(m.midasi for m in mrphs['start'])
    end = ''.join(m.midasi for m in mrphs['end'])
    return '{start}は{end}の象徴である。'.format(start=start, end=end)


def get_surface_symbolof_test():
    knp = KNP(jumanpp=True)
    start, end = '聖火', 'オリンピック'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_symbolof(mrphs).format(start=start, end=end)
    assert '聖火はオリンピックの象徴である。' == output, output


def get_surface_synonym(mrphs):
    """Return surfaceText for /r/Synonym

    Returns:
    {start}は{end}の類義語である。

    Constraints:
    - start=NP
    - end=NP

    Ref:
    (ja)
    /r/Synonym  [[start]]は[[end]]の類義語である        0

    (en)
    /r/Synonym  [[start]] is a synonym of [[end]]       1
    """
    start = ''.join(m.midasi for m in mrphs['start'])
    end = ''.join(m.midasi for m in mrphs['end'])
    return '{start}は{end}の類義語である。'.format(start=start, end=end)


def get_surface_synonym_test():
    knp = KNP(jumanpp=True)
    start, end = '林檎', 'アップル'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_synonym(mrphs).format(start=start, end=end)
    assert '林檎はアップルの類義語である。' == output, output


def get_surface_usedfor(mrphs):
    """Return surfaceText for /r/UsedFor

    Returns:
    {start}は{end}ために使われる。

    Constraints:
    - start=NP
    - end=VP

    Ref:
    (ja)
    /r/UsedFor  {start}は{end}のために使う物である．        800
    /r/UsedFor  {start}はあなたが{end}ために使う物である．  704
    /r/UsedFor  {start}は{end}ために利用することができる．  179
    /r/UsedFor  {start}は{end}のために利用することができる．        132

    (en)
    /r/UsedFor  You can use {start} to {end}        4219
    /r/UsedFor  {start} is for {end}        1660
    /r/UsedFor  {start} is used for {end}   1007
    /r/UsedFor  {start} is used to {end}    371
    /r/UsedFor  {start} can be used to {end}        156
    /r/UsedFor  {start} is used for {end}.  53
    /r/UsedFor  {start} are used to {end}   28
    /r/UsedFor  {start} is for {end}.       17
    /r/UsedFor  {start} may be used to {end}        6
    /r/UsedFor  {start} can be used for {end}       5
    /r/UsedFor  When you want to {end}, you will use {start}        4
    /r/UsedFor  You can use {start} to {end}]       2
    """
    start = ''.join(m.midasi for m in mrphs['start'])
    end = ''.join(m.midasi for m in mrphs['end'])
    if is_sahen(mrphs['end']):
        end += 'する'
    elif is_vpkoto(mrphs['end'], vp=True):
        end = ''.join(m.midasi for m in mrphs['end'][:-1])

    return '{start}は{end}ために使われる。'.format(start=start, end=end)


def get_surface_usedfor_test():
    knp = KNP(jumanpp=True)
    start, end = '浮き輪', '救助'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_usedfor(mrphs).format(start=start, end=end)
    assert '浮き輪は救助するために使われる。' == output, output
    start, end = '箸', '食べる'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_usedfor(mrphs).format(start=start, end=end)
    assert '箸は食べるために使われる。' == output, output
    start, end = '箸', '食べること'
    mrphs = {'start': knp.parse(start).mrph_list(),
             'end': knp.parse(end).mrph_list()}
    output = get_surface_usedfor(mrphs).format(start=start, end=end)
    assert '箸は食べるために使われる。' == output, output


def test():
    logger.info('antonym')
    get_surface_antonym_test()
    logger.info('atlocation')
    get_surface_atlocation_test()
    logger.info('capableof')
    get_surface_capableof_test()
    logger.info('causes')
    get_surface_causes_test()
    logger.info('causesdesire')
    get_surface_causesdesire_test()
    logger.info('createdby')
    get_surface_createdby_test()
    logger.info('definedas')
    get_surface_definedas_test()
    logger.info('derivedfrom')
    get_surface_derivedfrom_test()
    logger.info('desires')
    get_surface_desires_test()
    logger.info('distinctfrom')
    get_surface_distinctfrom_test()
    logger.info('entails')
    get_surface_entails_test()
    logger.info('etymologicallyrelatedto')
    get_surface_etymologicallyrelatedto_test()
    logger.info('formof')
    get_surface_formof_test()
    logger.info('hasa')
    get_surface_hasa_test()
    logger.info('hascontext')
    get_surface_hascontext_test()
    logger.info('hasfirstsubevent')
    get_surface_hasfirstsubevent_test()
    logger.info('haslastsubevent')
    get_surface_haslastsubevent_test()
    logger.info('hasprerequisite')
    get_surface_hasprerequisite_test()
    logger.info('hasproperty')
    get_surface_hasproperty_test()
    logger.info('instanceof')
    get_surface_instanceof_test()
    logger.info('isa')
    get_surface_isa_test()
    logger.info('locatednear')
    get_surface_locatednear_test()
    logger.info('madeof')
    get_surface_madeof_test()
    logger.info('mannerof')
    get_surface_mannerof_test()
    logger.info('motivatedbygoal')
    get_surface_motivatedbygoal_test()
    logger.info('partof')
    get_surface_partof_test()
    logger.info('receivesaction')
    get_surface_receivesaction_test()
    logger.info('relatedto')
    get_surface_relatedto_test()
    logger.info('similarto')
    get_surface_similarto_test()
    logger.info('symbolof')
    get_surface_symbolof_test()
    logger.info('synonym')
    get_surface_synonym_test()
    logger.info('usedfor')
    get_surface_usedfor_test()


def main(args):
    global verbose, enc, port
    verbose = args.verbose

    if args.path_input == 'test':
        test()
        return 0

    # Read input
    if args.flag_no_header:
        df = pd.read_table(args.path_input, header=None)
        # Set columns
        df.columns = df.columns.map(str)
        cols = ['uri', 'rel', 'start', 'end', 'start_en', 'end_en', 'meta']
        for i, col in enumerate(cols):
            df.columns.values[i] = col
    else:
        df = pd.read_table(args.path_input, comment='#')

    if verbose:
        logger.info('Read {} rows from {}'.format(len(df), args.path_input))
    df.loc[:, 'key'] = df['uri'] + df['start'] + df['end']
    indices = df['key'].values

    # Get surfaceTexts
    facts = df.drop_duplicates(['rel', 'start', 'end'])
    surfaceText = get_surfaceTexts(facts, n_jobs=args.n_jobs)
    df = pd.merge(df, surfaceText, on=['rel', 'start', 'end'])
    df = df.set_index('key').loc[indices]

    # Output
    cols = ['rel', 'start', 'end', 'text']
    if args.path_output:
        df[cols].to_csv(args.path_output, sep='\t', index=False, header=False)
    else:
        df[cols].to_csv(sys.stdout, sep='\t', index=False, header=False)
    if verbose:
        logger.info('Write {} rows to {}'.format(
            len(df), args.path_output if args.path_output else 'stdout'))

    return 0


if __name__ == '__main__':
    logger = init_logger('Text')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to input file')
    parser.add_argument('-o', '--output', dest='path_output',
                        help='path to output file')
    parser.add_argument('--no-header', dest='flag_no_header',
                        action='store_true', default=False,
                        help='indicates path_input does not have header')
    parser.add_argument('-j', '--job', dest='n_jobs',
                        type=int, default=int(1),
                        help='number of JUMAN++ jobs')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
