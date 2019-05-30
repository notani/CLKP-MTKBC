#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import numpy as np
import pandas as pd
import sys

debug = False
verbose = False
logger = None


def init_logger(name='logger'):
    """Return a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)
    return logger


def extract_surfaceText(meta):
    """Extract a surfaceText from a meta tag in ConceptNet."""
    def clean_expression(s):
        """Remove symbols."""
        s = s.replace('*', '')
        s = s.replace('[[', '')
        s = s.replace(']]', '')
        return s

    try:
        txt = clean_expression(json.loads(meta)['surfaceText'].lower())
    except KeyError:  # No default surfaceText is found
        return np.nan

    if not txt.endswith('.'):  # Add a period
        txt += '.'
    return txt


def complete_surfaceText(df):
    """Generate surfaceTexts."""
    indices = df[pd.isnull(df['text'])].index

    # Relati on-specific templates
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
        '/r/etymologicallyrelatedto': get_surface_etymologicallyrelatedto,
        '/r/entails': get_surface_entails,
        '/r/formof': get_surface_formof,
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

    memo = {}  # fact -> surfaceText
    unknown_rels = set()
    for idx in indices:
        row = df.loc[idx]
        rel = row['rel'].lower()
        start = row['start_en'].split('/')[3].replace('_', ' ')
        end = row['end_en'].split('/')[3].replace('_', ' ')
        try:
            df.loc[idx, 'text'] = memo['{}|||{}|||{}'.format(rel, start, end)]
            continue
        except KeyError:
            pass
        try:
            df.loc[idx, 'text'] = func[rel](start, end)
            # Save an obtained surfaceText to memo
            memo['{}|||{}|||{}'.format(rel, start, end)] = df.loc[idx, 'text']
            if debug:
                logger.debug(df.loc[idx, 'text'])
        except KeyError:
            unknown_rels.add(rel)

    # Report if there is an undefined relation type
    for rel in unknown_rels:
        logger.error('No surfaceText is defined for {}'.format(rel))


def get_surface_antonym(start, end):
    """Return surfaceText for /r/Antonym

    Returns:
    {start} is the opposite of {end}.

    Constraints:
    - start=NP
    - end=NP

    Ref:
    (ja)
    /r/Antonym  [[start]]は[[end]]の反義語である        0

    (en)
    /r/Antonym  [[start]] is the opposite of [[end]]    4556
    """
    return '{start} is the opposite of {end}.'.format(start=start, end=end)


def get_surface_atlocation(start, end):
    """Return surfaceText for /r/AtLocation

    Returns:
    You are likely to find {start} in {end}

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

    return 'You are likely to find {start} in {end}'.format(
        start=start, end=end)


def get_surface_capableof(start, end):
    """Return surfaceText for /r/CapableOf

    Returns:
    {start} can {end}.

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
    return '{start} can {end}.'.format(start=start, end=end)


def get_surface_causes(start, end):
    """Return surfaceText for /r/Causes

    Returns:
    Sometimes {start} causes {end}.

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
    return 'The effect of {start} is {end}.'.format(start=start, end=end)


def get_surface_causesdesire(start, end):
    """Return surfaceText for /r/CausesDesire

    Returns:
    {start} would make you want to {end}.

    Constraints:
    - start=NP
    - end=VP

    Ref:
    (ja)
    /r/CausesDesire     {start}はあなたが{end}ことをやる気にさせる．        580

    (en)
    /r/CausesDesire     {start} would make you want to {end}        1800
    """
    return '{start} would make you want to {end}'.format(start=start, end=end)


def get_surface_createdby(start, end):
    """Return surfaceText for /r/CreatedBy

    Returns:
    {start} is created by {end}.

    Constraints:
    - start=NP
    - end=VP

    Ref:
    (ja)
    /r/CreatedBy        {start}は{end}によって作られる      0

    (en)
    /r/CreatedBy        {start} is created by {end}.        136
    """
    return '{start} is created by {end}.'.format(start=start, end=end)


def get_surface_definedas(start, end):
    """Return surfaceText for /r/DefinedAs

    Returns:
    {start} is the {end}.

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
    return '{start} is the {end}.'.format(start=start, end=end)


def get_surface_derivedfrom(start, end):
    """Return surfaceText for /r/DerivedFrom

    Returns:
    {start} is derived from {end}.

    Constraints:
    - start=NP
    - end=NP

    Ref:
    (ja)
    /r/DerivedFrom      {start}は{end}に由来する    0

    (en)
    /r/DerivedFrom      {start} is derived from {end}       0
    """
    return '{start} is derived from {end}.'.format(start=start, end=end)


def get_surface_desires(start, end):
    """Return surfaceText for /r/Desires

    Returns:
    {start} wants {end}.

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
    return '{start} wants {end}.'.format(start=start, end=end)


def get_surface_distinctfrom(start, end):
    """Return surfaceText for /r/DistinctFrom

    Returns:
    {start} is not {end}.

    Constraints:
    - start=NP/VP/AP
    - end=NP/VP/AP

    Ref:
    (ja)
    /r/DistinctFrom     {start}は{end}ではない      0

    (en)
    /r/DistinctFrom     {start} is not {end}        1921
    """
    return '{start} is not {end}.'.format(start=start, end=end)


def get_surface_entails(start, end):
    """Return surfaceText for /r/Entails

    Returns:
    {start} entails {end}.

    Constraints:
    - start=NP/VP
    - end=NP/VP

    Ref:
    (ja)
    /r/Entails  [[start]] entails [[end]]       0

    (en)
    /r/Entails  [[start]]は[[end]]を含意する    0
    """
    
    return '{start} entails {end}.'.format(start=start, end=end)


def get_surface_etymologicallyrelatedto(start, end):
    """Return surfaceText for /r/EtymologicallyRelatedTo

    Returns:
    {start} is etymologically related to {end}.

    Constraints:
    - start=NP/VP/AP
    - end=NP/VP/AP

    Ref:
    (ja)
    /r/EtymologicallyRelatedTo  [[start]]は[[end]]と語源的な関係を持っている    0

    (en)
    /r/EtymologicallyRelatedTo  [[start]] is etymologically related to [[end]]  0
    """
    return '{start} is etymologically related to {end}.'.format(
        start=start, end=end)


def get_surface_formof(start, end):
    """Return surfaceText for /r/FormOf

    Returns:
    {start} is a form of {end}.

    Constraints:
    - start=NP/VP/AP
    - end=NP/VP/AP

    Ref:
    (ja)
    /r/FormOf	[[start]]は[[end]]の形態である	0

    (en)
    /r/FormOf	[[start]] is a form of [[end]]	0
    """
    return '{start} is a form of {end}.'.format(start=start, end=end)


def get_surface_hascontext(start, end):
    """Return surfaceText for /r/HasContext

    Returns:
    {start} has a context of {end}.

    Constraints:
    - start=NP/VP/AP
    - end=NP

    Ref:
    (ja)
    /r/HasContext	[[start]]は[[end]]という文脈を持っている	0

    (en)
    /r/HasContext       [[start]] has a context of [[end]]      0
    """
    return '{start} has a context of {end}.'.format(start=start, end=end)


def get_surface_hasa(start, end):
    """Return surfaceText for /r/HasA

    Returns:
    {start} has {end}.

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
    return '{start} has {end}.'.format(start=start, end=end)


def get_surface_hascontext(start, end):
    """Return surfaceText for /r/HasContext

    Returns:
    {start} has a context of {end}.

    Constraints:
    - start=NP/VP/AP
    - end=NP

    Ref:
    (ja)
    /r/HasContext	[[start]]は[[end]]という文脈を持っている	0

    (en)
    /r/HasContext       [[start]] has a context of [[end]]      0
    """
    return '{start} has a context of {end}.'.format(start=start, end=end)


def get_surface_relatedto(start, end):
    """Return surfaceText for /r/RelatedTo

    Returns:
    {start} is related to {end}.

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
    return '{start} is related to {end}.'.format(start=start, end=end)


def get_surface_hasfirstsubevent(start, end):
    """Return surfaceText for /r/HasFirstSubevent

    Returns:
    The first thing you do when you {start} is {end}.

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
    return 'The first thing you do when you {start} is {end}'.format(start=start, end=end)


def get_surface_haslastsubevent(start, end):
    """Return surfaceText for /r/HasLastsubevent

    Returns:
    The last thing you do when you {start} is {end}.

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
    return 'The last thing you do when you {start} is {end}.'.format(start=start, end=end)


def get_surface_hasprerequisite(start, end):
    """Return surfaceText for /r/HasPrerequisite

    Returns:
    If you want to {start} then you should {end}.

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
    return 'If you want to {start} then you should {end}.'.format(start=start, end=end)


def get_surface_hasproperty(start, end):
    """Return surfaceText for /r/HasProperty

    Returns:
    {start} is {end}.

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
    return '{start} is {end}.'.format(start=start, end=end)


def get_surface_instanceof(start, end):
    """Return surfaceText for /r/InstanceOf

    Returns:
    {start} is an example of {end}.

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
    return '{start} is an example of {end}.'.format(start=start, end=end)


def get_surface_isa(start, end):
    """Return surfaceText for /r/IsA

    Returns:
    {start} is {end}.

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
    return '{start} is {end}.'.format(start=start, end=end)


def get_surface_locatednear(start, end):
    """Return surfaceText for /r/LocatedNear

    Returns:
    {start} is typically near {end}.

    Constraints:
    - start=NP
    - end=NP

    Ref:
    (ja)
    /r/LocatedNear      [[start]]は通常[[end]]の近くにある      0

    (en)
    /r/LocatedNear      [[start]] is typically near [[end]]     41
    """
    return '{start} is typically near {end}.'.format(start=start, end=end)


def get_surface_madeof(start, end):
    """Return surfaceText for /r/MadeOf

    Returns:
    {start} is made of {end}.

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
    return '{start} is made of {end}.'.format(start=start, end=end)


def get_surface_mannerof(start, end):
    """Return surfaceText for /r/MannerOf

    Returns:
    {start} is a way to {end}.

    Constraints:
    - start=VP
    - end=VP

    Ref:
    (ja)
    /r/MannerOf	{start}は{end}するための方法である	0

    (en)
    /r/MannerOf	{start} is a way to {end}	0
    """
    return '{start} is a way to {end}.'.format(start=start, end=end)


def get_surface_motivatedbygoal(start, end):
    """Return surfaceText for /r/MotivatedByGoal

    Returns:
    You would {start} because you want to {end}.

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
    return 'You would {start} because you want to {end}.'.format(start=start, end=end)


def get_surface_partof(start, end):
    """Return surfaceText for /r/PartOf

    Returns:
    {start} is part of {end}.

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
    return '{start} is part of {end}.'.format(start=start, end=end)


def get_surface_receivesaction(start, end):
    """Return surfaceText for /r/Receivesaction

    Returns:
    {start} is {end}.

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
    return '{start} is {end}.'.format(start=start, end=end)


def get_surface_relatedto(start, end):
    """Return surfaceText for /r/RelatedTo

    Returns:
    {start} is related to {end}.

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
    return '{start} is related to {end}.'.format(start=start, end=end)


def get_surface_similarto(start, end):
    """Return surfaceText for /r/SimilarTo

    Returns:
    {start} is similar to {end}.

    Constraints:
    - start=NP/VP/AP
    - end=NP/VP/AP

    Ref:
    (ja)
    /r/SimilarTo        [[start]]は[[end]]に似ている    0

    (en)
    /r/SimilarTo        [[start]] is similar to [[end]] 0
    """
    return '{start} is similar to {end}.'.format(start=start, end=end)


def get_surface_synonym(start, end):
    """Return surfaceText for /r/Synonym

    Returns:
    {start} is a synonym of {end}.

    Constraints:
    - start=NP
    - end=NP

    Ref:
    (ja)
    /r/Synonym  [[start]]は[[end]]の類義語である        0

    (en)
    /r/Synonym  [[start]] is a synonym of [[end]]       1
    """
    return '{start} is a synonym of {end}.'.format(start=start, end=end)


def get_surface_symbolof(start, end):
    """Return surfaceText for /r/Symbolof

    Returns:
    {start} is a symbol of {end}.

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
    return '{start} is a symbol of {end}.'.format(start=start, end=end)


def get_surface_usedfor(start, end):
    """Return surfaceText for /r/UsedFor

    Returns:
    You can use {start} to {end}.

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
    return 'You can use {start} to {end}.'.format(start=start, end=end)


def main(args):
    """Entry point."""
    global debug, verbose
    debug = args.debug
    verbose = args.verbose or debug

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
    df['key'] = df['uri'] + df['start'] + df['end']
    indices = df['key'].values

    if verbose:
        logger.info('Read {} lines from {}'.format(len(df), args.path_input))

    # Extract surfaceTexts
    metas = df['meta'].unique()
    surfaceTexts = pd.DataFrame([[meta, extract_surfaceText(meta)]
                                 for meta in metas],
                                columns=['meta', 'text'])
    df = pd.merge(df, surfaceTexts, on='meta')
    complete_surfaceText(df)

    notfound = df[pd.isnull(df['text'])]
    if len(notfound):
        logger.error('surfaceText is not found:')
        cols = ['start_en', 'rel', 'end_en']
        for idx, row in notfound[cols].drop_duplicates().iterrows():
            logger.error('{}\t{}\t{}'.format(
                row['start_en'], row['rel'], row['end_en']))
        raise
    df = df.set_index('key').ix[indices]

    # Output
    cols = ['rel', 'start_en', 'end_en', 'text']
    if verbose:
        logger.info('Write {} lines to {}'.format(
            len(df), args.path_output if args.path_output else 'stdout'))
    if args.path_output:
        df[cols].to_csv(args.path_output, sep='\t', index=False, header=False)
    else:
        df[cols].to_csv(sys.stdout, sep='\t', index=False, header=False)

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
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    parser.add_argument('-d', '--debug',
                        action='store_true', default=False,
                        help='debug mode')
    args = parser.parse_args()
    main(args)
