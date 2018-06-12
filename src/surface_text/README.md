# Requirements

- Python 3.6
- pandas
- (Japanese)
    - [JUMAN++](https://github.com/ku-nlp/jumanpp), [pyknp (v0.3)](http://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=http://lotus.kuee.kyoto-u.ac.jp/nl-resource/pyknp/pyknp-0.3.tar.gz&name=pyknp-0.3.tar)
    - [mojimoji](https://github.com/studio-ousia/mojimoji)
    - [neologdn](https://github.com/ikegami-yukino/neologdn)

# Add expressions to facts

```shell
# Get expressions
python get_surfaceText_en.py examples/bat-capableof-fly.enja.tsv -o examples/bat-capableof-fly.surfaceText.en -v
python get_surfaceText_ja.py examples/bat-capableof-fly.enja.tsv -o examples/bat-capableof-fly.surfaceText.ja -v
python get_surfaceText_zh.py examples/bat-capableof-fly.enzh.tsv -o examples/bat-capableof-fly.surfaceText.zh -v

# Tokenization
python segment_surfaceText_en.py bat-capableof-fly.surfaceText.en bat-capableof-fly.surfaceText.seg.en -v
python segment_surfaceText_ja.py bat-capableof-fly.surfaceText.ja bat-capableof-fly.surfaceText.seg.ja -v
```
