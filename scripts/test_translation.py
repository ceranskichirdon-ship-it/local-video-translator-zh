import json, threading
from pathlib import Path
from pipeline import Pipeline
from translation import LocalTranslator
root = Path(__file__).resolve().parents[1]
config = Pipeline(root).config
segments = [
 {'id':1,'start':0,'end':3,'text':'Hello. We will meet at the station tomorrow at nine.'},
 {'id':2,'start':3,'end':6,'text':'今日はいい天気ですね。明日の朝、駅で会いましょう。'},
 {'id':3,'start':6,'end':9,'text':'안녕하세요. 내일 아침 아홉 시에 역에서 만나요.'}
]
model = LocalTranslator(config,lambda e:print(e.get('message',''),flush=True),threading.Event(),root/'work/translation_test')
try:
 model.start()
 results = {}
 for item, lang in zip(segments,['en','ja','ko']):
  results.update(model.translate_batch([item],lang))
 print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)
 (root/'work/translation_test/results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
 assert len(results)==3
 assert all(any('\u4e00'<=c<='\u9fff' for c in v) for v in results.values())
finally:
 model.close()
