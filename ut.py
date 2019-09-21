from u import *

Scratch = Path(os.environ['S'])
Proj = Scratch / 'micronet'
Cache = Proj / 'cache'
Data = Proj / 'data'

Wiki = (Proj / 'wikitext-103').mk()

def get_encoder(decoder):
    return OrderedDict((x, i) for i, x in enumerate(decoder))