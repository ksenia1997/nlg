from typing import List


def sample(model, sentences: List[str], beam: int = 1, verbose: bool = False, **kwargs) -> str:
    input = [model.encode(sentence) for sentence in sentences]
    hypos = model.generate(input, beam, verbose, **kwargs)
    return [model.decode(x['tokens']) for x in hypos]
