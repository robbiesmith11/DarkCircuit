import modal, itertools, json
buf = modal.Dict.from_name("trace-buffer")
def dump_keys():
    print(list(itertools.islice(buf.keys(), 20)))