import dis, marshal, sys, os, types
R = sys.argv[1]
raw = open(os.path.join(R, "code_v3", "__pycache__", "freeze_v3.cpython-312.pyc"), "rb").read()
code = marshal.loads(raw[16:])
for c in code.co_consts:
    if isinstance(c, types.CodeType) and c.co_name == "check":
        for ins in dis.get_instructions(c):
            if ins.opname not in ("CACHE", "RESUME", "NOP"):
                print(f"{ins.positions.lineno if ins.positions else '':>4} {ins.opname:24s} {ins.argrepr}")
print("---- module level line numbers of defs")
for ins in dis.get_instructions(code):
    if ins.opname in ("STORE_NAME",):
        print(ins.positions.lineno, ins.argrepr)
