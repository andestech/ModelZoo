import numpy as np, copy, os
from tensorflow.lite.python import schema_py_generated as fb
import flatbuffers

from paths import DECODE_SRC as SRC, DECODE_OUT as OUT
OUTDIR = os.path.dirname(OUT)

buf = bytearray(open(SRC, "rb").read())
m = fb.ModelT.InitFromObj(fb.Model.GetRootAsModel(buf, 0))
sg = m.subgraphs[0]
tensors = sg.tensors
ops = sg.operators
buffers = m.buffers

def opcode_name(op):
    bc = m.operatorCodes[op.opcodeIndex].builtinCode
    for k, v in fb.BuiltinOperator.__dict__.items():
        if not k.startswith("_") and v == bc: return k
    return "C%d" % bc

# find TRANSPOSE opcode index (reuse)
TRANSPOSE_BC = fb.BuiltinOperator.TRANSPOSE
transpose_opcode_idx = None
for i, oc in enumerate(m.operatorCodes):
    if oc.builtinCode == TRANSPOSE_BC:
        transpose_opcode_idx = i; break
assert transpose_opcode_idx is not None

# producer map (tensor idx -> op)
producer = {}
for op in ops:
    for o in op.outputs: producer[o] = op
graph_inputs = set(sg.inputs)

def newshape_of_reshape(op):
    return None
# identify 12 V paths (BMM out last-dim 64, y from concat)
def is_bmm(op): return opcode_name(op) == "BATCH_MATMUL"
def is_concat(op): return opcode_name(op) == "CONCATENATION"
def is_transpose(op): return opcode_name(op) == "TRANSPOSE"
def is_ss(op): return opcode_name(op) == "STRIDED_SLICE"

vpaths = []
for op in ops:
    if not is_bmm(op): continue
    y = op.inputs[1]; out = op.outputs[0]
    if tensors[out].shape is not None and list(tensors[out].shape)[-1] == 64:
        pop = producer.get(y)
        if pop is not None and is_concat(pop):
            vpaths.append((op, pop))
assert len(vpaths) == 12, len(vpaths)

def make_buffer(int32_list):
    b = fb.BufferT()
    b.data = np.frombuffer(np.array(int32_list, dtype=np.int32).tobytes(), dtype=np.uint8)
    buffers.append(b)
    return len(buffers) - 1

def make_perm_tensor(int32_list, name):
    bi = make_buffer(int32_list)
    t = fb.TensorT()
    t.shape = [len(int32_list)]
    t.type = fb.TensorType.INT32
    t.buffer = bi
    t.name = name.encode() if isinstance(name, str) else name
    tensors.append(t)
    return len(tensors) - 1

def copy_quant(src_q):
    if src_q is None: return None
    q = fb.QuantizationParametersT()
    q.scale = list(src_q.scale) if src_q.scale is not None else None
    q.zeroPoint = list(src_q.zeroPoint) if src_q.zeroPoint is not None else None
    q.min = list(src_q.min) if src_q.min is not None else None
    q.max = list(src_q.max) if src_q.max is not None else None
    q.quantizedDimension = src_q.quantizedDimension
    q.detailsType = src_q.detailsType
    return q

def empty_buffer():
    b = fb.BufferT(); b.data = None; buffers.append(b); return len(buffers)-1

# map concat op -> the new transpose op object (for list rebuild)
concat_to_newtr = {}
log = []

for bmm, concat in vpaths:
    cout_idx = concat.outputs[0]
    cout_t = tensors[cout_idx]
    # V cache input & new-token input
    cache_idx = new_idx = None
    for i in concat.inputs:
        if i in graph_inputs: cache_idx = i
        else: new_idx = i
    # --- item1: V cache shape (1,12,1024,64)->(1,12,64,1024) ---
    assert list(tensors[cache_idx].shape) == [1,12,1024,64]
    tensors[cache_idx].shape = [1,12,64,1024]
    # --- item2: new-tok transpose perm [0,2,1,3]->[0,2,3,1], out (1,12,1,64)->(1,12,64,1) ---
    tr = producer[new_idx]
    assert is_transpose(tr)
    # repoint perm to fresh buffer (avoid clobbering shared const)
    tr.inputs[1] = make_perm_tensor([0,2,3,1], "vmove_newtok_perm_%d" % new_idx)
    assert list(tensors[new_idx].shape) == [1,12,1,64]
    tensors[new_idx].shape = [1,12,64,1]
    # --- item3: concat axis 2->3, out (1,12,1025,64)->(1,12,64,1025) ---
    assert concat.builtinOptions.axis == 2
    concat.builtinOptions.axis = 3
    assert list(cout_t.shape) == [1,12,1025,64]
    cout_t.shape = [1,12,64,1025]
    # --- item4: insert TRANSPOSE perm[0,1,3,2] (1,12,64,1025)->(1,12,1025,64), feed BMM ---
    perm_ti = make_perm_tensor([0,1,3,2], "vmove_post_perm_c%d" % cout_idx)
    ntr_out = fb.TensorT()
    ntr_out.shape = [1,12,1025,64]
    ntr_out.type = fb.TensorType.INT8
    ntr_out.quantization = copy_quant(cout_t.quantization)
    ntr_out.buffer = empty_buffer()
    ntr_out.name = ("vmove_post_transpose_out_c%d" % cout_idx).encode()
    tensors.append(ntr_out); ntr_out_idx = len(tensors)-1
    ntr = fb.OperatorT()
    ntr.opcodeIndex = transpose_opcode_idx
    ntr.inputs = [cout_idx, perm_ti]
    ntr.outputs = [ntr_out_idx]
    ntr.builtinOptionsType = fb.BuiltinOptions.TransposeOptions
    ntr.builtinOptions = fb.TransposeOptionsT()
    concat_to_newtr[id(concat)] = ntr
    # rewire BMM y -> new transpose output
    assert bmm.inputs[1] == cout_idx
    bmm.inputs[1] = ntr_out_idx
    # --- item5: strided slice cache export: axis2->axis3 ---
    # find the strided slice consuming cout_idx
    ss = None
    for op in ops:
        if is_ss(op) and op.inputs[0] == cout_idx: ss = op; break
    assert ss is not None
    begin_i, end_i, strides_i = ss.inputs[1], ss.inputs[2], ss.inputs[3]
    # repoint begin/end to fresh buffers
    ss.inputs[1] = make_perm_tensor([0,0,0,1], "vmove_ss_begin_c%d" % cout_idx)
    ss.inputs[2] = make_perm_tensor([0,0,0,1025], "vmove_ss_end_c%d" % cout_idx)
    # masks 11 (0b1011, axes0,1,3 full) -> 7 (0b0111, axes0,1,2 full)
    assert ss.builtinOptions.beginMask == 11 and ss.builtinOptions.endMask == 11
    ss.builtinOptions.beginMask = 7
    ss.builtinOptions.endMask = 7
    ss_out = ss.outputs[0]
    assert list(tensors[ss_out].shape) == [1,12,1024,64]
    tensors[ss_out].shape = [1,12,64,1024]
    log.append((cache_idx, tr.inputs[1], new_idx, concat.outputs[0], ntr_out_idx, bmm.outputs[0], ss.outputs[0]))

# rebuild operators list: insert each new transpose right after its concat
new_ops = []
for op in ops:
    new_ops.append(op)
    if id(op) in concat_to_newtr:
        new_ops.append(concat_to_newtr[id(op)])
sg.operators = new_ops

os.makedirs(OUTDIR, exist_ok=True)
b = flatbuffers.Builder(0)
b.Finish(m.Pack(b), file_identifier=b"TFL3")
open(OUT, "wb").write(bytes(b.Output()))
print("wrote", OUT, "size", os.path.getsize(OUT))
print("layers modified:", len(log))
print("total tensors now", len(tensors), "ops now", len(sg.operators), "buffers now", len(buffers))
