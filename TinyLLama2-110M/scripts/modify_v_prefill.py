"""prefill_128 的 V-cache 匯出佈局修改（鏡射 decode 的做法）。

直接改 flatbuffer（不重跑 converter）。目標：把 12 個 V-cache 匯出從
(B,H,past,D)=(1,12,1024,64) 改成 (B,H,D,past)=(1,12,64,1024)，與改過的 decode 對齊。
attention（BMM）計算 bit-true 不變；K path / logits 不動。

做法與 decode 相同（token transpose 改 perm + BMM 前插 transpose + concat 改 axis）：
prefill 的 V transpose（reshape 後那顆 [0,2,1,3]）同時餵 BMM 與匯出 concat。把它改成
輸出新佈局，匯出 concat 直接吃；BMM 則靠一顆「只餵它」的 [0,1,3,2] transpose 轉回舊佈局。
那顆 transpose 是「最後兩軸對調」= matmul adj_y，可被 AnDLA 吸收進 BMM。

每條 V path 的 4 項改動（×12 層）：
  1. V transpose perm [0,2,1,3]->[0,2,3,1]：(1,12,128,64)->(1,12,64,128)（新建 perm buffer 重指）
  2. 新增 TRANSPOSE perm [0,1,3,2]：(1,12,64,128)->(1,12,128,64)，只餵該 BMM（rewire BMM.y）
  3. 匯出 concat axis 2->3，輸出 (1,12,1024,64)->(1,12,64,1024)
  4. pad const (1,12,896,64)->(1,12,64,896)：沿用舊 buffer（bytes 全同一值=zp，reshape 與 transpose byte 等價）

新增/改動的 perm 常數一律新建 buffer 再重指；pad 走 metadata reshape（改 shape、buffer 照用）。
純確定性流程；帶 assert 檢查來源 shape/perm，對已改過的模型重跑會 assert 失敗，防止重複套用。
"""
import numpy as np, os
from tensorflow.lite.python import schema_py_generated as fb
import flatbuffers

from paths import PREFILL_SRC as SRC, PREFILL_OUT as OUT
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

def is_bmm(op): return opcode_name(op) == "BATCH_MATMUL"
def is_concat(op): return opcode_name(op) == "CONCATENATION"
def is_transpose(op): return opcode_name(op) == "TRANSPOSE"

transpose_opcode_idx = None
for i, oc in enumerate(m.operatorCodes):
    if oc.builtinCode == fb.BuiltinOperator.TRANSPOSE:
        transpose_opcode_idx = i; break
assert transpose_opcode_idx is not None

producer = {}
consumers = {}
for op in ops:
    for o in op.outputs: producer[o] = op
    for i in op.inputs: consumers.setdefault(i, []).append(op)
graph_outputs = set(sg.outputs)

def is_const(ti):
    d = buffers[tensors[ti].buffer].data
    return d is not None and len(d) > 0

def perm_of(ti):
    d = buffers[tensors[ti].buffer].data
    return np.frombuffer(d.tobytes(), dtype=np.int32).tolist()

# 12 V tensors: BMM y with out last-dim 64
v_bmms = []
for op in ops:
    if not is_bmm(op): continue
    out = op.outputs[0]
    if tensors[out].shape is not None and list(tensors[out].shape)[-1] == 64:
        v_bmms.append(op)
assert len(v_bmms) == 12, len(v_bmms)

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
    t.name = name.encode()
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
    b = fb.BufferT(); b.data = None; buffers.append(b); return len(buffers) - 1

bmm_to_newtr = {}
log = []

for bmm in v_bmms:
    vt = bmm.inputs[1]
    vtr = producer[vt]
    assert is_transpose(vtr), opcode_name(vtr)
    # export concat consuming vt
    ce = [c for c in consumers.get(vt, []) if is_concat(c)]
    assert len(ce) == 1, (vt, len(ce))
    concat = ce[0]
    pad = [i for i in concat.inputs if is_const(i)]
    assert len(pad) == 1
    pad_idx = pad[0]
    cout = concat.outputs[0]
    assert cout in graph_outputs

    # --- item1: V transpose perm [0,2,1,3]->[0,2,3,1], out (1,12,128,64)->(1,12,64,128) ---
    assert perm_of(vtr.inputs[1]) == [0, 2, 1, 3], perm_of(vtr.inputs[1])
    assert list(tensors[vt].shape) == [1, 12, 128, 64]
    vtr.inputs[1] = make_perm_tensor([0, 2, 3, 1], "vmove_pf_vperm_v%d" % vt)
    tensors[vt].shape = [1, 12, 64, 128]

    # --- item2: new TRANSPOSE [0,1,3,2] (1,12,64,128)->(1,12,128,64), feeds only this BMM ---
    perm_ti = make_perm_tensor([0, 1, 3, 2], "vmove_pf_bmmperm_v%d" % vt)
    ntr_out = fb.TensorT()
    ntr_out.shape = [1, 12, 128, 64]
    ntr_out.type = fb.TensorType.INT8
    ntr_out.quantization = copy_quant(tensors[vt].quantization)
    ntr_out.buffer = empty_buffer()
    ntr_out.name = ("vmove_pf_bmm_transpose_out_v%d" % vt).encode()
    tensors.append(ntr_out); ntr_out_idx = len(tensors) - 1
    ntr = fb.OperatorT()
    ntr.opcodeIndex = transpose_opcode_idx
    ntr.inputs = [vt, perm_ti]
    ntr.outputs = [ntr_out_idx]
    ntr.builtinOptionsType = fb.BuiltinOptions.TransposeOptions
    ntr.builtinOptions = fb.TransposeOptionsT()
    bmm_to_newtr[id(bmm)] = ntr
    # rewire BMM y -> new transpose out (export concat still reads vt = new layout)
    bmm.inputs[1] = ntr_out_idx

    # --- item3: pad const (1,12,896,64)->(1,12,64,896), reuse buffer (bytes 全同一值) ---
    assert list(tensors[pad_idx].shape) == [1, 12, 896, 64]
    pad_buf = buffers[tensors[pad_idx].buffer].data
    assert pad_buf is not None and len(pad_buf) == 12 * 896 * 64
    assert len(set(np.frombuffer(pad_buf.tobytes(), dtype=np.int8).tolist())) == 1, "pad 非單一值填充"
    tensors[pad_idx].shape = [1, 12, 64, 896]

    # --- item4: concat axis 2->3, out (1,12,1024,64)->(1,12,64,1024) ---
    assert concat.builtinOptions.axis == 2
    concat.builtinOptions.axis = 3
    assert list(tensors[cout].shape) == [1, 12, 1024, 64]
    tensors[cout].shape = [1, 12, 64, 1024]

    log.append((vt, ntr_out_idx, pad_idx, cout, bmm.outputs[0]))

# rebuild operators list: insert each new transpose right BEFORE its BMM
new_ops = []
for op in ops:
    if id(op) in bmm_to_newtr:
        new_ops.append(bmm_to_newtr[id(op)])
    new_ops.append(op)
sg.operators = new_ops

os.makedirs(OUTDIR, exist_ok=True)
b = flatbuffers.Builder(0)
b.Finish(m.Pack(b), file_identifier=b"TFL3")
open(OUT, "wb").write(bytes(b.Output()))
print("wrote", OUT, "size", os.path.getsize(OUT))
print("V paths modified:", len(log))
print("total tensors now", len(tensors), "ops now", len(sg.operators), "buffers now", len(buffers))
