import copy
import onnx
import numpy as np
from onnxsim import simplify
from onnx import numpy_helper


def simplify_model(model_path, input_shapes):
    model = onnx.load(model_path)
    sim_model, check = simplify(model, overwrite_input_shapes=input_shapes)
    assert check, "onnxsim simplify failed"
    sim_model = onnx.shape_inference.infer_shapes(sim_model)
    return sim_model


def get_init_array(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def get_tensor_shape(model, tensor_name):
    infos = list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
    for v in infos:
        if v.name == tensor_name:
            dims = []
            for d in v.type.tensor_type.shape.dim:
                dims.append(int(d.dim_value) if d.dim_value > 0 else None)
            return dims
    return None


def get_reshape_nodes(model):
    return {
        n.output[0]: n
        for n in model.graph.node
        if n.op_type == "Reshape" and len(n.input) >= 2 and len(n.output) >= 1
    }


def add_initializer(model, name, arr, dtype=np.int64):
    init = numpy_helper.from_array(np.asarray(arr, dtype=dtype), name)
    model.graph.initializer.append(init)


def patch_dynamic_reshape(model_path, shape1, shape2, out_path):
    if len(shape1) != 1 or len(shape2) != 1:
        raise ValueError("only single-input model is supported")

    input_name1, dims1 = next(iter(shape1.items()))
    input_name2, dims2 = next(iter(shape2.items()))
    if input_name1 != input_name2:
        raise ValueError("shape1 and shape2 must use the same input name")

    batch1 = int(dims1[0])
    batch2 = int(dims2[0])

    sim1 = simplify_model(model_path, shape1)
    sim2 = simplify_model(model_path, shape2)
    patched = copy.deepcopy(sim1)

    r1 = get_reshape_nodes(sim1)
    r2 = get_reshape_nodes(sim2)

    patched_count = 0

    # 在 patched 裡建 output_name -> node 的對應，方便改 input[1]
    patched_r = get_reshape_nodes(patched)

    for out_name, n1 in r1.items():
        n2 = r2.get(out_name)
        pn = patched_r.get(out_name)
        if n2 is None or pn is None:
            continue

        out_shape1 = get_tensor_shape(sim1, n1.output[0])
        out_shape2 = get_tensor_shape(sim2, n2.output[0])
        if out_shape1 is None or out_shape2 is None:
            continue
        if len(out_shape1) != len(out_shape2):
            continue

        diff_idx = [i for i, (a, b) in enumerate(zip(out_shape1, out_shape2)) if a != b]
        if len(diff_idx) != 1:
            continue

        dyn_idx = diff_idx[0]
        a = out_shape1[dyn_idx]
        b = out_shape2[dyn_idx]
        if a is None or b is None:
            continue

        # 只接受和 batch 成比例變化的維度
        if b * batch1 != a * batch2:
            continue

        # 用 Reshape 的 output shape 當 base
        new_shape = list(out_shape1)
        new_shape[dyn_idx] = -1

        old_shape_name = pn.input[1]
        old_arr = get_init_array(patched, old_shape_name)
        old_dtype = old_arr.dtype if old_arr is not None else np.int64

        # 關鍵：每個 reshape 都建立自己的 shape initializer，避免 shared initializer 被覆寫
        new_shape_name = f"{old_shape_name}__patched_{patched_count}"
        add_initializer(patched, new_shape_name, new_shape, dtype=old_dtype)
        pn.input[1] = new_shape_name

        patched_count += 1
        print(
            f"[PATCH] node={pn.name} "
            f"old_shape_input={old_shape_name} "
            f"new_shape_input={new_shape_name} "
            f"{out_shape1} -> {new_shape}"
        )
import copy
import onnx
import numpy as np
from onnxsim import simplify
from onnx import numpy_helper


def simplify_model(model_path, input_shapes):
    model = onnx.load(model_path)
    sim_model, check = simplify(model, overwrite_input_shapes=input_shapes)
    assert check, "onnxsim simplify failed"
    sim_model = onnx.shape_inference.infer_shapes(sim_model)
    return sim_model


def get_init_array(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def get_tensor_shape(model, tensor_name):
    infos = list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
    for v in infos:
        if v.name == tensor_name:
            dims = []
            for d in v.type.tensor_type.shape.dim:
                dims.append(int(d.dim_value) if d.dim_value > 0 else None)
            return dims
    return None


def get_reshape_nodes(model):
    return {
        n.output[0]: n
        for n in model.graph.node
        if n.op_type == "Reshape" and len(n.input) >= 2 and len(n.output) >= 1
    }


def add_initializer(model, name, arr, dtype=np.int64):
    init = numpy_helper.from_array(np.asarray(arr, dtype=dtype), name)
    model.graph.initializer.append(init)


def patch_dynamic_reshape(model_path, shape1, shape2, out_path):
    if len(shape1) != 1 or len(shape2) != 1:
        raise ValueError("only single-input model is supported")

    input_name1, dims1 = next(iter(shape1.items()))
    input_name2, dims2 = next(iter(shape2.items()))
    if input_name1 != input_name2:
        raise ValueError("shape1 and shape2 must use the same input name")

    batch1 = int(dims1[0])
    batch2 = int(dims2[0])

    sim1 = simplify_model(model_path, shape1)
    sim2 = simplify_model(model_path, shape2)
    patched = copy.deepcopy(sim1)

    r1 = get_reshape_nodes(sim1)
    r2 = get_reshape_nodes(sim2)

    patched_count = 0

    # ▒~\▒ patched 裡建 output_name -> node ▒~Z~D▒~M▒~G~I▒~L▒~V▒便▒~T▒ input[1]
    patched_r = get_reshape_nodes(patched)

    for out_name, n1 in r1.items():
        n2 = r2.get(out_name)
        pn = patched_r.get(out_name)
        if n2 is None or pn is None:
            continue

        out_shape1 = get_tensor_shape(sim1, n1.output[0])
        out_shape2 = get_tensor_shape(sim2, n2.output[0])
        if out_shape1 is None or out_shape2 is None:
            continue
        if len(out_shape1) != len(out_shape2):
            continue

        diff_idx = [i for i, (a, b) in enumerate(zip(out_shape1, out_shape2)) if a != b]
        if len(diff_idx) != 1:
            continue

        dyn_idx = diff_idx[0]
        a = out_shape1[dyn_idx]
        b = out_shape2[dyn_idx]
        if a is None or b is None:
            continue

        # ▒~O▒▒~N▒▒~O~W▒~R~L batch ▒~H~P▒~T▒~K▒~J▒~L~V▒~Z~D維度
        if b * batch1 != a * batch2:
            continue

        # ▒~T▒ Reshape ▒~Z~D output shape ▒~U▒ base
        new_shape = list(out_shape1)
        new_shape[dyn_idx] = -1

        old_shape_name = pn.input[1]
        old_arr = get_init_array(patched, old_shape_name)
        old_dtype = old_arr.dtype if old_arr is not None else np.int64

        # ▒~W~\▒~M▒▒~Z▒~O▒~@~K reshape ▒~C▒建▒~K▒~G▒己▒~Z~D shape initializer▒~L▒~A▒▒~E~M shared initializer 被▒~F寫
        new_shape_name = f"{old_shape_name}__patched_{patched_count}"
        add_initializer(patched, new_shape_name, new_shape, dtype=old_dtype)
        pn.input[1] = new_shape_name

        patched_count += 1
        print(
            f"[PATCH] node={pn.name} "
            f"old_shape_input={old_shape_name} "
            f"new_shape_input={new_shape_name} "
            f"{out_shape1} -> {new_shape}"
        )

    patched = onnx.shape_inference.infer_shapes(patched)
    onnx.save(patched, out_path)
    print(f"done, patched {patched_count} reshape nodes -> {out_path}")
def patch_dynamic_reshape(model_path, shape1, shape2, out_path):
    if set(shape1.keys()) != set(shape2.keys()):
        raise ValueError("shape1 and shape2 must use the same input names")

    input_names = list(shape1.keys())
    if not input_names:
        raise ValueError("empty input shapes")

    # 檢查所有 input 的 batch 都一致
    batch1_set = {int(shape1[name][0]) for name in input_names}
    batch2_set = {int(shape2[name][0]) for name in input_names}

    if len(batch1_set) != 1 or len(batch2_set) != 1:
        raise ValueError("all inputs must share the same batch size")

    batch1 = next(iter(batch1_set))
    batch2 = next(iter(batch2_set))

    sim1 = simplify_model(model_path, shape1)
    sim2 = simplify_model(model_path, shape2)
    patched = copy.deepcopy(sim1)

    r1 = get_reshape_nodes(sim1)
    r2 = get_reshape_nodes(sim2)

    patched_count = 0
    patched_r = get_reshape_nodes(patched)

    for out_name, n1 in r1.items():
        n2 = r2.get(out_name)
        pn = patched_r.get(out_name)
        if n2 is None or pn is None:
            continue

        out_shape1 = get_tensor_shape(sim1, n1.output[0])
        out_shape2 = get_tensor_shape(sim2, n2.output[0])
        if out_shape1 is None or out_shape2 is None:
            continue
        if len(out_shape1) != len(out_shape2):
            continue

        diff_idx = [i for i, (a, b) in enumerate(zip(out_shape1, out_shape2)) if a != b]
        if len(diff_idx) != 1:
            continue

        dyn_idx = diff_idx[0]
        a = out_shape1[dyn_idx]
        b = out_shape2[dyn_idx]
        if a is None or b is None:
            continue

        # 確認這一維和 batch 成正比
        if b * batch1 != a * batch2:
            continue

        new_shape = list(out_shape1)
        new_shape[dyn_idx] = -1

        old_shape_name = pn.input[1]
        old_arr = get_init_array(patched, old_shape_name)
        old_dtype = old_arr.dtype if old_arr is not None else np.int64

        new_shape_name = f"{old_shape_name}__patched_{patched_count}"
        add_initializer(patched, new_shape_name, new_shape, dtype=old_dtype)
        pn.input[1] = new_shape_name

        patched_count += 1
        print(
            f"[PATCH] node={pn.name} "
            f"old_shape_input={old_shape_name} "
            f"new_shape_input={new_shape_name} "
            f"{out_shape1} -> {new_shape}"
        )

    patched = onnx.shape_inference.infer_shapes(patched)
    onnx.save(patched, out_path)
    print(f"done, patched {patched_count} reshape nodes -> {out_path}")



shape1={'input_ids':[1,384],'attention_mask':[1,384],'token_type_ids':[1,384]}
shape2={'input_ids':[10,384],'attention_mask':[10,384],'token_type_ids':[10,384]}
patch_dynamic_reshape('output_squadv2/model.onnx', shape1, shape2, 'model_squadv2.onnx')
