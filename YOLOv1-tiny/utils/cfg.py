__all__ = ["parse_cfg", "print_cfg"]


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list

    """
    with open(cfgfile, "r") as file:
        lines = file.read().split("\n")  # store the lines in a list
        lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
        lines = [x for x in lines if x[0] != "#"]
        lines = [x.rstrip().lstrip() for x in lines]

        block = {}
        blocks = []
        for line in lines:
            if line[0] == "[":  # This marks the start of a new block
                if len(block) != 0:
                    blocks.append(block)
                    block = {}
                block["type"] = line[1:-1].rstrip()
                if block["type"] == "convolutional":
                    block["batch_normalize"] = 0
            else:
                key, value = line.split("=")
                block[key.rstrip()] = value.lstrip()
        blocks.append(block)
        file.close()
        return blocks


def print_cfg(blocks, input_shape: list = [416, 416]):
    print("layer     filters    size              input                output")
    prev_width = input_shape[1]
    prev_height = input_shape[0]
    prev_filters = 3
    out_filters = []
    out_widths = []
    out_heights = []
    ind = -2
    for block in blocks:
        ind = ind + 1
        if block["type"] == "net":
            prev_width = int(block["width"])
            prev_height = int(block["height"])
            continue
        elif block["type"] == "convolutional":
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            is_pad = int(block["pad"])
            pad = (kernel_size - 1) / 2 if is_pad else 0
            width = (prev_width + 2 * pad - kernel_size) / stride + 1
            height = (prev_height + 2 * pad - kernel_size) / stride + 1
            print(
                "%3d %-8s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d"
                % (
                    ind,
                    "conv",
                    filters,
                    kernel_size,
                    kernel_size,
                    stride,
                    prev_width,
                    prev_height,
                    prev_filters,
                    width,
                    height,
                    filters,
                )
            )
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block["type"] == "maxpool":
            pool_size = int(block["size"])
            stride = int(block["stride"])
            width = prev_width / stride
            height = prev_height / stride
            print(
                "%3d %-8s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d"
                % (
                    ind,
                    "max",
                    pool_size,
                    pool_size,
                    stride,
                    prev_width,
                    prev_height,
                    prev_filters,
                    width,
                    height,
                    filters,
                )
            )
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block["type"] == "avgpool":
            width = 1
            height = 1
            print("%3d %-8s                   %3d x %3d x%4d   ->  %3d" % (ind, "avg", prev_width, prev_height, prev_filters, prev_filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block["type"] == "softmax":
            print("%3d %-8s                                    ->  %3d" % (ind, "softmax", prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block["type"] == "cost":
            print("%3d %-8s                                     ->  %3d" % (ind, "cost", prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block["type"] == "reorg":
            stride = int(block["stride"])
            filters = stride * stride * prev_filters
            width = prev_width / stride
            height = prev_height / stride
            print(
                "%3d %-8s             / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d"
                % (
                    ind,
                    "reorg",
                    stride,
                    prev_width,
                    prev_height,
                    prev_filters,
                    width,
                    height,
                    filters,
                )
            )
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block["type"] == "route":
            layers = block["layers"].split(",")
            layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
            if len(layers) == 1:
                print("%3d %-8s %d" % (ind, "route", layers[0]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                print("%3d %-8s %d %d" % (ind, "route", layers[0], layers[1]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert prev_width == out_widths[layers[1]]
                assert prev_height == out_heights[layers[1]]
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block["type"] == "shortcut":
            from_id = int(block["from"])
            from_id = from_id if from_id > 0 else from_id + ind
            print("%3d %-8s %d" % (ind, "shortcut", from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block["type"] == "connected":
            filters = int(block["output"])
            print("%3d %-8s                            %d  ->  %3d" % (ind, "connected", prev_filters, filters))
            prev_filters = filters
            out_widths.append(1)
            out_heights.append(1)
            out_filters.append(prev_filters)
        elif block["type"] == "upsample":
            stride = int(block["stride"])
            width = prev_width * stride
            height = prev_height * stride
            print(
                "%3d %-8s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d"
                % (
                    ind,
                    "upsample",
                    pool_size,
                    pool_size,
                    stride,
                    prev_width,
                    prev_height,
                    prev_filters,
                    width,
                    height,
                    filters,
                )
            )
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block["type"] == "region":
            print("%3d %-8s" % (ind, "detection"))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block["type"] == "yolo":
            print("%3d %-8s" % (ind, "detection"))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        else:
            print("unknown type %s" % (block["type"]))


if __name__ == "__main__":
    blocks = parse_cfg("weights/yolov3-tiny.cfg")
    print_cfg(blocks)
