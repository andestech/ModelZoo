#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:42:17 2021

@author: aitester
"""
import torch
import torch.fx as fx
import copy
import operator
import numpy as np



def add_idd(model: torch.nn.Module) -> torch.nn.Module:
    model = copy.deepcopy(model) 
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    new_graph = fx.Graph()
    env = {}
    
    record_node = None
    pre_node_name = None
    count = 0
    for node in fx_model.graph.nodes:
        if (str(node.target) in ['<built-in function add>','<built-in function mul>']) or (node.target == torch.cat):
            if record_node in node.args:
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                env[node.name] = new_node
                pre_node_name = node.name
                idd = {}
                idd[torch.nn.Identity] = pre_node_name + '_insert_idd'
                setattr(model, pre_node_name + '_insert_idd', torch.nn.Identity())
                new_node = new_graph.call_module(idd[torch.nn.Identity], args=(env[node.name],))
                env[node.args[0].name] = new_node
            else:
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                env[node.name] = new_node
                pre_node_name = node.name
                idd = {}
                idd[torch.nn.Identity] = pre_node_name + '_insert_idd'
                setattr(model, pre_node_name + '_insert_idd', torch.nn.Identity())
                new_node = new_graph.call_module(idd[torch.nn.Identity], args=(env[node.name],))
                if node.target == torch.cat:
                    env[node.args[0][0].name] = new_node
                else:
                    env[node.args[0].name] = new_node
            record_node = node
            count += 1
        elif record_node in node.args:
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
        else:
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node 
    
    record_node = None
    successor_record_node = None
    for node in new_graph.nodes:
        #print(node)
        if (str(node.target) in ['<built-in function add>','<built-in function mul>']) or (node.target == torch.cat):
            if record_node in node.args:
                new_args = ()
                for ii in range(len(node.args)):
                    if record_node == node.args[ii]:
                        new_args = new_args + (successor_record_node,)
                    else:
                        new_args = new_args + (node.args[ii],)
                
                node.args = new_args
                record_node = node
            else:
                record_node = node
            count += 1
        elif 'insert_idd' in str(node.target):
            successor_record_node = node
        elif record_node in node.args:
            new_args = ()
            for ii in range(len(node.args)):
                if record_node == node.args[ii]:
                    new_args = new_args + (successor_record_node,)
                else:
                    new_args = new_args + (node.args[ii],)
            node.args = new_args
        else:
            continue        
    
    print("Insertion of Identity_layers")
    new_graph.lint()
    return fx.GraphModule(model, new_graph)

def test_fromONNX(model: torch.nn.Module) -> torch.nn.Module:
    #model = copy.deepcopy(model) 
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    new_graph = fx.Graph()
    env = {}

    input_node = None
    target_node = None
    for node in fx_model.graph.nodes:
        if 'input_1' in node.name:
            input_node = node
            #print(node)
        elif 'mul' in node.name:
            #print(node.args)
            new_args = ()
            new_args = new_args + (node.args[0],)
            new_args = new_args + (node.args[0],)
            node.args = new_args
            #node = None
        elif 'sub' in node.name:
            new_args = ()
            new_args = new_args + (node.args[1],)
            new_args = new_args + (node.args[1],)
            node.args = new_args
            #print(node)
        elif 'permute' in node.name:
            target_node = node
        elif 'conv_0' in node.name:
            if target_node in node.args:
                #print(node)
                new_args = ()
                for ii in range(len(node.args)):
                    if target_node == node.args[ii]:
                        new_args = new_args + (input_node,)
                node.args = new_args
    for node in fx_model.graph.nodes:
        if 'permute' in node.name:
            if node.name == 'permute':
                fx_model.graph.erase_node(node)
    for node in fx_model.graph.nodes:
        if 'sub' in node.name:
            fx_model.graph.erase_node(node)
    for node in fx_model.graph.nodes:
        if 'onnx_initializer_1' in node.name:
            fx_model.graph.erase_node(node)
    for node in fx_model.graph.nodes:
        if 'mul' in node.name:
            if 'mat' in node.name:
                continue
            else:
                fx_model.graph.erase_node(node)
    for node in fx_model.graph.nodes:
        if 'onnx_initializer_0' in node.name:
            fx_model.graph.erase_node(node)
    print("ONNX_model pre_processing")
    print("Pass torch.fx")
    fx_model.graph.lint()
    return fx.GraphModule(model, fx_model.graph)

def see_fx_graph(model: torch.nn.Module) -> torch.nn.Module:
    #model = copy.deepcopy(model) 
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    new_graph = fx.Graph()
    env = {}
    
    #record_node = None
    #pre_node_name = None
    count = 0
    for node in fx_model.graph.nodes:
        count += 1
        new_node = new_graph.node_copy(node, lambda x: env[x.name])
        env[node.name] = new_node
    print("torch.fx forwards " + str(count) + ' nodes!!')
    print("Pass torch.fx")
    new_graph.lint()
    return fx.GraphModule(model, fx_model.graph)

def dag_process(model: torch.nn.Module) -> torch.nn.Module:
    #model = copy.deepcopy(model) 
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    new_graph = fx.Graph()
    env = {}
    prepare_dict = {}
    first_gemm = False
    
    record_node = None
    for node in fx_model.graph.nodes:
        if (not first_gemm):
            if ('pad' in node.name) or ('input' in node.name):
                if ('input' in node.name):
                    record_node = node
                else:
                    new_args = ()
                    new_args = new_args + (record_node,)
                    node.args = new_args
                #    print(node, node.op, node.target, node.kwargs)
                prepare_dict[node.name] = 1
            elif (('conv' in node.name) or ('gemm' in node.name) or ('matmul' in node.name)):
                prepare_dict[node.name] = 0
                first_gemm = True
            else:
                prepare_dict[node.name] = -1
        else:
            prepare_dict[node.name] = 1
    
    record_node = None
    for node in fx_model.graph.nodes:
        #print(node.name, node.args)
        if 'input' in node.name:
            record_node = node
        if prepare_dict[node.name] == -1:
            if ('permute' in node.name) or ('reshape' in node.name) or ('to' in node.name) :
                new_args = ()
                new_args = new_args + (record_node,)
                node.args = new_args
            elif ('mul' in node.name) or ('sub' in node.name) or ('add' in node.name) or ('div' in node.name):
                new_args = ()
                new_args = new_args + (record_node,)
                new_args = new_args + (record_node,)
                node.args = new_args
            elif ('getitem' in node.name) or ('resize' in node.name) or ('cat' in node.name):
                new_args = ()
                new_args = new_args + (record_node,)
                new_args = new_args + (record_node,)
                node.args = new_args
        elif prepare_dict[node.name]==0:
            new_args = ()
            need_new_args = False
            for p_node in node.args:
                if prepare_dict[p_node.name] == -1:
                    need_new_args = True
                    new_args = new_args + p_node.args
                else:
                    new_args = new_args + (p_node,)
            if need_new_args:
                node.args = new_args
        
    #print()            
    for node in fx_model.graph.nodes:
        if prepare_dict[node.name] in [0,1]:        
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
        else:
            #print(node.name)
            fx_model.graph.erase_node(node)
            #print('done')
        if 'output' in node.name:
            new_args = ()
            if isinstance(node.args[0],list):
                for i in range(0,len(node.args[0])):
                    new_args = new_args + (node.args[0][i],)    
                node.args = new_args
        #if str(node.target) =='<built-in function mul>':
        #    node.target = operator.mul
    
    prepare_dict = None
    del prepare_dict
    print("DAG_Input_Processing!!")
    
    new_graph.lint()
    return fx.GraphModule(model, fx_model.graph)


"""
need another function to add conv and solve the concat issue
"""
#def add_conv():
#   print()
#   return 0